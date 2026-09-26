"""
ACFM-Net v5 — Final Evaluation Module Suite
Spec: Stage 9 / Final Evaluation Module (25-Point Comprehensive Audit)

Runs all evaluators:
  - Statistical: KS, Wasserstein, Variance/Covariance, Tail Coverage, Correlation (Heatmap data), JSD
  - Utility: TSTR, TRTS across multi-models (LR, RF, GBDT) with 5-fold CV & 95% CIs
  - Privacy: MIA (AUC), Attribute Inference Attack (Delta), Re-ID Risk, DP Epsilon/Delta
  - Clinical: 22 Range rules, 2 Logical rules, Clinical Dependency Validation (Glucose->Outcome, etc.), Clinical Consistency Score
  - Fairness: Demographic Parity Gap across sensitive attributes
  - Overall Composite Score: 5-pillar weighted score (0-100) + 95% CIs
  - Stage Proof Verifier: Empirical evidence report for Stages 0 to 8
"""

from typing import Dict, Any, Optional
import pandas as pd

from .statistical import (
    KSTestEvaluator,
    WassersteinEvaluator,
    VarianceCovarianceEvaluator,
    CorrelationSimilarityEvaluator,
    JSDivergenceEvaluator,
)
from .utility  import TSTREvaluator, TRTSEvaluator
from .privacy  import MembershipInferenceAttack, AttributeInferenceAttack, DuplicateMemorizationChecker
from .clinical import ClinicalConstraintAuditor
from .fairness import FairnessEvaluator
from .report   import EvaluationReportBuilder
from .composite_score import compute_composite_score
from .module_proof import StageProofVerifier
from .harness import run_phase3_evaluation
from .fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    ClassifierTwoSampleTestHarness,
    TSTRHarness,
    ClinicalConstraintEvaluator,
)
from .privacy_eval import (
    TAPASClosestDistanceMIA,
    TAPASShadowModelMIA,
    DOMIASMIA,
    DCRDistributionEvaluator,
)


def run_full_evaluation(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    target_col: Optional[str] = None,
    sensitive_col: Optional[str] = None,
    eps_used: float = 0.0,
    target_epsilon: float = 1.0,
    dataset_name: str = "Dataset",
    pipeline_report: Optional[Dict[str, Any]] = None,
    privacy_verified: bool = False,
    privacy_scope: str = "training_only",
) -> Dict[str, Any]:
    """
    Run all ACFM-Net v5 evaluation modules and return structured 25-point audit results dict.
    """
    results: Dict[str, Any] = {}

    # --- 1. Statistical Fidelity ---
    ks_eval    = KSTestEvaluator()
    ws_eval    = WassersteinEvaluator()
    vc_eval    = VarianceCovarianceEvaluator()
    corr_eval  = CorrelationSimilarityEvaluator()
    jsd_eval   = JSDivergenceEvaluator()

    results["statistical"] = {
        "ks_test":              ks_eval.evaluate(df_real, df_syn),
        "wasserstein":          ws_eval.evaluate(df_real, df_syn),
        "variance_covariance":  vc_eval.evaluate(df_real, df_syn),
        "correlation":          corr_eval.evaluate(df_real, df_syn),
        "jsd":                  jsd_eval.evaluate(df_real, df_syn),
    }

    # --- 2. Utility ---
    tstr_eval = TSTREvaluator()
    trts_eval = TRTSEvaluator()
    results["utility"] = {
        "tstr": tstr_eval.evaluate(df_real, df_syn, target_col),
        "trts": trts_eval.evaluate(df_real, df_syn, target_col),
    }

    # --- 3. Privacy & Memorization ---
    mia_eval  = MembershipInferenceAttack()
    attr_eval = AttributeInferenceAttack()
    dcr_eval  = DuplicateMemorizationChecker()
    results["privacy"] = {
        "mia":                     mia_eval.evaluate(df_real, df_syn),
        "attribute_ia":            attr_eval.evaluate(df_syn, sensitive_col),
        "duplicate_memorization":  dcr_eval.evaluate(df_real, df_syn),
        "differential_privacy": {
            "eps_used":    round(eps_used, 6),
            "target_eps":  round(target_epsilon, 4),
            "satisfied":   bool(privacy_verified and eps_used <= target_epsilon),
            "verified":    bool(privacy_verified),
            "status":      "SATISFIED" if privacy_verified and eps_used <= target_epsilon else "UNVERIFIED",
            "scope":       privacy_scope,
            "delta":       1e-5,
        }
    }

    # --- 4. Clinical ---
    clin_eval = ClinicalConstraintAuditor()
    results["clinical"] = clin_eval.audit(df_syn, df_real)

    # --- 5. Fairness ---
    fair_eval = FairnessEvaluator()
    results["fairness"] = fair_eval.evaluate(df_real, df_syn, target_col)

    # --- 6. Composite Score & 95% CIs ---
    results["overall_composite_score"] = compute_composite_score(results)

    # --- 7. Stage Proof Verifier (S0-S8) ---
    stage_verifier = StageProofVerifier()
    results["framework_stage_proofs"] = stage_verifier.verify_pipeline_proof(
        pipeline_report or {}, results
    )

    # --- Metadata ---
    results["metadata"] = {
        "dataset_name":      dataset_name,
        "n_real":            len(df_real),
        "n_syn":             len(df_syn),
        "features":          list(df_real.columns),
        "target_col":        target_col,
        "sensitive_col":     sensitive_col,
        "eps_used":          round(eps_used, 6),
        "target_epsilon":    round(target_epsilon, 4),
    }

    return results


def build_pdf_report(
    eval_results: Dict[str, Any],
    dataset_name: str = "Dataset",
    eps_used: float = 0.0,
    target_epsilon: float = 1.0,
) -> bytes:
    """Generate a PDF report from evaluation results dict. Returns raw bytes."""
    meta = eval_results.get("metadata", {})
    dp = eval_results.get("privacy", {}).get("differential_privacy", {})
    builder = EvaluationReportBuilder()
    return builder.build_pdf(
        eval_results=eval_results,
        dataset_name=meta.get("dataset_name", dataset_name),
        n_real=meta.get("n_real", 0),
        n_syn=meta.get("n_syn", 0),
        eps_used=meta.get("eps_used", eps_used),
        target_epsilon=meta.get("target_epsilon", target_epsilon),
        privacy_verified=bool(dp.get("verified", False)),
    )


__all__ = [
    "run_full_evaluation",
    "build_pdf_report",
    "compute_composite_score",
    "StageProofVerifier",
    "KSTestEvaluator",
    "WassersteinEvaluator",
    "VarianceCovarianceEvaluator",
    "CorrelationSimilarityEvaluator",
    "JSDivergenceEvaluator",
    "TSTREvaluator",
    "TRTSEvaluator",
    "MembershipInferenceAttack",
    "AttributeInferenceAttack",
    "ClinicalConstraintAuditor",
    "FairnessEvaluator",
    "EvaluationReportBuilder",
]
