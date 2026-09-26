import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    ClassifierTwoSampleTestHarness,
    TSTRHarness,
    ClinicalConstraintEvaluator,
)
from acfm_net.evaluation.privacy_eval import (
    _encode_frames,
    TAPASClosestDistanceMIA,
    TAPASShadowModelMIA,
    DOMIASMIA,
    DCRDistributionEvaluator,
)


def run_phase3_evaluation(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    df_test: pd.DataFrame,
    df_syn: pd.DataFrame,
    target_col: str,
    schema_bounds: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Phase 3 Unified Evaluation Harness.
    Runs all required Phase 3 Fidelity and Privacy evaluation modules using
    fixed train/holdout/test splits with identical preprocessing.
    """
    t0 = time.time()
    results: Dict[str, Any] = {}

    print("\n================================================================================")
    print(" PHASE 3 EVALUATION HARNESS EXECUTION")
    print("================================================================================")
    print(f" [Splits] Train: {len(df_train)} rows | Holdout: {len(df_holdout)} rows | Test: {len(df_test)} rows | Syn: {len(df_syn)} rows")

    # -------------------------------------------------------------------------
    # 1. FIDELITY EVALUATION (WITH REAL-VS-REAL REFERENCE BASELINES)
    # -------------------------------------------------------------------------
    print("\n--- 1. Evaluating Fidelity Modules (Real-vs-Syn and Real-vs-Real References) ---")
    # 1-way & 2-way marginal errors
    marginal_eval = MarginalErrorEvaluator()
    err_1way_syn = marginal_eval.evaluate_1way(df_test, df_syn)
    err_1way_real = marginal_eval.evaluate_1way(df_test, df_train)
    err_2way_syn = marginal_eval.evaluate_2way(df_test, df_syn, seed=seed)
    err_2way_real = marginal_eval.evaluate_2way(df_test, df_train, seed=seed)
    print(f"  [1-Way Marginal Error] Real-vs-Syn TVD:  {err_1way_syn['mean_1way_marginal_error_tvd']:.4f} across {err_1way_syn['num_features_evaluated']} features")
    print(f"  [1-Way Marginal Error] Real-vs-Real Ref: {err_1way_real['mean_1way_marginal_error_tvd']:.4f} (Train vs Holdout)")
    print(f"  [2-Way Marginal Error] Real-vs-Syn TVD:  {err_2way_syn['mean_2way_marginal_error_tvd']:.4f} across {err_2way_syn['num_pairs_evaluated']} feature pairs")
    print(f"  [2-Way Marginal Error] Real-vs-Real Ref: {err_2way_real['mean_2way_marginal_error_tvd']:.4f} (Train vs Holdout)")

    # Correlation error
    corr_eval = CorrelationErrorEvaluator()
    corr_res_syn = corr_eval.evaluate(df_test, df_syn)
    corr_res_real = corr_eval.evaluate(df_test, df_train)
    print(f"  [Correlation Error] Real-vs-Syn:  Pearson Frob={corr_res_syn['pearson_frobenius_error']:.4f} | Off-diag MAE={corr_res_syn['pearson_offdiag_mae']:.4f}")
    print(f"  [Correlation Error] Real-vs-Real: Pearson Frob={corr_res_real['pearson_frobenius_error']:.4f} | Off-diag MAE={corr_res_real['pearson_offdiag_mae']:.4f}")
    print(f"  [Rank Corr Error]   Real-vs-Syn:  Spearman Frob={corr_res_syn['spearman_frobenius_error']:.4f} | Off-diag MAE={corr_res_syn['spearman_offdiag_mae']:.4f}")
    print(f"  [Rank Corr Error]   Real-vs-Real: Spearman Frob={corr_res_real['spearman_frobenius_error']:.4f} | Off-diag MAE={corr_res_real['spearman_offdiag_mae']:.4f}")

    # C2ST (Classifier Two-Sample Test)
    c2st_eval = ClassifierTwoSampleTestHarness()
    c2st_res_syn = c2st_eval.evaluate(df_test, df_syn, seed=seed)
    c2st_res_real = c2st_eval.evaluate(df_test, df_train, seed=seed)
    print(f"  [C2ST Indistinguishability] Real-vs-Syn:  AUC={c2st_res_syn['c2st_auc']:.4f} (95% CI {c2st_res_syn['c2st_95_ci']}) | SE={c2st_res_syn['c2st_se']:.4f}")
    print(f"  [C2ST Indistinguishability] Real-vs-Real: AUC={c2st_res_real['c2st_auc']:.4f} (95% CI {c2st_res_real['c2st_95_ci']}) | SE={c2st_res_real['c2st_se']:.4f} (should be ~0.50)")

    # TSTR vs TRTR
    tstr_eval = TSTRHarness()
    tstr_res = tstr_eval.evaluate(df_train, df_test, df_syn, target_col=target_col, seed=seed)
    if tstr_res.get("task") == "classification":
        print(f"  [TSTR Utility] AUROC: {tstr_res['tstr']['auroc']:.4f} (TRTR baseline: {tstr_res['trtr_baseline']['auroc']:.4f}) | Utility Ratio: {tstr_res['utility_ratio_auroc']:.4f}")
        print(f"  [TSTR Utility] AUPRC: {tstr_res['tstr']['auprc']:.4f} (TRTR baseline: {tstr_res['trtr_baseline']['auprc']:.4f}) | Utility Ratio: {tstr_res['utility_ratio_auprc']:.4f}")
        print(f"  [TSTR Utility] F1 (weighted): {tstr_res['tstr']['f1_weighted']:.4f} (TRTR baseline: {tstr_res['trtr_baseline']['f1_weighted']:.4f}) | Utility Ratio: {tstr_res['utility_ratio_f1']:.4f}")
    else:
        print(f"  [TSTR Utility] RMSE: {tstr_res['tstr']['rmse']:.4f} (TRTR baseline: {tstr_res['trtr_baseline']['rmse']:.4f}) | RMSE Gap: {tstr_res['rmse_gap']:.4f}")

    # Clinical constraint violation rate
    clin_eval = ClinicalConstraintEvaluator()
    clin_res_syn = clin_eval.evaluate(df_syn, schema_bounds=schema_bounds)
    clin_res_real = clin_eval.evaluate(df_holdout, schema_bounds=schema_bounds)
    print(f"  [Clinical Constraints] Real-vs-Syn:  Pass Rate={clin_res_syn['overall_record_pass_rate']:.2%} | Violation Rate={clin_res_syn['overall_record_violation_rate']:.2%} ({clin_res_syn['violating_records_count']}/{clin_res_syn['total_records_evaluated']} records)")
    print(f"  [Clinical Constraints] Real Holdout: Pass Rate={clin_res_real['overall_record_pass_rate']:.2%} | Violation Rate={clin_res_real['overall_record_violation_rate']:.2%} ({clin_res_real['violating_records_count']}/{clin_res_real['total_records_evaluated']} records)")
    if clin_res_syn["violating_records_count"] > 0:
        print("  [Clinical Constraints Failure Diagnosis - Syn]:")
        for rk, rv in clin_res_syn.get("rule_breakdown", {}).items():
            if rv.get("violation_count", 0) > 0:
                print(f"    - {rk}: {rv['violation_count']} violations ({rv['violation_rate']:.2%})")
    if clin_res_real["violating_records_count"] > 0:
        print("  [Clinical Constraints Failure Diagnosis - Real Holdout]:")
        for rk, rv in clin_res_real.get("rule_breakdown", {}).items():
            if rv.get("violation_count", 0) > 0:
                print(f"    - {rk}: {rv['violation_count']} violations ({rv['violation_rate']:.2%})")

    results["fidelity"] = {
        "one_way_marginal_error": err_1way_syn,
        "one_way_marginal_error_real_ref": err_1way_real,
        "two_way_marginal_error": err_2way_syn,
        "two_way_marginal_error_real_ref": err_2way_real,
        "correlation_error": corr_res_syn,
        "correlation_error_real_ref": corr_res_real,
        "c2st": c2st_res_syn,
        "c2st_real_ref": c2st_res_real,
        "tstr": tstr_res,
        "clinical_constraint_violations": clin_res_syn,
        "clinical_constraint_violations_real_ref": clin_res_real,
    }

    # -------------------------------------------------------------------------
    # 2. PRIVACY EVALUATION (DISJOINT HOLDOUT SET)
    # -------------------------------------------------------------------------
    print("\n--- 2. Evaluating Privacy Modules (Disjoint Holdout Set) ---")
    X_train, X_holdout, X_syn = _encode_frames(df_train, df_holdout, df_syn)

    # TAPAS Closest-Distance MIA
    tapas_cd = TAPASClosestDistanceMIA()
    tapas_cd_res = tapas_cd.evaluate(X_train, X_holdout, X_syn, seed=seed)
    print(f"  [TAPAS Closest-Distance MIA] AUC: {tapas_cd_res['mia_auc']:.4f} (95% CI {tapas_cd_res['mia_95_ci']}) | SE: {tapas_cd_res['mia_se']:.4f}")

    # TAPAS Shadow-Model MIA
    tapas_sm = TAPASShadowModelMIA()
    tapas_sm_res = tapas_sm.evaluate(X_train, X_holdout, X_syn, seed=seed)
    print(f"  [TAPAS Shadow-Model MIA] AUC: {tapas_sm_res['mia_auc']:.4f} (95% CI {tapas_sm_res['mia_95_ci']}) | SE: {tapas_sm_res['mia_se']:.4f}")

    # DOMIAS Density-Ratio MIA
    domias = DOMIASMIA()
    domias_res = domias.evaluate(X_train, X_holdout, X_syn, seed=seed)
    print(f"  [DOMIAS Density-Ratio MIA] AUC: {domias_res['domias_auc']:.4f} (95% CI {domias_res['domias_95_ci']}) | SE: {domias_res['domias_se']:.4f}")

    # DCR Distributions
    dcr_eval = DCRDistributionEvaluator()
    dcr_res = dcr_eval.evaluate(X_train, X_holdout, X_syn)
    print(f"  [DCR Distributions] Space: {dcr_res['space']} ({dcr_res['units']})")
    print(f"  [DCR Train Percentiles]   p5={dcr_res['dcr_train_distribution']['p5']}, p25={dcr_res['dcr_train_distribution']['p25']}, median={dcr_res['median_dcr_train']}, p75={dcr_res['dcr_train_distribution']['p75']}, p95={dcr_res['dcr_train_distribution']['p95']}")
    print(f"  [DCR Holdout Percentiles] p5={dcr_res['dcr_holdout_distribution']['p5']}, p25={dcr_res['dcr_holdout_distribution']['p25']}, median={dcr_res['median_dcr_holdout']}, p75={dcr_res['dcr_holdout_distribution']['p75']}, p95={dcr_res['dcr_holdout_distribution']['p95']}")
    print(f"  [DCR Median Ratio (Train / Holdout)] {dcr_res['train_to_holdout_dcr_ratio']:.4f}")
    print(f"  [DCR Interpretation]: {dcr_res['interpretation']}")

    results["privacy"] = {
        "tapas_closest_distance_mia": tapas_cd_res,
        "tapas_shadow_model_mia": tapas_sm_res,
        "domias_mia": domias_res,
        "dcr_distribution": dcr_res,
    }

    results["runtime_seconds"] = round(time.time() - t0, 3)
    print(f"\n[Phase 3 Complete] Total evaluation runtime: {results['runtime_seconds']}s")
    print("================================================================================\n")
    return results
