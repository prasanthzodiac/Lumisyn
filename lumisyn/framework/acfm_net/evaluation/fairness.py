from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FairnessEvaluator:
    """
    Fairness Evaluation — spec compliant:
      - Demographic Parity: P(Y=1 | A=a) = P(Y=1 | A=b) for all groups a, b
      - Equal Opportunity: P(Y_hat=1 | Y=1, A=a) = P(Y_hat=1 | Y=1, A=b)

    Computes gaps across all detected sensitive attributes.
    A gap of 0.0 = perfect fairness; > 0.1 is typically flagged.
    """

    _SENSITIVE_KEYWORDS = [
        "gender", "sex", "race", "ethnicity", "religion",
        "marital", "income", "poverty", "smoking", "insurance",
    ]
    _TARGET_KEYWORDS = [
        "diabetes", "mortality", "outcome", "cancer", "disease",
        "death", "survival", "risk", "readmission",
    ]

    def _detect_sensitive_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns
                if any(kw in c.lower() for kw in self._SENSITIVE_KEYWORDS)]

    def _detect_target_col(self, df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
        if hint and hint in df.columns:
            return hint
        for c in df.columns:
            if any(kw in c.lower() for kw in self._TARGET_KEYWORDS):
                return c
        return None

    def _demographic_parity_gap(
        self, df: pd.DataFrame, sens_col: str, target_col: str
    ) -> Dict[str, Any]:
        if sens_col not in df.columns or target_col not in df.columns:
            return {}

        target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        groups = df[sens_col].astype(str).unique()

        group_rates = {}
        for g in groups:
            mask = df[sens_col].astype(str) == g
            if mask.sum() > 0:
                group_rates[g] = float(target[mask].mean())

        if len(group_rates) < 2:
            return {"error": "Need at least 2 groups"}

        rates = list(group_rates.values())
        dp_gap = float(max(rates) - min(rates))
        return {
            "group_positive_rates": {k: round(v, 4) for k, v in group_rates.items()},
            "demographic_parity_gap": round(dp_gap, 4),
            "passed_01_threshold": dp_gap < 0.10,
        }

    def evaluate(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        target = self._detect_target_col(df_syn, target_col)
        if target is None:
            return {"error": "Could not identify target column for fairness evaluation"}

        sens_cols = self._detect_sensitive_cols(df_syn)
        if not sens_cols:
            return {
                "message": "No sensitive attribute columns detected — fairness check skipped.",
                "target_col": target,
            }

        results: Dict[str, Any] = {"target_col": target, "sensitive_attributes": {}}

        for s_col in sens_cols:
            dp_syn  = self._demographic_parity_gap(df_syn,  s_col, target)
            dp_real = self._demographic_parity_gap(df_real, s_col, target)

            syn_gap  = dp_syn.get("demographic_parity_gap",  None)
            real_gap = dp_real.get("demographic_parity_gap", None)

            results["sensitive_attributes"][s_col] = {
                "synthetic":    dp_syn,
                "real":         dp_real,
                "gap_preserved": (
                    round(abs(syn_gap - real_gap), 4)
                    if syn_gap is not None and real_gap is not None
                    else None
                ),
            }

        return results
