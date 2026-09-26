from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

class SensitiveAttributeDetector:
    """
    1.6 Sensitive Attribute Detector (UDIE Stage 1)
    Calculates feature sensitivity scores s_i ∈ [0, 1] based on clinical domain rules,
    PHI keywords, quasi-identifier combinations, and k-anonymity estimation.
    """

    def __init__(self):
        self.sensitive_keywords = {
            "high": ["ssn", "social_security", "name", "mrn", "address", "zip", "phone", "email", "ip_address", "biometric"],
            "medium": ["age", "gender", "sex", "race", "ethnicity", "dob", "birth", "income", "marital", "insurance"],
            "clinical": ["diagnosis", "disease", "icd", "cancer", "hiv", "mental_health", "substance", "mortality"]
        }

    def detect(self, df: pd.DataFrame, use_data_dependent_risk: bool = True) -> Dict[str, Any]:
        columns = list(df.columns)
        num_features = len(columns)
        sensitivity_vector = np.zeros(num_features)
        sensitivity_map = {}

        quasi_identifiers = []

        for i, col in enumerate(columns):
            col_lower = str(col).lower()
            score = 0.1  # base sensitivity

            # Rule 1: Direct High-Risk PHI
            if any(kw in col_lower for kw in self.sensitive_keywords["high"]):
                score = 0.95
                quasi_identifiers.append(col)

            # Rule 2: Demographic / Quasi-identifiers
            elif any(kw in col_lower for kw in self.sensitive_keywords["medium"]):
                score = 0.65
                quasi_identifiers.append(col)

            # Rule 3: Clinical Condition / Diagnosis
            elif any(kw in col_lower for kw in self.sensitive_keywords["clinical"]):
                score = 0.50

            # Cardinality/k-anonymity is disabled in strict mode because it is
            # a data-dependent statistic. The public column-name rules remain.
            if use_data_dependent_risk:
                unique_ratio = df[col].nunique() / max(len(df), 1)
                if unique_ratio > 0.8 and df[col].dtype == "object":
                    score = max(score, 0.85)

            sensitivity_vector[i] = score
            sensitivity_map[col] = float(score)

        # Estimate k-anonymity score
        k_anonymity = self._estimate_k_anonymity(df, quasi_identifiers) if use_data_dependent_risk else None

        return {
            "sensitivity_vector": sensitivity_vector,
            "sensitivity_map": sensitivity_map,
            "quasi_identifiers": quasi_identifiers,
            "k_anonymity": k_anonymity
        }

    def _estimate_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> int:
        if not quasi_identifiers or len(df) == 0:
            return len(df)
        selected_qi = quasi_identifiers[:4]  # Top quasi-identifiers
        counts = df.groupby(selected_qi, observed=False).size()
        if counts.empty:
            return 1
        return int(counts.min())
