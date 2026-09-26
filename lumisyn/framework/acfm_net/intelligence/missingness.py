from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

class MissingValueAnalyzer:
    """
    1.4 Multi-resolution Missing Value Analyzer (UDIE Stage 1)
    Calculates global, feature, and pair-level missingness,
    executes Little's MCAR test (chi2), and tests MAR/MNAR via logistic regression.
    """

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        num_rows, num_cols = df.shape
        if num_rows == 0 or num_cols == 0:
            return {
                "global_missing_pct": 0.0,
                "feature_missing_pct": {},
                "joint_missing_matrix": np.zeros((0, 0)),
                "mcar_p_value": 1.0,
                "mcar_chi2": 0.0,
                "missing_mechanism": "MCAR"
            }

        # 1. Global missing %
        total_cells = num_rows * num_cols
        missing_mask = df.isna().values
        total_missing = int(np.sum(missing_mask))
        global_missing_pct = float(total_missing / total_cells)

        # 2. Feature level missing %
        feature_missing_pct = {col: float(df[col].isna().mean()) for col in df.columns}

        # 3. Pairwise joint missingness matrix
        joint_missing = np.dot(missing_mask.astype(float).T, missing_mask.astype(float)) / num_rows

        # 4. Little's MCAR test approximation
        mcar_chi2, mcar_p_value = self._littles_mcar_test(df)

        # 5. MAR / MNAR classification via missingness indicator regression
        missing_mechanisms = self._classify_mechanisms(df, mcar_p_value)

        return {
            "global_missing_pct": global_missing_pct,
            "feature_missing_pct": feature_missing_pct,
            "joint_missing_matrix": joint_missing,
            "mcar_p_value": mcar_p_value,
            "mcar_chi2": mcar_chi2,
            "missing_mechanisms": missing_mechanisms
        }

    def _littles_mcar_test(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Approximation of Little's MCAR test using chi-square test of independence
        across missingness patterns.
        """
        missing_mask = df.isna()
        # Find distinct missingness patterns
        patterns = missing_mask.drop_duplicates()
        if len(patterns) <= 1:
            return 0.0, 1.0  # No missingness or uniform missingness

        # Compute chi2 statistic comparing observed pattern counts vs expected
        pattern_counts = missing_mask.groupby(list(df.columns), observed=False).size().values
        chi2_stat, p_val = stats.chisquare(pattern_counts)
        if np.isnan(p_val):
            p_val = 1.0
            chi2_stat = 0.0
        return float(chi2_stat), float(p_val)

    def _classify_mechanisms(self, df: pd.DataFrame, mcar_p_value: float) -> Dict[str, str]:
        mechanisms = {}
        missing_mask = df.isna()

        for col in df.columns:
            m_col = missing_mask[col].values
            if np.sum(m_col) == 0:
                mechanisms[col] = "NONE"
            elif mcar_p_value > 0.05:
                mechanisms[col] = "MCAR"
            else:
                # Test logistic regression of missingness on other features
                other_cols = [c for c in df.columns if c != col]
                numeric_others = df[other_cols].select_dtypes(include=[np.number]).fillna(0)
                if numeric_others.shape[1] > 0 and len(np.unique(m_col)) > 1:
                    try:
                        clf = LogisticRegression(max_iter=100)
                        clf.fit(numeric_others, m_col)
                        score = clf.score(numeric_others, m_col)
                        if score > 0.65:
                            mechanisms[col] = "MAR"
                        else:
                            mechanisms[col] = "MNAR"
                    except Exception:
                        mechanisms[col] = "MAR"
                else:
                    mechanisms[col] = "MAR"

        return mechanisms
