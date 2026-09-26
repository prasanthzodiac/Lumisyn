from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score

class DependencyEstimationEngine:
    """
    1.7 Dependency Estimation Engine (UDIE Stage 1)
    Computes type-appropriate dependency matrix R:
    • Cont-Cont: Pearson + Spearman
    • Cont-Cat: Point-biserial / ANOVA F
    • Cat-Cat: Cramér's V + Mutual Information
    Returns complete d x d dependency matrix R.
    """

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        columns = list(df.columns)
        d = len(columns)
        R = np.eye(d)

        # Downsample for fast dependency matrix estimation
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)

        for i in range(d):
            for j in range(i + 1, d):
                col_i = columns[i]
                col_j = columns[j]
                val_i = df[col_i].dropna()
                val_j = df[col_j].dropna()

                common_idx = df[[col_i, col_j]].dropna().index
                if len(common_idx) < 5:
                    dep = 0.0
                else:
                    s_i = df.loc[common_idx, col_i]
                    s_j = df.loc[common_idx, col_j]

                    is_i_num = pd.api.types.is_numeric_dtype(s_i)
                    is_j_num = pd.api.types.is_numeric_dtype(s_j)

                    if is_i_num and is_j_num:
                        # Pearson & Spearman average
                        p_corr, _ = stats.pearsonr(s_i, s_j)
                        s_corr, _ = stats.spearmanr(s_i, s_j)
                        dep = 0.5 * (abs(p_corr) + abs(s_corr))
                    elif is_i_num != is_j_num:
                        # Point-biserial / ANOVA F norm
                        num_s = s_i if is_i_num else s_j
                        cat_s = s_j if is_i_num else s_i
                        if cat_s.nunique() > 30:
                            dep = 0.0
                        else:
                            groups = [group.values for _, group in num_s.groupby(cat_s, observed=False)]
                            if len(groups) > 1:
                                f_stat, _ = stats.f_oneway(*groups)
                                dep = min(1.0, float(f_stat / (f_stat + len(common_idx))))
                            else:
                                dep = 0.0
                    else:
                        # Categorical - Categorical: Cramér's V
                        if s_i.nunique() > 30 or s_j.nunique() > 30:
                            dep = 0.0
                        else:
                            crosstab = pd.crosstab(s_i, s_j)
                            chi2, _, _, _ = stats.chi2_contingency(crosstab)
                            n = len(common_idx)
                            r, k = crosstab.shape
                            min_dim = min(r - 1, k - 1)
                            dep = np.sqrt(chi2 / (n * max(1, min_dim))) if min_dim > 0 else 0.0

                if np.isnan(dep):
                    dep = 0.0
                R[i, j] = float(dep)
                R[j, i] = float(dep)

        return {
            "dependency_matrix": R,
            "columns": columns
        }
