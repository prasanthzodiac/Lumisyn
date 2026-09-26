from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


class KSTestEvaluator:
    """Kolmogorov-Smirnov test per numeric feature."""

    def evaluate(self, df_real: pd.DataFrame, df_syn: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        num_cols = [c for c in df_real.columns
                    if pd.api.types.is_numeric_dtype(df_real[c]) and c in df_syn.columns]
        ks_stats, p_values = [], []
        for col in num_cols:
            r = df_real[col].dropna().values.astype(float)
            s = df_syn[col].dropna().values.astype(float)
            if len(r) < 2 or len(s) < 2:
                continue
            stat, p = stats.ks_2samp(r, s)
            ks_stats.append(stat)
            p_values.append(p)
            results[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p), 4),
                            "passed": p > 0.05}
        results["_summary"] = {
            "mean_ks_stat": round(float(np.mean(ks_stats)), 4) if ks_stats else None,
            "columns_tested": len(ks_stats),
            "columns_passed_005": sum(1 for p in p_values if p > 0.05),
        }
        return results


class WassersteinEvaluator:
    """1-D Wasserstein (Earth Mover) distance per numeric feature."""

    def evaluate(self, df_real: pd.DataFrame, df_syn: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        num_cols = [c for c in df_real.columns
                    if pd.api.types.is_numeric_dtype(df_real[c]) and c in df_syn.columns]
        distances = []
        for col in num_cols:
            r = df_real[col].dropna().values.astype(float)
            s = df_syn[col].dropna().values.astype(float)
            if len(r) < 2 or len(s) < 2:
                continue
            std_r = float(np.std(r)) or 1.0
            wd = float(stats.wasserstein_distance(r / std_r, s / std_r))
            distances.append(wd)
            results[col] = {"wasserstein": round(wd, 4)}
        results["_summary"] = {
            "mean_wasserstein": round(float(np.mean(distances)), 4) if distances else None,
            "columns_tested": len(distances),
        }
        return results


class VarianceCovarianceEvaluator:
    """
    Evaluates variance preservation, covariance matrix Frobenius difference,
    and tail extreme value coverage (Issues 1, 2, 7, 8).
    """

    def evaluate(self, df_real: pd.DataFrame, df_syn: pd.DataFrame) -> Dict[str, Any]:
        num_cols = [c for c in df_real.columns
                    if pd.api.types.is_numeric_dtype(df_real[c]) and c in df_syn.columns]
        if not num_cols:
            return {"error": "No numeric columns for variance evaluation"}

        var_results = {}
        tail_results = {}

        for col in num_cols:
            r = df_real[col].dropna().values.astype(float)
            s = df_syn[col].dropna().values.astype(float)
            var_r = float(np.var(r)) or 1e-6
            var_s = float(np.var(s))
            var_ratio = var_s / var_r

            var_results[col] = {
                "real_var": round(var_r, 4),
                "syn_var": round(var_s, 4),
                "variance_ratio": round(var_ratio, 4),
                "variance_preserved": 0.8 <= var_ratio <= 1.25,
            }

            # Tail / Extreme value coverage
            r_min, r_max = float(r.min()), float(r.max())
            s_min, s_max = float(s.min()), float(s.max())
            r_q01, r_q99 = float(np.quantile(r, 0.01)), float(np.quantile(r, 0.99))
            s_q01, s_q99 = float(np.quantile(s, 0.01)), float(np.quantile(s, 0.99))

            tail_results[col] = {
                "real_range": [r_min, r_max],
                "syn_range": [s_min, s_max],
                "real_q01_q99": [round(r_q01, 2), round(r_q99, 2)],
                "syn_q01_q99": [round(s_q01, 2), round(s_q99, 2)],
                "tail_coverage_ratio": round(
                    min(1.0, (s_max - s_min) / max(1e-5, (r_max - r_min))), 4
                ),
            }

        # Covariance Matrix comparison
        cov_r = df_real[num_cols].cov().fillna(0).values
        cov_s = df_syn[num_cols].cov().fillna(0).values
        cov_frob_diff = float(np.linalg.norm(cov_r - cov_s, "fro"))

        return {
            "feature_variance": var_results,
            "tail_extreme_coverage": tail_results,
            "covariance_frobenius_diff": round(cov_frob_diff, 4),
            "_summary": {
                "mean_variance_ratio": round(
                    float(np.mean([v["variance_ratio"] for v in var_results.values()])), 4
                ),
                "cov_diff_norm": round(cov_frob_diff, 4),
            }
        }


class CorrelationSimilarityEvaluator:
    """
    Frobenius norm of correlation matrix diff and full 2D correlation matrices
    for heatmap rendering (Issues 4, 8, 9).
    """

    def evaluate(self, df_real: pd.DataFrame, df_syn: pd.DataFrame) -> Dict[str, Any]:
        num_cols = [c for c in df_real.columns
                    if pd.api.types.is_numeric_dtype(df_real[c]) and c in df_syn.columns]
        if len(num_cols) < 2:
            return {"error": "Not enough numeric columns for correlation comparison"}

        corr_r_df = df_real[num_cols].corr().fillna(0).round(4)
        corr_s_df = df_syn[num_cols].corr().fillna(0).round(4)

        C_r = corr_r_df.values
        C_s = corr_s_df.values
        diff_matrix = (corr_r_df - corr_s_df).round(4)

        frob = float(np.linalg.norm(C_r - C_s, "fro"))
        max_frob = float(np.sqrt(len(num_cols) ** 2))

        per_feature = {}
        for col in num_cols:
            mae = float((corr_r_df[col] - corr_s_df[col]).abs().mean())
            per_feature[col] = {"corr_mae": round(mae, 4)}

        return {
            "frobenius_norm": round(frob, 4),
            "frobenius_similarity": round(max(0.0, 1.0 - frob / max_frob), 4),
            "real_correlation_matrix": corr_r_df.to_dict(),
            "syn_correlation_matrix": corr_s_df.to_dict(),
            "diff_correlation_matrix": diff_matrix.to_dict(),
            "per_feature": per_feature,
            "_summary": {
                "columns": num_cols,
                "num_features": len(num_cols),
            }
        }


class JSDivergenceEvaluator:
    """Jensen-Shannon Divergence per numeric feature (histogram-based)."""

    @staticmethod
    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        p = p + 1e-10;  q = q + 1e-10
        p /= p.sum();    q /= q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    def evaluate(self, df_real: pd.DataFrame, df_syn: pd.DataFrame,
                 n_bins: int = 20) -> Dict[str, Any]:
        results = {}
        num_cols = [c for c in df_real.columns
                    if pd.api.types.is_numeric_dtype(df_real[c]) and c in df_syn.columns]
        jsds = []
        for col in num_cols:
            r = df_real[col].dropna().values.astype(float)
            s = df_syn[col].dropna().values.astype(float)
            if len(r) < 4 or len(s) < 4:
                continue
            lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
            bins = np.linspace(lo, hi + 1e-8, n_bins + 1)
            p_hist, _ = np.histogram(r, bins=bins)
            q_hist, _ = np.histogram(s, bins=bins)
            jsd = self._jsd(p_hist.astype(float), q_hist.astype(float))
            jsds.append(jsd)
            results[col] = {"jsd": round(jsd, 4)}
        results["_summary"] = {
            "mean_jsd": round(float(np.mean(jsds)), 4) if jsds else None,
            "columns_tested": len(jsds),
        }
        return results
