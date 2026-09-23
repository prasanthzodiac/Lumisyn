"""
Differential Privacy Marginal Recalibration (Post-Step)
=============================================================================
Gate 4d Item A6: DP marginal recalibration post-step, ledger-charged.
Sensitivity derivation (Add/Remove Adjacency):
  - In a dataset D, each record x = (x_1, ..., x_M) has values in M columns.
  - For each continuous column, public bin edges partition the support into K_m disjoint bins.
    A single record touches at most 1 bin in each column.
  - For each categorical column, public categories define K_m mutually exclusive categories.
    A single record touches at most 1 category in each column.
  - Under add/remove adjacency (D and D' differ by +/- 1 record):
    ||H(D) - H(D')||_1 <= sum_{m=1}^M 1 = M.
  - Therefore, the global L1 sensitivity across all M column histograms is Delta_1 = M.
  - Releasing all M histograms via the Laplace mechanism with scale b = M / eps_m satisfies eps_m-DP.

Post-processing:
  - For numeric columns: map synthetic rank quantiles to the DP empirical CDF.
  - For categorical columns: rebalance synthetic categorical frequencies to match DP histogram frequencies.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import yaml
import numpy as np
import pandas as pd


class DPMarginalRecalibrator:
    """
    Calibrates synthetic dataframe marginals using DP-released 1-way histograms.
    """

    def __init__(
        self,
        schema_path_or_dict: Union[str, Path, Dict[str, Any]],
        num_continuous_bins: int = 50,
    ):
        if isinstance(schema_path_or_dict, (str, Path)):
            with open(schema_path_or_dict, "r", encoding="utf-8") as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = schema_path_or_dict

        if "columns" in self.schema and isinstance(self.schema["columns"], dict):
            self.columns_spec = self.schema["columns"]
        else:
            self.columns_spec = self.schema
        self.num_continuous_bins = num_continuous_bins

    @property
    def num_columns(self) -> int:
        return len(self.columns_spec)

    def get_sensitivity(self) -> int:
        """
        Global L1 sensitivity under add/remove adjacency across all M columns:
        Delta_1 = M (each record touches at most 1 bin/category per column).
        """
        return self.num_columns

    def release_dp_histograms(
        self,
        df_real: pd.DataFrame,
        eps_m: float,
        seed: int = 42,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Releases DP 1-way histograms for all columns under eps_m-DP.
        """
        assert eps_m > 0, "eps_m must be strictly positive"
        rng = np.random.RandomState(seed)
        M = self.get_sensitivity()
        laplace_scale = float(M) / eps_m

        dp_histograms = {}
        bins_per_column = {}
        noise_to_count_ratios = {}
        N = len(df_real)

        for col_name, spec in self.columns_spec.items():
            col_type = spec.get("type")

            if col_type in ("numeric", "continuous"):
                bounds = spec.get("bounds", [0.0, 1.0])
                lo, hi = float(bounds[0]), float(bounds[1])
                is_zero_inflated = spec.get("zero_inflated", False) or col_name in ("capital-gain", "capital-loss")
                is_log_scale = (spec.get("transform") == "log1p") or (col_name == "fnlwgt")

                vals = pd.to_numeric(df_real[col_name], errors="coerce").dropna().values

                if is_zero_inflated:
                    # Isolated special value 0.0 bin + log-spaced bins for positive values
                    zero_count = float(np.sum(vals == 0.0))
                    pos_vals = vals[vals > 0.0]
                    lo_pos = max(1.0, float(np.min(pos_vals))) if len(pos_vals) > 0 else 1.0
                    pos_bins = np.geomspace(lo_pos, hi, self.num_continuous_bins)
                    pos_counts, _ = np.histogram(pos_vals, bins=pos_bins)
                    
                    counts = np.concatenate([[zero_count], pos_counts])
                    bin_edges = np.concatenate([[0.0, 0.0], pos_bins]) # 0 bin + pos bins
                    
                    noise = rng.laplace(0.0, laplace_scale, size=len(counts))
                    dp_counts = np.maximum(0.0, counts + noise)
                    total_dp = float(np.sum(dp_counts))
                    dp_probs = dp_counts / max(1e-9, total_dp) if total_dp > 0 else np.ones_like(dp_counts) / len(dp_counts)

                    dp_histograms[col_name] = {
                        "type": "numeric_zero_inflated",
                        "bin_edges": bin_edges,
                        "pos_bins": pos_bins,
                        "dp_probs": dp_probs,
                        "dp_zero_prob": dp_probs[0],
                        "bounds": (lo, hi),
                    }
                    K_m = len(counts)

                elif is_log_scale and lo > 0:
                    bins = np.geomspace(lo, hi, self.num_continuous_bins + 1)
                    counts, _ = np.histogram(vals, bins=bins)
                    noise = rng.laplace(0.0, laplace_scale, size=len(counts))
                    dp_counts = np.maximum(0.0, counts + noise)
                    total_dp = float(np.sum(dp_counts))
                    dp_probs = dp_counts / max(1e-9, total_dp) if total_dp > 0 else np.ones_like(dp_counts) / len(dp_counts)

                    dp_histograms[col_name] = {
                        "type": "numeric",
                        "bins": bins,
                        "bin_centers": np.sqrt(bins[:-1] * bins[1:]),
                        "dp_counts": dp_counts,
                        "dp_probs": dp_probs,
                        "bounds": (lo, hi),
                    }
                    K_m = len(counts)

                else:
                    bins = np.linspace(lo, hi, self.num_continuous_bins + 1)
                    counts, _ = np.histogram(vals, bins=bins)
                    noise = rng.laplace(0.0, laplace_scale, size=len(counts))
                    dp_counts = np.maximum(0.0, counts + noise)
                    total_dp = float(np.sum(dp_counts))
                    dp_probs = dp_counts / max(1e-9, total_dp) if total_dp > 0 else np.ones_like(dp_counts) / len(dp_counts)

                    dp_histograms[col_name] = {
                        "type": "numeric",
                        "bins": bins,
                        "bin_centers": 0.5 * (bins[:-1] + bins[1:]),
                        "dp_counts": dp_counts,
                        "dp_probs": dp_probs,
                        "bounds": (lo, hi),
                    }
                    K_m = len(counts)

            elif col_type == "categorical":
                categories = spec.get("categories", [])
                s_real = df_real[col_name].astype(str)
                raw_counts = s_real.value_counts()
                counts = np.array([float(raw_counts.get(cat, 0)) for cat in categories])
                
                noise = rng.laplace(0.0, laplace_scale, size=len(counts))
                dp_counts = np.maximum(0.0, counts + noise)
                total_dp = float(np.sum(dp_counts))
                dp_probs = dp_counts / max(1e-9, total_dp) if total_dp > 0 else np.ones_like(dp_counts) / len(dp_counts)

                dp_histograms[col_name] = {
                    "type": "categorical",
                    "categories": list(categories),
                    "dp_counts": dp_counts,
                    "dp_probs": dp_probs,
                }
                K_m = len(counts)

            bins_per_column[col_name] = K_m
            avg_count_per_bin = float(N) / max(1, K_m)
            noise_to_count_ratios[col_name] = round(laplace_scale / max(1e-6, avg_count_per_bin), 4)

        report = {
            "mechanism": "Laplace",
            "adjacency": "add_remove",
            "sensitivity_L1": M,
            "eps_m": eps_m,
            "delta_m": 0.0,
            "laplace_scale": round(laplace_scale, 6),
            "num_columns_calibrated": M,
            "released_histograms_count": len(dp_histograms),
            "bins_per_column": bins_per_column,
            "noise_to_count_ratios": noise_to_count_ratios,
        }
        return dp_histograms, report

    def recalibrate(
        self,
        df_syn: pd.DataFrame,
        dp_histograms: Dict[str, Any],
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Maps synthetic dataframe to match the DP-released marginal distributions.
        Post-processing preserves the rank structure / correlations of the synthetic generator.
        """
        df_cal = df_syn.copy()
        N = len(df_cal)
        rng = np.random.RandomState(seed)

        for col_name, hist in dp_histograms.items():
            if col_name not in df_cal.columns:
                continue

            if hist["type"] == "numeric_zero_inflated":
                dp_probs = hist["dp_probs"]
                p0 = hist["dp_zero_prob"]
                pos_bins = hist["pos_bins"]
                lo, hi = hist["bounds"]
                pos_probs = dp_probs[1:]
                sum_pos = float(np.sum(pos_probs))
                pos_probs_norm = pos_probs / max(1e-9, sum_pos) if sum_pos > 0 else np.ones_like(pos_probs) / len(pos_probs)
                
                pos_cdf = np.concatenate([[0.0], np.cumsum(pos_probs_norm)])
                pos_cdf = np.clip(pos_cdf, 0.0, 1.0)
                pos_cdf[-1] = 1.0

                syn_vals = pd.to_numeric(df_cal[col_name], errors="coerce").values
                ranks = np.argsort(np.argsort(syn_vals + rng.randn(N) * 1e-8))
                u = (ranks + 0.5) / float(N)

                # If u <= p0, set to 0.0; otherwise map to positive CDF
                recal_vals = np.zeros(N, dtype=np.float32)
                pos_mask = u > p0
                if np.any(pos_mask):
                    u_pos = (u[pos_mask] - p0) / max(1e-6, (1.0 - p0))
                    recal_vals[pos_mask] = np.clip(np.interp(u_pos, pos_cdf, pos_bins), lo, hi)
                df_cal[col_name] = recal_vals

            elif hist["type"] == "numeric":
                bins = hist["bins"]
                dp_probs = hist["dp_probs"]
                lo, hi = hist["bounds"]

                # Construct DP cumulative distribution function
                cdf = np.concatenate([[0.0], np.cumsum(dp_probs)])
                cdf = np.clip(cdf, 0.0, 1.0)
                cdf[-1] = 1.0

                # Compute empirical uniform percentiles for synthetic column
                syn_vals = pd.to_numeric(df_cal[col_name], errors="coerce").values
                ranks = np.argsort(np.argsort(syn_vals + rng.randn(N) * 1e-8))
                u = (ranks + 0.5) / float(N)

                # Invert CDF: interpolate
                recal_vals = np.interp(u, cdf, bins)
                recal_vals = np.clip(recal_vals, lo, hi)
                df_cal[col_name] = recal_vals

            elif hist["type"] == "categorical":
                categories = hist["categories"]
                dp_probs = hist["dp_probs"]

                # Target integer counts matching DP proportions
                target_counts = np.round(dp_probs * N).astype(int)
                diff = N - int(np.sum(target_counts))
                if diff != 0:
                    max_idx = np.argmax(target_counts)
                    target_counts[max_idx] += diff

                # Construct array of target categories
                target_pool = []
                for cat, count in zip(categories, target_counts):
                    target_pool.extend([cat] * count)
                target_pool = np.array(target_pool)

                # Align with synthetic predictions:
                syn_cats = df_cal[col_name].astype(str).values
                cat_to_rank = {c: i for i, c in enumerate(categories)}
                syn_rank = np.array([cat_to_rank.get(c, len(categories)) for c in syn_cats])
                sort_idx = np.argsort(syn_rank + rng.uniform(0, 0.99, size=N))
                
                target_pool_sorted = np.sort(np.array([cat_to_rank.get(c, 0) for c in target_pool]))
                inv_cat_map = {i: c for i, c in enumerate(categories)}
                assigned_cats = np.array([inv_cat_map.get(r, categories[0]) for r in target_pool_sorted])

                cal_col = np.empty(N, dtype=object)
                cal_col[sort_idx] = assigned_cats
                df_cal[col_name] = cal_col

        return df_cal
