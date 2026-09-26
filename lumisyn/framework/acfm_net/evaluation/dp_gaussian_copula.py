"""
Differentially Private Gaussian Copula Baseline Generator (DPCopula)
=============================================================================
Gate 4d Item 2: Replaces arbitrary rank-normal Gaussian mechanism with
the formal DPCopula Kendall's-Tau Estimator.

Sensitivity Derivation (Add/Remove Adjacency):
  - Let dataset D have N records and M columns.
  - Number of column pairs: P = M * (M - 1) / 2.
  - Kendall's tau between columns p and q is:
      tau_{pq} = (C - D) / binom(N, 2)
    where C and D are concordant and discordant pairs.
  - Under add/remove adjacency, adding or removing 1 record changes at most N pairs.
    Each pair contributes at most 1 to (C - D).
    Thus, for a single pair (p, q), Delta_1(tau_{pq}) = N / binom(N, 2) = 2 / (N - 1).
  - Across all P = M*(M-1)/2 pairs, the global L1 sensitivity of the vector tau is:
      Delta_1(tau) = P * (2 / (N - 1)) = M * (M - 1) / (N - 1).
  - Noise: To release all pairwise tau values under eps_corr-DP, add Laplace noise:
      b_tau = Delta_1(tau) / eps_corr = (M * (M - 1)) / ((N - 1) * eps_corr).
  - Greiner's Relation: For bivariate Gaussian copulas, Pearson correlation R_{pq}
    is mapped from Kendall's tau via:
      R_{pq} = sin( (pi / 2) * tau_{pq}^{DP} ).
  - Projection: R is projected to the nearest Positive Semi-Definite (PSD) correlation matrix.
  - Marginals: Released via Laplace mechanism with scale b_m = M / eps_m (sensitivity M).
  - Total Privacy: eps_total = eps_m + eps_corr.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau


class DPGaussianCopulaGenerator:
    """
    Differentially Private Gaussian Copula Generator based on Kendall's Tau.
    """

    def __init__(
        self,
        schema_path_or_dict: Union[str, Path, Dict[str, Any]],
        epsilon: float = 1.0,
        eps_marginal_ratio: float = 0.5,
        num_continuous_bins: int = 50,
        seed: int = 42,
    ):
        if isinstance(schema_path_or_dict, (str, Path)):
            with open(schema_path_or_dict, "r", encoding="utf-8") as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = schema_path_or_dict

        self.columns_spec = self.schema.get("columns", {})
        self.epsilon = float(epsilon)
        self.eps_m = self.epsilon * eps_marginal_ratio
        self.eps_corr = self.epsilon * (1.0 - eps_marginal_ratio)
        self.num_continuous_bins = num_continuous_bins
        self.seed = seed
        self.dp_histograms = {}
        self.corr_matrix_dp = None
        self.column_order = list(self.columns_spec.keys())

    def get_sensitivity(self, N: int) -> Tuple[int, float]:
        """
        Returns (L1 sensitivity of marginals, L1 sensitivity of Kendall's tau vector).
        For Kendall's tau, delta_1(tau_{pq}) <= 4 / N under both add/remove and replacement adjacency.
        Across P = M*(M-1)/2 pairs, Delta_1(tau) <= 2 * M * (M - 1) / N.
        """
        M = len(self.column_order)
        delta1_m = M
        delta1_tau = float(2 * M * (M - 1)) / max(1, N)
        return delta1_m, delta1_tau

    def fit(self, df_real: pd.DataFrame):
        rng = np.random.RandomState(self.seed)
        N = len(df_real)
        M = len(self.column_order)
        assert N > 2, "Need at least 3 records for Kendall's tau"

        delta1_m, delta1_tau = self.get_sensitivity(N)

        # ---------------------------------------------------------------------
        # 1. DP Marginals (eps_m)
        # ---------------------------------------------------------------------
        lap_scale_m = float(delta1_m) / max(1e-6, self.eps_m)
        for col_name in self.column_order:
            spec = self.columns_spec.get(col_name, {})
            col_type = spec.get("type")

            if col_type in ("numeric", "continuous"):
                bounds = spec.get("bounds", [0.0, 1.0])
                lo, hi = float(bounds[0]), float(bounds[1])
                bins = np.linspace(lo, hi, self.num_continuous_bins + 1)
                vals = pd.to_numeric(df_real[col_name], errors="coerce").dropna().values
                counts, _ = np.histogram(vals, bins=bins)
                noise = rng.laplace(0.0, lap_scale_m, size=len(counts))
                dp_counts = np.maximum(0.0, counts + noise)
                total = np.sum(dp_counts)
                dp_probs = dp_counts / max(1e-9, total) if total > 0 else np.ones_like(dp_counts) / len(dp_counts)

                self.dp_histograms[col_name] = {
                    "type": "numeric",
                    "bins": bins,
                    "dp_probs": dp_probs,
                    "bounds": (lo, hi),
                }

            elif col_type == "categorical":
                categories = spec.get("categories", [])
                s_real = df_real[col_name].astype(str)
                raw_counts = s_real.value_counts()
                counts = np.array([float(raw_counts.get(cat, 0)) for cat in categories])
                noise = rng.laplace(0.0, lap_scale_m, size=len(counts))
                dp_counts = np.maximum(0.0, counts + noise)
                total = np.sum(dp_counts)
                dp_probs = dp_counts / max(1e-9, total) if total > 0 else np.ones_like(dp_counts) / len(dp_counts)

                self.dp_histograms[col_name] = {
                    "type": "categorical",
                    "categories": list(categories),
                    "dp_probs": dp_probs,
                }

        # ---------------------------------------------------------------------
        # 2. DP Kendall's Tau & Greiner Copula Correlation (eps_corr)
        # ---------------------------------------------------------------------
        # Convert columns to numerical ranks for Kendall tau computation
        num_ranks = np.zeros((N, M), dtype=np.float32)
        for j, col_name in enumerate(self.column_order):
            spec = self.columns_spec.get(col_name, {})
            if spec.get("type") in ("numeric", "continuous"):
                vals = pd.to_numeric(df_real[col_name], errors="coerce").values
            else:
                cats = list(spec.get("categories", []))
                cat_to_rank = {c: i for i, c in enumerate(cats)}
                vals = np.array([cat_to_rank.get(str(x), 0) for x in df_real[col_name]])
            # Rank transform
            num_ranks[:, j] = np.argsort(np.argsort(vals + rng.randn(N) * 1e-8)).astype(np.float32)

        lap_scale_tau = float(delta1_tau) / max(1e-6, self.eps_corr)

        # Pairwise Kendall's tau matrix
        tau_matrix_dp = np.eye(M, dtype=np.float32)
        for i in range(M):
            for j in range(i + 1, M):
                # Fast sample Kendall tau or subsampled if N is very large
                # For N=30000, scipy.stats.kendalltau takes O(N log N)
                tau_val, _ = kendalltau(num_ranks[:, i], num_ranks[:, j])
                if np.isnan(tau_val):
                    tau_val = 0.0
                noise = rng.laplace(0.0, lap_scale_tau)
                tau_noisy = np.clip(tau_val + noise, -1.0, 1.0)
                tau_matrix_dp[i, j] = tau_noisy
                tau_matrix_dp[j, i] = tau_noisy

        # Greiner's relation: R_ij = sin( (pi/2) * tau_ij )
        R_matrix = np.sin(0.5 * np.pi * tau_matrix_dp)
        np.fill_diagonal(R_matrix, 1.0)

        # Project to nearest PSD correlation matrix
        eigvals, eigvecs = np.linalg.eigh(R_matrix)
        eigvals = np.maximum(eigvals, 1e-4)
        psd_R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        inv_std = 1.0 / np.sqrt(np.maximum(1e-8, np.diag(psd_R)))
        self.corr_matrix_dp = inv_std[:, None] * psd_R * inv_std[None, :]
        np.fill_diagonal(self.corr_matrix_dp, 1.0)

    def generate(self, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        M = len(self.column_order)

        # Draw from Gaussian copula
        L = np.linalg.cholesky(self.corr_matrix_dp + 1e-5 * np.eye(M))
        Z_samples = rng.randn(n_samples, M) @ L.T
        U_samples = norm.cdf(Z_samples)

        data = {}
        for j, col_name in enumerate(self.column_order):
            hist = self.dp_histograms[col_name]
            u_j = U_samples[:, j]

            if hist["type"] == "numeric":
                bins = hist["bins"]
                dp_probs = hist["dp_probs"]
                lo, hi = hist["bounds"]
                cdf = np.concatenate([[0.0], np.cumsum(dp_probs)])
                cdf = np.clip(cdf, 0.0, 1.0)
                cdf[-1] = 1.0
                gen_vals = np.interp(u_j, cdf, bins)
                data[col_name] = np.clip(gen_vals, lo, hi)

            elif hist["type"] == "categorical":
                categories = hist["categories"]
                dp_probs = hist["dp_probs"]
                cdf = np.cumsum(dp_probs)
                cdf[-1] = 1.0
                cat_indices = np.searchsorted(cdf, u_j, side="right")
                cat_indices = np.clip(cat_indices, 0, len(categories) - 1)
                data[col_name] = [categories[idx] for idx in cat_indices]

        return pd.DataFrame(data)
