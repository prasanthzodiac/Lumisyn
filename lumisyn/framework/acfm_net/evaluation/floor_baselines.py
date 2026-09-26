import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class IndependentMarginalsDPGenerator:
    """
    Independent-marginals baseline generator from DP histograms at target epsilon.
    Allocates epsilon uniformly across columns: eps_col = epsilon / d.
    For each categorical column: adds Laplace noise with scale 2.0 / eps_col to counts.
    For continuous columns: uses DP histogram or bounded DP Gaussian distribution.
    Generates records where all features are mutually independent.
    """
    def __init__(self, epsilon: float = 1.0, seed: int = 42):
        self.epsilon = float(epsilon)
        self.seed = seed
        self.column_distributions = {}

    def fit(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None):
        rng = np.random.RandomState(self.seed)
        d = len(df.columns)
        eps_per_col = self.epsilon / max(1, d)
        N = len(df)

        self.column_distributions = {}

        for col in df.columns:
            series = df[col].dropna()
            is_num = pd.api.types.is_numeric_dtype(series)

            if is_num:
                # 1D bounded numeric: DP histogram with 20 bins
                col_min = float(series.min()) if schema is None else float(schema.get(col, {}).get("bounds", [0, 1])[0])
                col_max = float(series.max()) if schema is None else float(schema.get(col, {}).get("bounds", [0, 1])[1])
                if col_max <= col_min:
                    col_max = col_min + 1.0
                bins = np.linspace(col_min, col_max, 21)
                counts, _ = np.histogram(series, bins=bins)
                # Laplace noise with sensitivity 2 (add/remove 1 person changes at most 2 bin counts)
                noise = rng.laplace(0.0, 2.0 / eps_per_col, size=len(counts))
                noisy_counts = np.maximum(0.0, counts + noise)
                if noisy_counts.sum() == 0:
                    noisy_counts = np.ones_like(noisy_counts)
                probs = noisy_counts / noisy_counts.sum()
                bin_midpoints = 0.5 * (bins[:-1] + bins[1:])
                self.column_distributions[col] = {
                    "type": "numeric",
                    "values": bin_midpoints,
                    "probs": probs,
                    "dtype": str(df[col].dtype),
                }
            else:
                # Categorical column: DP value counts
                cats = schema.get(col, {}).get("categories") if schema else None
                if cats is None:
                    cats = sorted(series.astype(str).unique().tolist())
                counts = np.array([np.sum(series.astype(str) == str(c)) for c in cats], dtype=np.float64)
                # Laplace noise with sensitivity 2
                noise = rng.laplace(0.0, 2.0 / eps_per_col, size=len(counts))
                noisy_counts = np.maximum(0.0, counts + noise)
                if noisy_counts.sum() == 0:
                    noisy_counts = np.ones_like(noisy_counts)
                probs = noisy_counts / noisy_counts.sum()
                self.column_distributions[col] = {
                    "type": "categorical",
                    "values": list(cats),
                    "probs": probs,
                }

    def generate(self, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.RandomState(seed or self.seed)
        df_out = {}
        for col, dist in self.column_distributions.items():
            vals = dist["values"]
            probs = dist["probs"]
            sampled = rng.choice(vals, size=n_samples, p=probs)
            if dist["type"] == "numeric" and "int" in dist.get("dtype", ""):
                sampled = np.round(sampled).astype(np.int32)
            df_out[col] = sampled
        return pd.DataFrame(df_out)


class ConstantMeanGenerator:
    """
    Constant Mean Baseline Generator.
    For each numerical column, predicts the constant sample mean.
    For each categorical column, predicts the constant modal category.
    Represents the zero-information marginal baseline.
    """
    def __init__(self):
        self.column_means = {}

    def fit(self, df: pd.DataFrame):
        self.column_means = {}
        for col in df.columns:
            series = df[col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                self.column_means[col] = {
                    "type": "numeric",
                    "val": float(series.mean()),
                    "dtype": str(df[col].dtype),
                }
            else:
                mode_val = series.mode()[0] if len(series) else "MISSING"
                self.column_means[col] = {
                    "type": "categorical",
                    "val": mode_val,
                }

    def generate(self, n_samples: int) -> pd.DataFrame:
        df_out = {}
        for col, spec in self.column_means.items():
            if spec["type"] == "numeric":
                val = spec["val"]
                if "int" in spec.get("dtype", ""):
                    val = int(round(val))
                df_out[col] = np.full(n_samples, val)
            else:
                df_out[col] = [spec["val"]] * n_samples
        return pd.DataFrame(df_out)
