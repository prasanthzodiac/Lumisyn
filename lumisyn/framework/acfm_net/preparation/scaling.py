from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer

class AdaptiveFeatureScaler:
    """
    2.3 Adaptive Feature Scaling (ADPE Stage 2)
    Auto-selects optimal scaler per numeric feature based on skewness & kurtosis:
    • Skewed (|skew| > 1.0): RobustScaler
    • Heavy-tailed (kurtosis > 3.0): QuantileTransformer (Gaussian output)
    • Bounded (e.g. [0, 1]): MinMaxScaler
    • Standard (normal-like): StandardScaler
    Provides invertible scale() and inverse_scale().
    """

    def __init__(self):
        self.scalers = {}
        self.feature_scaler_types = {}

    def fit_transform(
        self,
        X: np.ndarray,
        feature_specs: Dict[str, Any] = None,
        privacy_mode: bool = False,
        encoded_column_names: list[str] = None,
    ) -> np.ndarray:
        X_scaled = X.copy()
        num_cols = X.shape[1]

        if privacy_mode:
            # Public-schema affine scaling. No means, variances, quantiles, or
            # extrema are learned from the cohort.
            names = encoded_column_names or list((feature_specs or {}).keys())
            for numeric_idx, encoded_name in enumerate(names[:num_cols]):
                spec = (feature_specs or {}).get(encoded_name)
                if not spec:
                    self.scalers[numeric_idx] = (0.0, 1.0)
                    self.feature_scaler_types[numeric_idx] = "public_identity"
                    continue
                if spec.get("type") == "numeric":
                    lo = float(spec.get("orig_min", -1.0))
                    hi = float(spec.get("orig_max", 1.0))
                    if hi <= lo:
                        hi = lo + 1.0
                    X_scaled[:, numeric_idx] = 2.0 * (np.clip(X[:, numeric_idx], lo, hi) - lo) / (hi - lo) - 1.0
                    self.scalers[numeric_idx] = (lo, hi)
                    self.feature_scaler_types[numeric_idx] = "public_affine"
                elif spec.get("type") == "categorical_ord":
                    num_cats = len(spec.get("categories", []))
                    lo = 0.0
                    hi = float(num_cats - 1) if num_cats > 1 else 1.0
                    X_scaled[:, numeric_idx] = 2.0 * (np.clip(X[:, numeric_idx], lo, hi) - lo) / (hi - lo) - 1.0
                    self.scalers[numeric_idx] = (lo, hi)
                    self.feature_scaler_types[numeric_idx] = "public_affine"
                else:
                    self.scalers[numeric_idx] = (0.0, 1.0)
                    self.feature_scaler_types[numeric_idx] = "public_identity"
            for idx in range(len(names), num_cols):
                self.scalers[idx] = (0.0, 1.0)
                self.feature_scaler_types[idx] = "public_identity"
            self.privacy_mode = True
            return X_scaled

        for j in range(num_cols):
            vals_1d = X[:, j].ravel()
            skew_v = float(stats.skew(vals_1d)) if len(vals_1d) > 0 else 0.0
            kurt_v = float(stats.kurtosis(vals_1d)) if len(vals_1d) > 0 else 0.0

            if abs(skew_v) > 1.5:
                scaler = RobustScaler()
                s_type = "robust"
            elif kurt_v > 4.0:
                scaler = QuantileTransformer(output_distribution="normal", random_state=42)
                s_type = "quantile"
            else:
                scaler = StandardScaler()
                s_type = "standard"

            col_vals = vals_1d.reshape(-1, 1)
            X_scaled[:, j] = scaler.fit_transform(col_vals).ravel()
            self.scalers[j] = scaler
            self.feature_scaler_types[j] = s_type

        return X_scaled

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        X_orig = X_scaled.copy()
        num_cols = X_scaled.shape[1]

        for j in range(num_cols):
            if j in self.scalers:
                scaler = self.scalers[j]
                col_vals = X_scaled[:, j].reshape(-1, 1)
                if isinstance(scaler, tuple):
                    lo, hi = scaler
                    X_orig[:, j] = (col_vals.ravel() + 1.0) * 0.5 * (hi - lo) + lo
                else:
                    X_orig[:, j] = scaler.inverse_transform(col_vals).ravel()

        return X_orig
