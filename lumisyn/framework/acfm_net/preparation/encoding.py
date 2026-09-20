from typing import Dict, Any, List, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def _detect_column_kind(series: pd.Series) -> str:
    """
    Detect the semantic kind of a numeric column:
    - 'binary'   : exactly 2 unique values (0/1 or True/False or 0&1 in int)
    - 'integer'  : integer-valued (no decimals) with small cardinality (< 150 unique)
    - 'count'    : non-negative integer (counts: Pregnancies, etc.)
    - 'continuous': everything else
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical"

    notna = series.dropna()
    if len(notna) == 0:
        return "continuous"

    unique_vals = notna.unique()
    n_unique = len(unique_vals)

    # Binary check: only 2 distinct values and both are 0 or 1 (or boolean)
    if n_unique <= 2:
        vals_set = set(float(v) for v in unique_vals)
        if vals_set.issubset({0.0, 1.0}):
            return "binary"
        # Could still be binary-like int
        if n_unique == 2:
            return "binary_int"

    # Integer check: all values are whole numbers
    is_int_valued = np.all(notna.values == np.floor(notna.values))
    if is_int_valued:
        min_v = float(notna.min())
        if min_v >= 0 and n_unique <= 150:
            return "count"          # non-negative integer (Pregnancies, Age, etc.)
        elif n_unique <= 150:
            return "integer"        # signed integer

    return "continuous"


class DistributionPreservingEncoder:
    """
    2.2 Distribution-preserving Feature Encoding (ADPE Stage 2) — v5.1 FIXED
    Tracks semantic column kind (binary / count-integer / continuous / categorical).
    Enforces proper dtypes and constraints during decode():
      • binary columns   → rounded to {0, 1}
      • count columns    → rounded to non-negative integers
      • integer columns  → rounded to integers
      • continuous cols  → clipped to original [min, max]
      • categorical cols → argmax reconstruction from OHE
    """

    def __init__(self, enable_schema_log1p: bool = False):
        self.enable_schema_log1p = enable_schema_log1p
        self.feature_specs: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.column_order: List[str] = []
        self.encoded_column_names: List[str] = []

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        df: pd.DataFrame,
        privacy_schema: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.column_order = list(df.columns)
        encoded_blocks = []
        self.encoded_column_names = []

        for col in self.column_order:
            series = df[col]
            is_num = pd.api.types.is_numeric_dtype(series)
            schema = (privacy_schema or {}).get(col, {})

            if is_num:
                kind = schema.get("kind", _detect_column_kind(series))
                transform = schema.get("transform") if self.enable_schema_log1p else None
                vals_raw = series.values.astype(np.float32)
                if transform == "log1p":
                    vals = np.log1p(np.maximum(0.0, vals_raw)).reshape(-1, 1)
                else:
                    vals = vals_raw.reshape(-1, 1)
                encoded_blocks.append(vals)
                self.encoded_column_names.append(col)

                public_bounds = schema.get("bounds")
                notna = series.dropna()
                raw_lo = float(public_bounds[0]) if public_bounds else (float(notna.min()) if len(notna) else 0.0)
                raw_hi = float(public_bounds[1]) if public_bounds else (float(notna.max()) if len(notna) else 1.0)
                lo = float(np.log1p(max(0.0, raw_lo))) if transform == "log1p" else raw_lo
                hi = float(np.log1p(max(0.0, raw_hi))) if transform == "log1p" else raw_hi

                self.feature_specs[col] = {
                    "type": "numeric",
                    "kind": kind,
                    "transform": transform,
                    "dim": 1,
                    "orig_min": lo,
                    "orig_max": hi,
                    "raw_min": raw_lo,
                    "raw_max": raw_hi,
                    "orig_dtype": str(series.dtype),
                }
            else:
                # Categorical encoding
                public_categories = schema.get("categories")
                if public_categories:
                    unknown_level = "MISSING" if "MISSING" in public_categories else str(public_categories[-1])
                    series_sanitized = series.astype(object).where(series.isin(public_categories), unknown_level)
                else:
                    series_sanitized = series

                num_unique = len(public_categories) if public_categories else series_sanitized.nunique()
                if num_unique <= 20:
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    if public_categories:
                        ohe.fit(np.asarray(public_categories, dtype=object).reshape(-1, 1))
                    else:
                        ohe.fit(series_sanitized.values.reshape(-1, 1))
                    vals = ohe.transform(series_sanitized.values.reshape(-1, 1)).astype(np.float32)
                    self.encoders[col] = ohe
                    dim = vals.shape[1]
                    self.feature_specs[col] = {
                        "type": "categorical_ohe",
                        "kind": "categorical",
                        "dim": dim,
                        "categories": list(ohe.categories_[0]),
                    }
                    encoded_blocks.append(vals)
                    for cat in ohe.categories_[0]:
                        self.encoded_column_names.append(f"{col}_{cat}")
                else:
                    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    if public_categories:
                        ord_enc.fit(np.asarray(public_categories, dtype=object).reshape(-1, 1))
                    else:
                        ord_enc.fit(series_sanitized.values.reshape(-1, 1))
                    vals = ord_enc.transform(series_sanitized.values.reshape(-1, 1)).astype(np.float32)
                    self.encoders[col] = ord_enc
                    self.feature_specs[col] = {
                        "type": "categorical_ord",
                        "kind": "categorical",
                        "dim": 1,
                        "categories": list(ord_enc.categories_[0]),
                    }
                    encoded_blocks.append(vals)
                    self.encoded_column_names.append(col)

        encoded_matrix = np.hstack(encoded_blocks)

        info = {
            "feature_specs": self.feature_specs,
            "encoded_column_names": self.encoded_column_names,
            "total_dim": encoded_matrix.shape[1],
        }
        return encoded_matrix, info

    # ------------------------------------------------------------------
    def decode(self, encoded_matrix: np.ndarray) -> pd.DataFrame:
        """
        Inverse transformation with strict dtype & constraint enforcement:
        - binary     → clamp to [0,1] then round to int (0 or 1)
        - count      → clamp to [0, orig_max], round to int
        - integer    → clamp to [orig_min, orig_max], round to int
        - continuous → clamp to [orig_min, orig_max]
        - categorical→ argmax / OHE inverse
        """
        df_dict: Dict[str, np.ndarray] = {}
        curr_idx = 0

        for col in self.column_order:
            spec = self.feature_specs[col]
            dim  = spec["dim"]
            block = encoded_matrix[:, curr_idx:curr_idx + dim]
            kind  = spec.get("kind", "continuous")

            if spec["type"] == "numeric":
                raw = block.ravel()
                lo  = spec["orig_min"]
                hi  = spec["orig_max"]

                if kind == "binary":
                    # Clamp to [0, 1] then round to nearest integer
                    out = np.clip(raw, 0.0, 1.0)
                    out = np.round(out).astype(np.int32)

                elif kind == "binary_int":
                    unique_vals = sorted({lo, hi})
                    mid = (unique_vals[0] + unique_vals[1]) / 2.0
                    out = np.where(raw >= mid, unique_vals[1], unique_vals[0]).astype(np.int32)

                elif kind == "count":
                    # Non-negative integer (Pregnancies, Age when integer, etc.)
                    out = np.clip(np.round(raw), 0.0, hi).astype(np.int32)

                elif kind == "integer":
                    out = np.clip(np.round(raw), lo, hi).astype(np.int32)

                else:
                    # Continuous: only clip to realistic range
                    out = np.clip(raw, lo, hi).astype(np.float32)

                if spec.get("transform") == "log1p":
                    out = np.expm1(np.maximum(0.0, out))
                    raw_lo = spec.get("raw_min", 0.0)
                    raw_hi = spec.get("raw_max", 1e9)
                    out = np.clip(out, raw_lo, raw_hi)
                    if "int" in spec.get("orig_dtype", ""):
                        out = np.round(out).astype(np.int32)

                df_dict[col] = out

            elif spec["type"] == "categorical_ohe":
                ohe = self.encoders[col]
                # Normalize logits to valid probability simplex, then argmax
                block_softmax = np.exp(block - block.max(axis=1, keepdims=True))
                block_softmax /= block_softmax.sum(axis=1, keepdims=True)
                # Hard one-hot
                hard_ohe = np.zeros_like(block_softmax)
                hard_ohe[np.arange(len(block_softmax)), block_softmax.argmax(axis=1)] = 1.0
                decoded_cats = ohe.inverse_transform(hard_ohe).ravel()
                df_dict[col] = decoded_cats

            elif spec["type"] == "categorical_ord":
                ord_enc = self.encoders[col]
                clipped = np.clip(np.round(block), 0, len(spec["categories"]) - 1)
                decoded_cats = ord_enc.inverse_transform(clipped).ravel()
                df_dict[col] = decoded_cats

            curr_idx += dim
 
        df_res = pd.DataFrame(df_dict)
        # Gate 2b Item 7: Confirm decoding restores NaN when the *_isna indicator is 1
        # and that no raw 0.0 fill values reach the output.
        for col in list(df_res.columns):
            isna_col = f"{col}_isna"
            if isna_col in df_res.columns:
                isna_mask = df_res[isna_col] >= 0.5
                if isna_mask.any():
                    if pd.api.types.is_numeric_dtype(df_res[col]):
                        if not pd.api.types.is_float_dtype(df_res[col]):
                            df_res[col] = df_res[col].astype(float)
                    df_res.loc[isna_mask, col] = np.nan

        return df_res
