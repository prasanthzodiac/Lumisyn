from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def infer_task_type(df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Any]:
    """
    Infer the dataset learning task type from the target column characteristics.

    Returns:
        {
          "task": "classification" | "regression" | "observational",
          "confidence": float,
          "num_classes": int or None,
          "target_is_binary": bool,
          "rationale": str
        }
    """
    if not target_col or target_col not in df.columns:
        return {
            "task": "observational",
            "confidence": 0.90,
            "num_classes": None,
            "target_is_binary": False,
            "rationale": "No identifiable target column — observational dataset."
        }

    target = df[target_col].dropna()
    n_unique = target.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target)

    if not is_numeric:
        # String target -> classification
        return {
            "task": "classification",
            "confidence": 0.95,
            "num_classes": n_unique,
            "target_is_binary": n_unique == 2,
            "rationale": f"Categorical target with {n_unique} classes."
        }

    # Numeric target
    is_integer_valued = np.all(target.values == np.floor(target.values))
    if n_unique <= 2 and is_integer_valued:
        return {
            "task": "classification",
            "confidence": 0.99,
            "num_classes": 2,
            "target_is_binary": True,
            "rationale": "Binary integer target — binary classification."
        }
    if n_unique <= 20 and is_integer_valued:
        return {
            "task": "classification",
            "confidence": 0.90,
            "num_classes": n_unique,
            "target_is_binary": False,
            "rationale": f"Integer target with {n_unique} distinct values — multi-class classification."
        }
    # High cardinality numeric -> regression
    return {
        "task": "regression",
        "confidence": 0.85,
        "num_classes": None,
        "target_is_binary": False,
        "rationale": f"Continuous target ({n_unique} unique values) — regression."
    }


class AnalyticalMetaFeatureExtractor:
    """
    1.8 & 1.9 Analytical Meta-Feature Extractor (UDIE Stage 1) — Honest Naming.
    Computes deterministic statistical summaries (sparsity, cardinality, moments)
    projected into a fixed-width vector I in R^256.
    Note: This is an analytical meta-feature extractor, NOT a pre-trained foundation model.
    Includes:
    - Task-type inference (classification / regression / observational)
    - Analytical complexity vector C
    - Projected feature summary fingerprint I in R^256
    """

    def __init__(self, fingerprint_dim: int = 256):
        self.fingerprint_dim = fingerprint_dim

    def analyze_and_encode(
        self,
        df: pd.DataFrame,
        dependency_matrix: np.ndarray,
        sensitivity_vector: np.ndarray,
        target_col: str = None,
    ) -> Dict[str, Any]:
        num_rows, num_cols = df.shape

        # --- Task-type inference ---
        task_info = infer_task_type(df, target_col)

        # --- Complexity vector ---
        ir = 1.0
        if target_col and target_col in df.columns:
            val_counts = df[target_col].value_counts()
            if len(val_counts) > 1:
                ir = float(val_counts.max() / max(1, val_counts.min()))

        dep_shape = dependency_matrix.shape
        if dep_shape[0] == num_cols and dep_shape[1] == num_cols:
            upper_tri = dependency_matrix[np.triu_indices(num_cols, k=1)]
            redundancy = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        else:
            redundancy = 0.0

        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        if numeric_df.shape[1] >= 2 and num_rows > 10:
            std_vals = numeric_df.std().values
            non_zero_stds = std_vals[std_vals > 1e-5]
            intrinsic_dim = float(len(non_zero_stds))
        else:
            intrinsic_dim = float(num_cols)

        missing_rate = float(df.isna().mean().mean())

        complexity_vector = np.array([
            ir,
            redundancy,
            intrinsic_dim,
            float(num_rows),
            float(num_cols),
            float(np.mean(sensitivity_vector)),
            missing_rate,
            float(task_info["num_classes"] or 0),
        ], dtype=np.float32)

        # --- Deterministic summary projection fingerprint ---
        feature_stats = []
        for i, col in enumerate(df.columns):
            s = df[col]
            missing_p = float(s.isna().mean())
            unique_r = float(s.nunique() / max(1, num_rows))
            sens = float(sensitivity_vector[i]) if i < len(sensitivity_vector) else 0.1
            is_num = 1.0 if pd.api.types.is_numeric_dtype(s) else 0.0

            if is_num:
                vals = s.dropna()
                mean_v = float(vals.mean()) if len(vals) > 0 else 0.0
                std_v = float(vals.std()) if len(vals) > 0 else 1.0
                skew_v = float(vals.skew()) if len(vals) > 0 else 0.0
            else:
                mean_v, std_v, skew_v = 0.0, 0.0, 0.0

            f_vec = np.array(
                [missing_p, unique_r, sens, is_num, mean_v, std_v, skew_v],
                dtype=np.float32
            )
            feature_stats.append(f_vec)

        feature_matrix = np.array(feature_stats)  # (d, 7)

        rng = np.random.RandomState(42)
        W_proj = rng.randn(7, self.fingerprint_dim).astype(np.float32) / np.sqrt(7)

        projected = np.tanh(np.dot(feature_matrix, W_proj))  # (d, 256)
        fingerprint = np.mean(projected, axis=0)              # (256,)
        fingerprint[:len(complexity_vector)] += 0.1 * complexity_vector
        fingerprint = fingerprint / (np.linalg.norm(fingerprint) + 1e-8)

        return {
            "complexity_vector": complexity_vector,
            "imbalance_ratio": ir,
            "redundancy": redundancy,
            "intrinsic_dimension": intrinsic_dim,
            "missing_rate": missing_rate,
            "fingerprint": fingerprint,
            "task_info": task_info,
        }


# Backwards compatibility alias
DatasetIntelligenceGenerator = AnalyticalMetaFeatureExtractor
