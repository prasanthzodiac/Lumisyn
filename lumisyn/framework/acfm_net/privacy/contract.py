"""End-to-end privacy contract for data-dependent pipeline operations.

The strict path treats the dataset schema (column names, declared types,
clinical bounds, and category vocabulary) as public metadata.  All learned
statistics are either removed from preprocessing or passed through a bounded
DP mechanism before they influence the graph or budget controller.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class PrivacyLedger:
    target_epsilon: float
    target_delta: float
    entries: list[Dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, epsilon: float, delta: float, mechanism: str):
        self.entries.append({
            "name": name,
            "epsilon": float(epsilon),
            "delta": float(delta),
            "mechanism": mechanism,
        })

    @property
    def epsilon(self) -> float:
        return float(sum(item["epsilon"] for item in self.entries))

    @property
    def delta(self) -> float:
        return float(sum(item["delta"] for item in self.entries))

    def verify(self) -> bool:
        return bool(
            np.isfinite(self.epsilon)
            and np.isfinite(self.delta)
            and self.epsilon <= self.target_epsilon + 1e-8
            and self.delta <= self.target_delta + 1e-12
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "composition": "basic sequential composition",
            "entries": self.entries,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "verified": self.verify(),
        }


def _clinical_bounds(column: str) -> Optional[tuple[float, float]]:
    name = str(column).lower()
    rules = {
        "age": (0.0, 120.0), "bmi": (10.0, 80.0),
        "systolic": (50.0, 250.0), "diastolic": (30.0, 150.0),
        "glucose": (0.0, 1000.0), "hba1c": (0.0, 20.0),
        "cholesterol": (0.0, 600.0), "hdl": (0.0, 250.0),
        "heart_rate": (20.0, 250.0), "creatinine": (0.0, 30.0),
        "lactate": (0.0, 40.0), "sofa": (0.0, 30.0),
        "poverty": (0.0, 25.0), "ratio": (0.0, 25.0),
    }
    for token, bounds in rules.items():
        if token in name:
            return bounds
    return None


def public_schema_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a declared schema without using observed numeric extrema.
    Only ~13 clinical-vitals column names are recognized; every other numeric
    column falls back to a generic +/-1000 / +/-10000 bound, which is almost
    certainly wrong for that column's real domain and will corrupt its
    generated range. Callers running on non-clinical data (or any column this
    doesn't recognize) MUST pass an explicit `public_schema=` instead of
    relying on this fallback — see backend/schema_registry.py for examples.
    """
    schema: Dict[str, Any] = {}
    fallback_columns = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            bounds = _clinical_bounds(column)
            if bounds is None:
                bounds = (-1000.0, 1000.0) if pd.api.types.is_integer_dtype(series) else (-10000.0, 10000.0)
                fallback_columns.append(str(column))
            schema[column] = {
                "type": "numeric",
                "bounds": list(bounds),
                "kind": "binary" if series.dropna().isin([0, 1]).all() else "continuous",
            }
        else:
            # Category vocabulary is an explicit schema assumption. It is not
            # inferred as a released statistic and is recorded in the report.
            categories = sorted({str(value) for value in series.dropna().unique()})
            if "MISSING" not in categories:
                categories.append("MISSING")
            schema[column] = {
                "type": "categorical",
                "categories": categories,
            }
    if fallback_columns:
        warnings.warn(
            f"public_schema_from_dataframe(): {len(fallback_columns)} column(s) "
            f"not in the recognized clinical-vitals list got a generic, almost "
            f"certainly wrong +/-1000 / +/-10000 bound: {fallback_columns}. "
            f"Pass an explicit public_schema= with real bounds for these columns.",
            RuntimeWarning,
            stacklevel=2,
        )
    return schema


def sanitize_dependency_matrix(matrix: np.ndarray, epsilon: float, n_rows: int, rng=None) -> np.ndarray:
    """
    Bounded Laplace mechanism for dependency matrix M in [0, 1]^{d x d}.

    Sensitivity Derivation (Gate 2b Item 6):
    -----------------------------------------
    1. The matrix M is symmetric (M_jk = M_kj) with fixed constant diagonal M_jj = 1.0.
       Thus, there are exactly K = d*(d-1)/2 unique non-trivial query coordinates in the upper triangle.
    2. For any two neighboring datasets D, D' differing by one patient row:
       Each pairwise bounded association entry m_jk in [0, 1] computed over N rows has
       single-entry sensitivity Delta_entry <= 2.0 / N (worst-case bounded deviation of 1 patient).
    3. The vector of all K released upper-triangle entries has L1 global sensitivity:
       Delta_1 = sum_{1 <= j < k <= d} |m_jk(D) - m_jk(D')| <= K * Delta_entry = (d*(d-1)/2) * (2.0 / N) = d*(d-1) / N.
    4. By the vector Laplace mechanism theorem (Dwork & Roth 2014, Thm 3.6), adding independent
       Laplace noise with scale b = Delta_1 / epsilon to each unique upper-triangle entry guarantees
       that the entire released matrix satisfies (epsilon, 0)-differential privacy in TOTAL.
    5. Privacy Cost: epsilon (e.g. 0.08 * target_epsilon) is the TOTAL privacy budget charged for
       releasing the entire dependency matrix, NOT a per-entry charge.
    """
    # Unseeded by default: a fixed seed in public code would make the noise a known constant.
    rng = rng or np.random.default_rng()
    values = np.asarray(matrix, dtype=np.float64).copy()
    d = values.shape[0]
    if d <= 1:
        return np.ones((d, d), dtype=np.float32)

    n_unique_pairs = (d * (d - 1)) / 2.0
    entry_sensitivity = 2.0 / max(1, int(n_rows))
    # Total L1 global sensitivity across all unique released entries:
    total_l1_sensitivity = n_unique_pairs * entry_sensitivity
    scale = total_l1_sensitivity / max(float(epsilon), 1e-8)

    # Generate independent Laplace noise scaled to the TOTAL L1 sensitivity:
    noise_upper = rng.laplace(0.0, scale, size=values.shape)
    # Mirror upper triangle symmetrically to lower triangle, zero on diagonal
    noise_sym = np.triu(noise_upper, k=1)
    noise_sym = noise_sym + noise_sym.T

    noisy = np.clip(values + noise_sym, 0.0, 1.0)
    np.fill_diagonal(noisy, 1.0)
    return noisy.astype(np.float32)
