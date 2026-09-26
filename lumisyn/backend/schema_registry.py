"""
Public-schema catalog for L-OTFM. A `public_schema` dict declares, per column,
a bound/category vocabulary the pipeline is allowed to treat as non-private
(zero epsilon). Without one, ACFMNetV5Pipeline falls back to a generic
+/-1000 / +/-10000 default for any unrecognized numeric column, which silently
corrupts that column's generated range (see report/RESEARCH_PAPER_BENCHMARK_REPORT.md
Corrigendum, Sec 6). This module centralizes known-dataset schemas so every
caller (benchmark runner, dashboard, future scripts) uses the same declared
bounds instead of re-deriving them ad hoc.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIR = REPO_ROOT / "lumisyn" / "data" / "kaggle_datasets"


def _numeric(lo: float, hi: float, kind: str = "continuous") -> Dict[str, Any]:
    return {"type": "numeric", "bounds": [float(lo), float(hi)], "kind": kind}


def _categorical(categories: List[str]) -> Dict[str, Any]:
    cats = sorted({str(c) for c in categories})
    if "MISSING" not in cats:
        cats.append("MISSING")
    return {"type": "categorical", "categories": cats}


# Known public benchmarks: (display name, file, target column, task, sample cap, schema builder)
def _pima_schema(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "Pregnancies": _numeric(0, 20),
        "Glucose": _numeric(0, 200),
        "BloodPressure": _numeric(0, 140),
        "SkinThickness": _numeric(0, 100),
        "Insulin": _numeric(0, 900),
        "BMI": _numeric(0, 70),
        "DiabetesPedigreeFunction": _numeric(0.0, 2.5),
        "Age": _numeric(18, 90),
        "Outcome": _numeric(0, 1, kind="binary"),
    }


def _breast_cancer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    # Canonical public reference copy of this exact 30-feature Wisconsin
    # diagnostic dataset (scikit-learn), used only for its documented
    # per-feature min/max — not the private benchmark split itself.
    from sklearn.datasets import load_breast_cancer
    ref = load_breast_cancer(as_frame=True).frame
    schema = {
        col: _numeric(float(ref[col].min()), float(ref[col].max()))
        for col in ref.columns if col != "target"
    }
    schema["target"] = _numeric(0, 1, kind="binary")
    return schema


def _bank_churn_schema(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "CreditScore": _numeric(300, 900),
        "Geography": _categorical(["France", "Spain", "Germany"]),
        "Gender": _categorical(["Male", "Female"]),
        "Age": _numeric(18, 100),
        "Tenure": _numeric(0, 10),
        "Balance": _numeric(0, 300000),
        "Num Of Products": _numeric(1, 4),
        "Has Credit Card": _numeric(0, 1, kind="binary"),
        "Is Active Member": _numeric(0, 1, kind="binary"),
        "Estimated Salary": _numeric(0, 200000),
        "Churn": _numeric(0, 1, kind="binary"),
    }


def _actuarial_schema(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "age": _numeric(18, 64),
        "sex": _categorical(["male", "female"]),
        "bmi": _numeric(10, 60),
        "children": _numeric(0, 5),
        "smoker": _categorical(["yes", "no"]),
        "region": _categorical(["southwest", "southeast", "northwest", "northeast"]),
        "charges": _numeric(0, 80000),
    }


KNOWN_DATASETS: List[Dict[str, Any]] = [
    {
        "key": "pima",
        "name": "Pima Indians Diabetes (Kaggle)",
        "file": KAGGLE_DIR / "pima_diabetes.csv",
        "target": "Outcome",
        "task": "classification",
        "sample_n": None,
        "drop_cols": [],
        "schema_fn": _pima_schema,
    },
    {
        "key": "wisconsin",
        "name": "Breast Cancer Wisconsin (High-Dimensional 31-D)",
        "file": KAGGLE_DIR / "breast_cancer_wisconsin.csv",
        "target": "target",
        "task": "classification",
        "sample_n": None,
        "drop_cols": [],
        "schema_fn": _breast_cancer_schema,
    },
    {
        "key": "churn",
        "name": "Kaggle Bank Customer Churn (Heterogeneous Mixed-Type)",
        "file": KAGGLE_DIR / "bank_churn_kaggle.csv",
        "target": "Churn",
        "task": "classification",
        "sample_n": 2000,
        "drop_cols": ["CustomerId", "Surname"],
        "schema_fn": _bank_churn_schema,
    },
    {
        "key": "actuarial",
        "name": "Actuarial Healthcare & Medical Risk",
        "file": KAGGLE_DIR / "stroke_prediction_kaggle.csv",
        "target": "charges",
        "task": "regression",
        "sample_n": None,
        "drop_cols": [],
        "schema_fn": _actuarial_schema,
    },
]


def known_schema_for(dataset_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Backwards-compatible name-prefix lookup (used by run_hard_stress_test.py)."""
    for ds in KNOWN_DATASETS:
        if dataset_name.startswith(ds["name"].split(" (")[0]):
            return ds["schema_fn"](df)
    raise ValueError(f"No declared public schema for dataset: {dataset_name}")


def infer_schema_for_dataframe(df: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    """
    Best-effort schema for an arbitrary/unknown CSV, for interactive terminal use only.
    Returns (schema, notes) where notes flags every column whose bounds came from the
    observed sample rather than a declared public source, so the operator can review
    them before trusting downstream privacy/utility numbers.
    """
    schema: Dict[str, Any] = {}
    notes: List[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            observed_lo = float(series.min()) if series.notna().any() else 0.0
            observed_hi = float(series.max()) if series.notna().any() else 1.0
            is_binary = series.dropna().isin([0, 1]).all()
            kind = "binary" if is_binary else ("integer" if pd.api.types.is_integer_dtype(series) else "continuous")
            schema[col] = _numeric(observed_lo, observed_hi, kind=kind)
            notes.append(f"{col}: numeric, auto bounds [{observed_lo:g}, {observed_hi:g}] from observed sample (UNVERIFIED)")
        else:
            categories = sorted({str(v) for v in series.dropna().unique()})
            schema[col] = _categorical(categories)
            notes.append(f"{col}: categorical, {len(categories)} observed categories (UNVERIFIED)")
    return schema, notes
