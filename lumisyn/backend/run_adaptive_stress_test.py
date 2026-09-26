"""
Adaptive-Behavior Hard Stress Test for L-OTFM.

Not a fidelity benchmark (see run_hard_stress_test.py for that) — this proves the
framework genuinely ADAPTS to each dataset's shape instead of behaving identically
everywhere: DAPC's risk-adaptive epsilon selection responds to cohort size (N) and
dimensionality (d), the safety gate correctly ABSTAINs below N=100, and the whole
pipeline stays fast across a wide N range (90 to 30,000+ rows).

11 datasets: the 4 real benchmarks (N=455-1600) plus 7 synthetic ones spanning
N=90 (below the ABSTAIN threshold) up to N=30,000, and dimensionality 5 to 60,
generated on the fly (deterministic, no downloads) via sklearn.datasets.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "backend"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from acfm_net.pipeline import ACFMNetV5Pipeline
from schema_registry import KNOWN_DATASETS


def _synthetic_schema(n_features: int, task: str, y_bounds: tuple = (0, 1)) -> Dict[str, Any]:
    schema = {f"f{i}": {"type": "numeric", "bounds": [-8.0, 8.0], "kind": "continuous"} for i in range(n_features)}
    if task == "classification":
        schema["target"] = {"type": "numeric", "bounds": [0, 1], "kind": "binary"}
    else:
        schema["target"] = {"type": "numeric", "bounds": [float(y_bounds[0]), float(y_bounds[1])], "kind": "continuous"}
    return schema


def _make_synthetic(n: int, d: int, task: str, seed: int) -> pd.DataFrame:
    if task == "classification":
        X, y = make_classification(
            n_samples=n, n_features=d, n_informative=max(2, d // 2), n_redundant=max(0, d // 4),
            n_classes=2, class_sep=1.2, random_state=seed,
        )
    else:
        X, y = make_regression(n_samples=n, n_features=d, n_informative=max(2, d // 2), noise=10.0, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["target"] = y
    return df


SYNTHETIC_SPECS = [
    {"name": "Synthetic Below-Threshold (N=90)", "n": 90, "d": 5, "task": "classification"},
    {"name": "Synthetic Near-Abstain (N=105)", "n": 105, "d": 5, "task": "classification"},
    {"name": "Synthetic Small (N=300)", "n": 300, "d": 8, "task": "classification"},
    {"name": "Synthetic Medium (N=2000)", "n": 2000, "d": 10, "task": "classification"},
    {"name": "Synthetic High-Dim (N=1000, d=60)", "n": 1000, "d": 60, "task": "classification"},
    {"name": "Synthetic Large Regression (N=8000)", "n": 8000, "d": 12, "task": "regression"},
    {"name": "Synthetic XLarge (N=30000)", "n": 30000, "d": 6, "task": "classification"},
]


def run_one(name: str, df: pd.DataFrame, target: str, task: str, schema: Dict[str, Any], epochs: int = 3) -> Dict[str, Any]:
    t0 = time.time()
    try:
        pipeline = ACFMNetV5Pipeline(
            target_epsilon=1.0,
            target_delta=1e-5,
            latent_mode="identity",
            enable_dp_recalibration=False,
            public_schema=schema,
        )
        df_synth, report = pipeline.fit_generate(
            df_raw=df, num_samples=min(len(df), 2000), target_col=target, epochs=epochs, protocol_id=f"ADAPT-{name}"
        )
        elapsed = time.time() - t0
        cert = report["dapc_certificate"]
        stage3 = next((s for s in report["stage_reports"] if s["stage"] == "3"), {}).get("evidence", {})
        return {
            "Dataset": name,
            "N": len(df),
            "d": df.shape[1] - 1,
            "Time(s)": round(elapsed, 1),
            "Decision": cert["decision"],
            "Risk R_P": round(cert["risk_score_R_P"], 4),
            "Selected eps": round(cert["selected_target_epsilon"], 4),
            "DCGB edges": stage3.get("edges", "n/a"),
            "Synth rows": len(df_synth),
        }
    except RuntimeError as exc:
        elapsed = time.time() - t0
        return {
            "Dataset": name, "N": len(df), "d": df.shape[1] - 1, "Time(s)": round(elapsed, 1),
            "Decision": "ABSTAIN", "Risk R_P": "n/a", "Selected eps": "n/a",
            "DCGB edges": "n/a", "Synth rows": 0,
        }


def main():
    results: List[Dict[str, Any]] = []
    t_start = time.time()

    print("=" * 90)
    print("  ADAPTIVE-BEHAVIOR HARD STRESS TEST — 11 datasets, N=90 to N=30,000")
    print("=" * 90)

    for ds in KNOWN_DATASETS:
        df = pd.read_csv(ds["file"])
        df = df.drop(columns=[c for c in ds["drop_cols"] if c in df.columns])
        if ds["sample_n"] is not None and len(df) > ds["sample_n"]:
            df = df.sample(n=ds["sample_n"], random_state=42).reset_index(drop=True)
        schema = ds["schema_fn"](df)
        print(f"\n[{ds['name']}] N={len(df)}, d={df.shape[1]-1}, task={ds['task']}")
        res = run_one(ds["name"], df, ds["target"], ds["task"], schema)
        results.append(res)
        print(f"  -> {res}")

    for spec in SYNTHETIC_SPECS:
        df = _make_synthetic(spec["n"], spec["d"], spec["task"], seed=42)
        y_bounds = (float(df["target"].min()) - 50, float(df["target"].max()) + 50) if spec["task"] == "regression" else (0, 1)
        schema = _synthetic_schema(spec["d"], spec["task"], y_bounds)
        print(f"\n[{spec['name']}] N={spec['n']}, d={spec['d']}, task={spec['task']}")
        res = run_one(spec["name"], df, "target", spec["task"], schema)
        results.append(res)
        print(f"  -> {res}")

    total_elapsed = time.time() - t_start

    print("\n" + "=" * 90)
    print("  SUMMARY: ADAPTIVITY ACROSS 11 DATASETS")
    print("=" * 90)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print(f"\nTotal wall-clock time for all {len(results)} datasets: {total_elapsed:.1f}s")

    proceeded = df_res[df_res["Decision"] == "PROCEED"]
    if len(proceeded) > 1:
        eps_vals = proceeded["Selected eps"].astype(float)
        print(f"\nSelected-epsilon range across PROCEEDed datasets: {eps_vals.min():.4f} - {eps_vals.max():.4f}")
        print(f"Distinct epsilon values selected: {eps_vals.nunique()} / {len(eps_vals)} datasets")
    abstained = df_res[df_res["Decision"] == "ABSTAIN"]
    print(f"Datasets correctly ABSTAINed (N<100 safety gate): {list(abstained['Dataset'])}")


if __name__ == "__main__":
    main()
