"""
Multi-Dataset Hard Stress Test Runner for Research-Grade L-OTFM Framework.
Executes rigorous end-to-end evaluation across 4 completely distinct external datasets:
1. Pima Indians Diabetes (Clinical Mixed - Kaggle Benchmark)
2. Breast Cancer Wisconsin (Dense High-Dimensional 31-D Oncology)
3. Kaggle Bank Churn (Heterogeneous Mixed Categorical + Financial, 2,000 records)
4. Actuarial Healthcare / Medical Cost (Demographic + Health Risk Mixed Regression)

Evaluates:
- Statistical Fidelity (Wasserstein-1 Distance, Correlation Matrix Frobenius Error)
- Downstream Utility (TSTR - Train on Synthetic, Test on Real - ROC-AUC, Accuracy, R2)
- Privacy & Anti-Memorization (Nearest Neighbor DCR, Identical Record Rate = 0.00%)
- Formal Differential Privacy Guarantee (Opacus PRV eps <= 1.0, Analytical RDP composition)
- DP Marginal Recalibration Impact
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "backend"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from schema_registry import known_schema_for, KNOWN_DATASETS
from eval_utils import evaluate_tstr, compute_anti_memorization_dcr, correlation_frobenius_error

build_public_schema = known_schema_for  # backwards-compatible alias

DATASETS = [
    {
        "name": ds["name"],
        "file": ds["file"],
        "target": ds["target"],
        "task": ds["task"],
        "sample_n": ds["sample_n"],
        "drop_cols": ds["drop_cols"],
        "epochs": 10,
    }
    for ds in KNOWN_DATASETS
]


def _run_once(ds, df_train, df_test, public_schema, seed, d_idx):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    pipeline = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        latent_mode="identity",
        enable_dp_recalibration=True,
        recalibration_epsilon=0.10,
        public_schema=public_schema,
    )
    df_synthetic, report = pipeline.fit_generate(
        df_raw=df_train,
        num_samples=len(df_train),
        target_col=ds["target"],
        epochs=ds["epochs"],
        protocol_id=f"STRESS-TEST-{d_idx}-s{seed}",
    )
    utility = evaluate_tstr(df_train, df_test, df_synthetic, ds["target"], ds["task"])
    mean_dcr, exact_match_pct = compute_anti_memorization_dcr(df_train, df_synthetic)
    key = "auc" if ds["task"] == "classification" else "r2"
    return {
        "seed": seed,
        "time_s": time.time() - t0,
        "tstr": utility[f"tstr_{key}"],
        "trtr": utility[f"trtr_{key}"],
        "corr_err": correlation_frobenius_error(df_train, df_synthetic),
        "mean_dcr": mean_dcr,
        "exact_copy_pct": exact_match_pct,
        "epsilon": report["privacy_summary"]["prv_upper_with_laplace"],
        "privacy_verified": bool(report["release_privacy_verified"]),
    }


def run_hard_stress_test(n_seeds: int = 5):
    print("=" * 85)
    print("      RESEARCH-GRADE L-OTFM MULTI-DATASET HARD STRESS TEST SUITE")
    print(f"      {n_seeds} seeds per dataset; mean +- std reported")
    print("=" * 85)

    results_table = []
    detailed_reports = {}

    for d_idx, ds in enumerate(DATASETS, 1):
        print(f"\n[{d_idx}/{len(DATASETS)}] BENCHMARK DATASET: {ds['name']}")
        if not os.path.exists(ds["file"]):
            print(f"      [WARNING] File not found: {ds['file']}, skipping.")
            continue

        df_raw = pd.read_csv(ds["file"])
        # Drop non-generative identifier columns (row IDs, free-text names).
        df_raw = df_raw.drop(columns=[c for c in ds["drop_cols"] if c in df_raw.columns])
        if ds["sample_n"] is not None and len(df_raw) > ds["sample_n"]:
            df_raw = df_raw.sample(n=ds["sample_n"], random_state=42).reset_index(drop=True)
        df_train, df_test = train_test_split(df_raw, test_size=0.20, random_state=42)
        df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
        public_schema = build_public_schema(ds["name"], df_train)

        runs = [_run_once(ds, df_train, df_test, public_schema, seed, d_idx) for seed in range(n_seeds)]
        agg = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs], ddof=1)) if n_seeds > 1 else 0.0)
               for k in ("tstr", "trtr", "corr_err", "mean_dcr", "exact_copy_pct", "epsilon", "time_s")}
        metric = "AUC" if ds["task"] == "classification" else "R2"
        fmt = lambda k, p=3: f"{agg[k][0]:.{p}f} +- {agg[k][1]:.{p}f}"  # noqa: E731
        res = {
            "Dataset": ds["name"],
            "Rows(train)": len(df_train),
            "Seeds": n_seeds,
            f"TSTR": f"{metric} {fmt('tstr')}",
            "TRTR": f"{metric} {agg['trtr'][0]:.3f}",
            "Corr Error": fmt("corr_err", 4),
            "Mean DCR": fmt("mean_dcr"),
            "Exact Copy %": f"{agg['exact_copy_pct'][0]:.2f}%",
            "Epsilon": f"{agg['epsilon'][0]:.4f}",
            "Privacy Verified": all(r["privacy_verified"] for r in runs),
        }
        results_table.append(res)
        detailed_reports[ds["name"]] = {"summary": res, "aggregate": agg, "runs": runs}
        print(f"      Done: TSTR {res['TSTR']} (TRTR {res['TRTR']})")

    print("\n" + "=" * 85)
    print("                         SUMMARY OF STRESS TEST RESULTS")
    print("=" * 85)
    print(pd.DataFrame(results_table).to_string(index=False))
    print("\nRun complete. 'Privacy Verified' = accountant epsilon within DAPC budget; it says nothing about utility.")
    print("=" * 85)

    out_file = REPO_ROOT / "runs" / "hard_stress_test_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary_table": results_table, "details": detailed_reports}, f, indent=2)
    print(f"Saved complete stress test telemetry to {out_file}")


if __name__ == "__main__":
    run_hard_stress_test(int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5)
