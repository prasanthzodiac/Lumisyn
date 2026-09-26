"""
Matched-epsilon comparison against external DP synthesizers on Adult (eps = 0.741, delta = 1e-5).

    python lumisyn/backend/baselines_eval.py --prepare     # write split + schema, run our variants
    .venv_baselines/Scripts/python lumisyn/backend/baselines_gen.py
    python lumisyn/backend/baselines_eval.py               # score everything -> runs/baselines/results.json

Same split as verify_icfm_sensitivity.py (15k sample, 80/20, seed 42). Metrics: TSTR AUC (RF),
mean total-variation distance of all 1-way and 2-way marginals on public bins (the standard
workload error used by AIM/MST), correlation Frobenius error, mean DCR.
"""

import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_icfm_sensitivity as v  # noqa: E402  (sets sys.path for acfm_net)
from baselines_gen import discretise  # noqa: E402
from eval_utils import evaluate_tstr, compute_anti_memorization_dcr, correlation_frobenius_error  # noqa: E402

ROOT = v.REPO_ROOT / "runs" / "baselines"
OURS = {"ours_icfm": dict(enable_dp_recalibration=False),
        "ours_icfm_7b": dict(enable_dp_recalibration=True, recalibration_epsilon=0.10)}


def split():
    df, schema = v.load_adult(v.N_ROWS)
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    return tr.reset_index(drop=True), te.reset_index(drop=True), schema


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    tr, te, schema = split()
    tr.to_csv(ROOT / "train.csv", index=False)
    (ROOT / "schema.json").write_text(json.dumps(schema, indent=2))
    from acfm_net.pipeline import ACFMNetV5Pipeline
    timings = json.loads((ROOT / "timings.json").read_text()) if (ROOT / "timings.json").exists() else {}
    for name, kw in OURS.items():
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            pipe = ACFMNetV5Pipeline(target_epsilon=1.0, target_delta=1e-5, latent_mode="identity",
                                     public_schema=schema, **kw)
            syn, rep = pipe.fit_generate(df_raw=tr, num_samples=len(tr), target_col="class", epochs=5,
                                         protocol_id=f"BASE-{name}-{seed}")
            syn.to_csv(ROOT / f"syn_{name}_{seed}.csv", index=False)
            timings[f"{name}_{seed}"] = round(time.time() - t0, 1)
            timings[f"{name}_{seed}_epsilon"] = rep["privacy_summary"]["prv_upper_with_laplace"]
            (ROOT / "timings.json").write_text(json.dumps(timings, indent=2))


def marginal_tvd(real_codes: pd.DataFrame, syn_codes: pd.DataFrame, order: int) -> float:
    errs = []
    for cols in itertools.combinations(real_codes.columns, order):
        cols = list(cols)
        p = real_codes.groupby(cols).size() / len(real_codes)
        q = syn_codes.groupby(cols).size() / len(syn_codes)
        errs.append(0.5 * p.subtract(q, fill_value=0).abs().sum())
    return float(np.mean(errs))


def evaluate():
    tr, te, schema = split()
    real_codes = discretise(tr, schema)
    timings = {}
    for name in ("timings.json", "timings_gen.json"):
        if (ROOT / name).exists():
            timings.update(json.loads((ROOT / name).read_text()))
    rows = defaultdict(list)
    for f in sorted(ROOT.glob("syn_*.csv")):
        method, seed = f.stem[4:].rsplit("_", 1)
        syn = pd.read_csv(f)[tr.columns]
        syn_codes = discretise(syn, schema)
        u = evaluate_tstr(tr, te, syn, "class", "classification")
        rows[method].append({
            "seed": int(seed), "time_s": timings.get(f"{method}_{seed}"),
            "tstr_auc": u["tstr_auc"], "trtr_auc": u["trtr_auc"],
            "tvd_1way": marginal_tvd(real_codes, syn_codes, 1), "tvd_2way": marginal_tvd(real_codes, syn_codes, 2),
            "corr_err": correlation_frobenius_error(tr, syn), "mean_dcr": compute_anti_memorization_dcr(tr, syn)[0],
        })
        print(f"{method:14s} seed={seed} AUC={u['tstr_auc']:.3f} TVD1={rows[method][-1]['tvd_1way']:.3f} "
              f"TVD2={rows[method][-1]['tvd_2way']:.3f}", flush=True)
    summary = {}
    for method, rs in rows.items():
        summary[method] = {"n_seeds": len(rs)}
        for m in ("tstr_auc", "tvd_1way", "tvd_2way", "corr_err", "mean_dcr", "time_s"):
            a = np.array([r[m] for r in rs if r[m] is not None], dtype=float)
            summary[method][m] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}
    summary["_trtr_auc"] = rows[next(iter(rows))][0]["trtr_auc"]
    (ROOT / "results.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    prepare() if "--prepare" in sys.argv else evaluate()
