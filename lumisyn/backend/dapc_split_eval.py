"""
DAPC v2 evaluation: adaptive budget split between Stage 7c class-conditional marginals and
DP-SGD flow matching, against fixed splits and ablations, at the DAPC-selected epsilon.

    python lumisyn/backend/dapc_split_eval.py --dataset adult|cardio|pima [--seeds 3] [--eps 1.0] [--configs a,b]

Resumable: appends one JSON line per (dataset, config, seed) to runs/dapc_split/results.jsonl and
skips finished ones. One dataset per invocation keeps each GPU job short.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_icfm_sensitivity as v  # noqa: E402  (sets sys.path for acfm_net)
from acfm_net.pipeline import ACFMNetV5Pipeline  # noqa: E402
from baselines_gen import discretise  # noqa: E402
from eval_utils import evaluate_tstr  # noqa: E402
from schema_registry import KNOWN_DATASETS, known_schema_for  # noqa: E402

OUT = v.REPO_ROOT / "runs" / "dapc_split" / "results.jsonl"

CONFIGS = {
    "flow_only": dict(),
    "flow_7b_1way": dict(enable_dp_recalibration=True, recalibration_epsilon=0.10),
    "share_0.2": dict(conditional_marginals=0.2),
    "share_0.4": dict(conditional_marginals=0.4),
    "share_0.6": dict(conditional_marginals=0.6),
    "share_0.9": dict(conditional_marginals=0.9),
    "share_0.9_marginals_only": dict(conditional_marginals=0.9, conditional_copula="independent"),
    "adaptive": dict(conditional_marginals="adaptive"),
    "adaptive_marginals_only": dict(conditional_marginals="adaptive", conditional_copula="independent"),
    "adaptive_nm4_h64": dict(conditional_marginals="adaptive", noise_multiplicity=4, flow_hidden_dim=64),
}

# Cardio (Kaggle cardiovascular disease): bounds from clinical plausibility, not from the data.
CARDIO_SCHEMA = {
    "age": {"type": "numeric", "bounds": [10000.0, 25000.0], "kind": "continuous"},  # days
    "gender": {"type": "numeric", "bounds": [1, 2], "kind": "integer"},
    "height": {"type": "numeric", "bounds": [100.0, 220.0], "kind": "continuous"},
    "weight": {"type": "numeric", "bounds": [30.0, 200.0], "kind": "continuous"},
    "ap_hi": {"type": "numeric", "bounds": [60.0, 250.0], "kind": "continuous"},
    "ap_lo": {"type": "numeric", "bounds": [30.0, 180.0], "kind": "continuous"},
    "cholesterol": {"type": "numeric", "bounds": [1, 3], "kind": "integer"},
    "gluc": {"type": "numeric", "bounds": [1, 3], "kind": "integer"},
    "smoke": {"type": "numeric", "bounds": [0, 1], "kind": "binary"},
    "alco": {"type": "numeric", "bounds": [0, 1], "kind": "binary"},
    "active": {"type": "numeric", "bounds": [0, 1], "kind": "binary"},
    "cardio": {"type": "numeric", "bounds": [0, 1], "kind": "binary"},
}


def load(name: str):
    if name == "adult":
        df, schema = v.load_adult(v.N_ROWS)
        return df, schema, "class", 5
    if name == "cardio":
        df = pd.read_csv(v.REPO_ROOT / "lumisyn" / "data" / "kaggle_datasets" / "cardio_train.csv").drop(columns=["id"])
        return df.sample(n=20000, random_state=42).reset_index(drop=True), CARDIO_SCHEMA, "cardio", 5
    ds = next(d for d in KNOWN_DATASETS if "Pima" in d["name"])
    df = pd.read_csv(ds["file"])
    return df, known_schema_for(ds["name"], df), ds["target"], 10


def tvd(real_codes, syn_codes, cols_list):
    errs = []
    for cols in cols_list:
        cols = list(cols)
        p = real_codes.groupby(cols).size() / len(real_codes)
        q = syn_codes.groupby(cols).size() / len(syn_codes)
        errs.append(0.5 * p.subtract(q, fill_value=0).abs().sum())
    return float(np.mean(errs))


def main():
    name = sys.argv[sys.argv.index("--dataset") + 1]
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
    eps = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else 1.0
    only = sys.argv[sys.argv.index("--configs") + 1].split(",") if "--configs" in sys.argv else list(CONFIGS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["dataset"], r["config"], r["seed"], r.get("target_eps", 1.0)) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()

    df, schema, target, epochs = load(name)
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    real_codes = discretise(tr, schema)
    feats = [c for c in real_codes.columns if c != target]
    pairs_all = list(itertools.combinations(real_codes.columns, 2))
    pairs_ff = list(itertools.combinations(feats, 2))
    pairs_lf = [(target, f) for f in feats]

    for cfg, seed in itertools.product(only, range(seeds)):
        if (name, cfg, seed, eps) in done:
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        pipe = ACFMNetV5Pipeline(target_epsilon=eps, target_delta=1e-5, latent_mode="identity",
                                 public_schema=schema, **CONFIGS[cfg])
        syn, rep = pipe.fit_generate(df_raw=tr, num_samples=len(tr), target_col=target, epochs=epochs,
                                     protocol_id=f"SPLIT-{name}-{cfg}-{seed}")
        syn = syn[tr.columns]
        u = evaluate_tstr(tr, te, syn, target, "classification")
        sc = discretise(syn, schema)
        row = {"dataset": name, "config": cfg, "seed": seed, "target_eps": eps, "time_s": round(time.time() - t0, 1),
               "epsilon": rep["privacy_summary"]["prv_upper_with_laplace"], "verified": bool(rep["release_privacy_verified"]),
               "sigma_flow": rep["privacy_training"]["noise_multiplier"],
               "cm": rep.get("dapc_certificate", {}).get("conditional_marginals"),
               "tstr_auc": u["tstr_auc"], "trtr_auc": u["trtr_auc"],
               "tvd_1way": tvd(real_codes, sc, [(c,) for c in real_codes.columns]),
               "tvd_2way_all": tvd(real_codes, sc, pairs_all),
               "tvd_2way_label_feature": tvd(real_codes, sc, pairs_lf),
               "tvd_2way_feature_feature": tvd(real_codes, sc, pairs_ff)}
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"RESULT {name} {cfg} seed={seed} AUC={row['tstr_auc']:.3f} TVD2ff={row['tvd_2way_feature_feature']:.3f} "
              f"TVD2lf={row['tvd_2way_label_feature']:.3f} eps={row['epsilon']:.3f} {row['time_s']}s", flush=True)


if __name__ == "__main__":
    main()
