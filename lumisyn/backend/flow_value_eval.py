"""
Does the flow add value? Pilot/full study for the "adaptive allocation + flow matching" claim.

Metrics that can see dependence the class-conditional marginals cannot carry:
  - allcol_acc: for every column, a random forest trained on synthetic data predicts that column from all others
    (public bins); accuracy on the real test split, averaged over columns. Reference: the same on real training data.
  - tvd_3way_label: mean TVD of all (target, feature_a, feature_b) marginals, i.e. within-class feature dependence.
Also the usual TSTR AUC and 2-way TVD.

    python lumisyn/backend/flow_value_eval.py --dataset adult [--eps 1.0] [--seeds 3] [--configs a,b]
Output: runs/flow_value/results.jsonl (resumable).
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dapc_split_eval as dse  # noqa: E402  (sets sys.path; load(), tvd())
from acfm_net.pipeline import ACFMNetV5Pipeline  # noqa: E402
from baselines_gen import discretise  # noqa: E402
from eval_utils import evaluate_tstr  # noqa: E402

OUT = dse.v.REPO_ROOT / "runs" / "flow_value" / "results.jsonl"
CONFIGS = {
    "marginals_only_0.9": dict(conditional_marginals=0.9, conditional_copula="independent"),
    "flow_0.9": dict(conditional_marginals=0.9),
    "warm_0.9": dict(conditional_marginals=0.9, pretrain_on_marginals=True),
    "warm_0.5": dict(conditional_marginals=0.5, pretrain_on_marginals=True),
    "warm_0.2": dict(conditional_marginals=0.2, pretrain_on_marginals=True),
    "warm_0.9_pretrain_only": dict(conditional_marginals=0.9, pretrain_on_marginals=True, dp_finetune=False),
    "warm_0.2_pretrain_only": dict(conditional_marginals=0.2, pretrain_on_marginals=True, dp_finetune=False),
}


def allcol_acc(train_codes, test_codes, cols):
    accs = []
    for c in cols:
        y = train_codes[c]
        if y.nunique() < 2:
            continue
        clf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0)
        clf.fit(train_codes.drop(columns=[c]), y)
        accs.append(float((clf.predict(test_codes.drop(columns=[c])) == test_codes[c]).mean()))
    return float(np.mean(accs))


def main():
    name = sys.argv[sys.argv.index("--dataset") + 1]
    eps = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else 1.0
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
    only = sys.argv[sys.argv.index("--configs") + 1].split(",") if "--configs" in sys.argv else list(CONFIGS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["dataset"], r["config"], r["seed"], r["target_eps"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()

    df, schema, target, epochs = dse.load(name)
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    trc, tec = discretise(tr, schema), discretise(te, schema)
    cols = list(trc.columns)
    feats = [c for c in cols if c != target]
    triples = [(target, a, b) for a, b in itertools.combinations(feats, 2)]
    ref = {"allcol_acc_trtr": allcol_acc(trc, tec, cols)}
    print(f"REF {name} all-column accuracy (real->real) = {ref['allcol_acc_trtr']:.4f}", flush=True)

    for cfg, seed in itertools.product(only, range(seeds)):
        if (name, cfg, seed, eps) in done:
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        pipe = ACFMNetV5Pipeline(target_epsilon=eps, target_delta=1e-5, latent_mode="identity", public_schema=schema, **CONFIGS[cfg])
        syn, rep = pipe.fit_generate(df_raw=tr, num_samples=len(tr), target_col=target, epochs=epochs, protocol_id=f"FV-{name}-{cfg}-{seed}")
        syn = syn[tr.columns]
        sc = discretise(syn, schema)
        row = {"dataset": name, "config": cfg, "seed": seed, "target_eps": eps, "time_s": round(time.time() - t0, 1),
               "epsilon": rep["privacy_summary"]["prv_upper_with_laplace"], "verified": bool(rep["release_privacy_verified"]),
               "tstr_auc": evaluate_tstr(tr, te, syn, target, "classification")["tstr_auc"],
               "allcol_acc": allcol_acc(sc, tec, cols), **ref,
               "tvd_2way": dse.tvd(trc, sc, list(itertools.combinations(cols, 2))),
               "tvd_3way_label": dse.tvd(trc, sc, triples)}
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"RESULT {name} eps={eps} {cfg} seed={seed} AUC={row['tstr_auc']:.3f} allcol={row['allcol_acc']:.4f} "
              f"TVD2={row['tvd_2way']:.4f} TVD3={row['tvd_3way_label']:.4f} {row['time_s']}s", flush=True)


if __name__ == "__main__":
    main()
