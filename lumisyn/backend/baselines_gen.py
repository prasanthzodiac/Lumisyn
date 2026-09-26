"""
External DP baselines on the paper's Adult split, run inside the isolated smartnoise venv:

    .venv_baselines/Scripts/python lumisyn/backend/baselines_gen.py [method ...]

Reads runs/baselines/{train.csv,schema.json} written by baselines_eval.py --prepare and
writes runs/baselines/syn_<method>_<seed>.csv. Every method gets the same public schema our
framework gets: numeric columns are discretised on PUBLIC bounds (32 bins, log-spaced for
heavy-tailed columns), categories come from the schema, and no epsilon is spent on
preprocessing (preprocessor_eps=0). Decoding samples uniformly inside the generated bin.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "runs" / "baselines"
EPS, DELTA, SEEDS, BINS = 0.741, 1e-5, 3, 32
METHODS = {
    "mst": dict(epsilon=EPS, delta=DELTA),
    "aim": dict(epsilon=EPS, delta=DELTA),
    "dpctgan": dict(epsilon=EPS, delta=DELTA, epochs=50),
    "patectgan": dict(epsilon=EPS, delta=DELTA),
}


def heavy_tailed(spec):
    lo, hi = spec["bounds"]
    return lo >= 0 and hi / max(lo, 1.0) > 1000


def edges_for(spec):
    lo, hi = spec["bounds"]
    if heavy_tailed(spec):  # bin log1p on public bounds
        return np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), BINS + 1))
    return np.linspace(lo, hi, BINS + 1)


def discretise(df, schema):
    out = {}
    for col, spec in schema.items():
        if spec["type"] == "categorical":
            cats = spec["categories"]
            fallback = cats.index("MISSING") if "MISSING" in cats else 0
            idx = {c: i for i, c in enumerate(cats)}
            out[col] = df[col].fillna("MISSING").astype(str).map(lambda x: idx.get(x, fallback)).astype(int)
        else:
            e = edges_for(spec)
            x = pd.to_numeric(df[col], errors="coerce").fillna(spec["bounds"][0]).clip(e[0], e[-1])
            out[col] = np.clip(np.searchsorted(e, x, side="right") - 1, 0, BINS - 1)
    return pd.DataFrame(out)


def undiscretise(codes, schema, rng):
    out = {}
    for col, spec in schema.items():
        c = codes[col].astype(int).to_numpy()
        if spec["type"] == "categorical":
            cats = spec["categories"]
            out[col] = [cats[min(max(i, 0), len(cats) - 1)] for i in c]
        else:
            e = edges_for(spec)
            c = np.clip(c, 0, BINS - 1)
            vals = rng.uniform(e[c], e[c + 1])
            integral = spec.get("kind") in ("integer", "count") or heavy_tailed(spec)
            out[col] = np.round(vals) if integral else vals
    return pd.DataFrame(out)


def make_aim_target(target):
    """AIM restricted to the label x feature workload: the fair, target-aware counterpart of Stage 7c."""
    from snsynth.aim import AIMSynthesizer

    class AIMTarget(AIMSynthesizer):
        def get_workload(self, data, degree, max_cells, num_marginals=None):
            return [cl for cl in itertools.combinations(data.domain, 2) if target in cl]

    return AIMTarget(epsilon=EPS, delta=DELTA)


def main():
    from snsynth import Synthesizer  # venv-only; keeps discretise() importable elsewhere

    schema = json.loads((ROOT / "schema.json").read_text())
    train = pd.read_csv(ROOT / "train.csv")
    codes = discretise(train, schema)
    timings = json.loads((ROOT / "timings_gen.json").read_text()) if (ROOT / "timings_gen.json").exists() else {}
    for method in (sys.argv[1:] or list(METHODS)):
        for seed in range(SEEDS):
            np.random.seed(seed)
            t0 = time.time()
            synth = make_aim_target(f"col{list(codes.columns).index('class')}") if method == "aim_target" else Synthesizer.create(method, **METHODS[method])
            synth.fit(codes, categorical_columns=list(codes.columns), preprocessor_eps=0.0)
            syn_codes = pd.DataFrame(synth.sample(len(train)), columns=codes.columns)
            syn = undiscretise(syn_codes, schema, np.random.default_rng(seed))
            syn.to_csv(ROOT / f"syn_{method}_{seed}.csv", index=False)
            timings[f"{method}_{seed}"] = round(time.time() - t0, 1)
            print(f"{method} seed={seed} done in {timings[f'{method}_{seed}']}s", flush=True)
            (ROOT / "timings_gen.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
