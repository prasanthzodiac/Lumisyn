"""
Applies the pre-registered analysis (runs/flow_value/PREREGISTRATION.md) to runs/flow_value/results.jsonl.

  composite = mean of min-max-normalised [TSTR AUC, all-column acc, -TVD2, -TVD3] within each (dataset, eps) cell
  rule      = fitted on Adult only (best rho per eps, then a monotone threshold rule in eps*)
  criterion = on Cardio, averaged over eps, adaptive composite > every fixed rho; and DP fine-tuning beats
              the warm-only ablation on all-column accuracy where the rule gives the flow a real share.

    python lumisyn/backend/flow_value_analysis.py  ->  runs/flow_value/analysis.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "runs" / "flow_value"
rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text().splitlines()]
SPLITS = {"warm_0.2": 0.2, "warm_0.5": 0.5, "warm_0.9": 0.9}
CELL_CONFIGS = ["marginals_only_0.9", "warm_0.2", "warm_0.5", "warm_0.9", "warm_0.2_pretrain_only"]
METRICS = [("tstr_auc", 1), ("allcol_acc", 1), ("tvd_2way", -1), ("tvd_3way_label", -1)]

cell = defaultdict(lambda: defaultdict(list))  # pre-registered grid: seeds 0-2 only
allseeds = defaultdict(lambda: defaultdict(list))  # Addendum A: every seed
eps_star = {}
for r in rows:
    allseeds[(r["dataset"], r["target_eps"])][r["config"]].append(r)
    if r["config"] in CELL_CONFIGS and r["seed"] < 3:
        cell[(r["dataset"], r["target_eps"])][r["config"]].append(r)
        eps_star[(r["dataset"], r["target_eps"])] = r["epsilon"]


def composite(ds, eps):
    c = cell[(ds, eps)]
    means = {cfg: {m: float(np.mean([r[m] for r in rs])) for m, _ in METRICS} for cfg, rs in c.items()}
    out = {}
    for cfg in means:
        parts = []
        for m, sign in METRICS:
            vals = [sign * means[k][m] for k in means]
            lo, hi = min(vals), max(vals)
            parts.append(0.5 if hi == lo else (sign * means[cfg][m] - lo) / (hi - lo))
        out[cfg] = float(np.mean(parts))
    return out, means


result = {"cells": {}}
for (ds, eps) in sorted(cell):
    comp, means = composite(ds, eps)
    result["cells"][f"{ds}@{eps}"] = {"eps_star": eps_star[(ds, eps)], "configs": sorted(comp), "composite": comp, "means": means}

# Fit on Adult only: best warm split per eps, then a monotone step rule in eps* with midpoint thresholds.
adult_eps = sorted(e for d, e in cell if d == "adult")
best_adult = {e: max(SPLITS, key=lambda k: result["cells"][f"adult@{e}"]["composite"].get(k, -1.0)) for e in adult_eps}
pts = sorted((eps_star[("adult", e)], SPLITS[best_adult[e]]) for e in adult_eps)


def rule(e_star):
    rho = pts[0][1]
    for (x0, _), (x1, r1) in zip(pts, pts[1:]):
        if e_star >= (x0 + x1) / 2:
            rho = r1
    return rho


result["adult_best_split_per_eps"] = {str(e): best_adult[e] for e in adult_eps}
result["rule_points_eps_star_to_rho"] = pts

# Evaluate on Cardio (held out).
card_eps = sorted(e for d, e in cell if d == "cardio")
ev = {"adaptive": [], **{k: [] for k in SPLITS}}
chosen = {}
for e in card_eps:
    comp = result["cells"][f"cardio@{e}"]["composite"]
    k = next(k for k, v in SPLITS.items() if v == rule(eps_star[("cardio", e)]))
    chosen[str(e)] = k
    ev["adaptive"].append(comp[k])
    for kk in SPLITS:
        ev[kk].append(comp[kk])
summary = {k: float(np.mean(v)) for k, v in ev.items()}
result["cardio_rule_choices"] = chosen
result["cardio_mean_composite"] = summary
beats_all = all(summary["adaptive"] > summary[k] + 1e-12 for k in SPLITS)

# Does DP fine-tuning (private data) beat the warm-only ablation on all-column accuracy?
ft = {}
for (ds, eps), c in cell.items():
    if "warm_0.2" in c and "warm_0.2_pretrain_only" in c:
        a = [r["allcol_acc"] for r in c["warm_0.2"]]
        b = [r["allcol_acc"] for r in c["warm_0.2_pretrain_only"]]
        ft[f"{ds}@{eps}"] = {"finetune": float(np.mean(a)), "warm_only": float(np.mean(b)), "diff": float(np.mean(a) - np.mean(b)),
                             "ranges_disjoint": bool(min(a) > max(b))}
result["finetune_vs_warm_only_allcol"] = ft
flow_eps = [e for e in card_eps if SPLITS[chosen[str(e)]] < 0.9]
ft_ok = bool(flow_eps) and all(ft.get(f"cardio@{e}", {}).get("diff", -1.0) > 0 for e in flow_eps)
result["criterion"] = {"adaptive_beats_every_fixed_split_on_cardio": beats_all,
                       "finetune_helps_where_flow_gets_budget": ft_ok, "claim_supported": bool(beats_all and ft_ok)}

# Addendum A (10-seed confirmation): one-sided Welch t on all-column accuracy, Holm over the 6 cells; TOST on AUC.
from scipy import stats  # noqa: E402


def holm(ps):
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


conf = {}
for (ds, eps), c in sorted(allseeds.items()):
    if not all(k in c and len(c[k]) >= 10 for k in ["warm_0.9", "marginals_only_0.9"]):
        continue
    w, m = c["warm_0.9"], c["marginals_only_0.9"]
    wo = c.get("warm_0.9_pretrain_only", [])
    g = lambda rs, k: np.array([r[k] for r in rs])  # noqa: E731
    d_auc = g(w, "tstr_auc").mean() - g(m, "tstr_auc").mean()
    tost = max(stats.ttest_ind(g(w, "tstr_auc") + 0.01, g(m, "tstr_auc"), equal_var=False, alternative="greater").pvalue,
               stats.ttest_ind(g(w, "tstr_auc") - 0.01, g(m, "tstr_auc"), equal_var=False, alternative="less").pvalue)
    conf[f"{ds}@{eps}"] = {
        "n": [len(w), len(m), len(wo)], "eps_star": w[0]["epsilon"], "ref_allcol": w[0]["allcol_acc_trtr"],
        **{f"{k}_{name}": [float(g(rs, k).mean()), float(g(rs, k).std(ddof=1))] for k in ["allcol_acc", "tstr_auc", "tvd_2way", "tvd_3way_label"]
           for name, rs in [("warm", w), ("marg", m), ("warm_only", wo)] if len(rs) >= 2},
        "H1_diff": float(g(w, "allcol_acc").mean() - g(m, "allcol_acc").mean()),
        "H1_p": float(stats.ttest_ind(g(w, "allcol_acc"), g(m, "allcol_acc"), equal_var=False, alternative="greater").pvalue),
        "H2_diff": float(g(w, "allcol_acc").mean() - g(wo, "allcol_acc").mean()) if len(wo) >= 2 else None,
        "H2_p": float(stats.ttest_ind(g(w, "allcol_acc"), g(wo, "allcol_acc"), equal_var=False, alternative="greater").pvalue) if len(wo) >= 2 else 1.0,
        "auc_diff": float(d_auc), "auc_tost_p": float(tost)}
if conf:
    keys = list(conf)
    for h in ["H1", "H2"]:
        for k, a in zip(keys, holm([conf[k][f"{h}_p"] for k in keys])):
            conf[k][f"{h}_p_holm"] = a
    result["confirmation"] = {"cells": conf,
                              "H1_cells_passed": sum(conf[k]["H1_p_holm"] < 0.05 for k in keys),
                              "H2_cells_passed": sum(conf[k]["H2_p_holm"] < 0.05 for k in keys),
                              "auc_equivalent_cells": sum(conf[k]["auc_tost_p"] < 0.05 for k in keys), "n_cells": len(keys)}
    print(json.dumps({k: v for k, v in result["confirmation"].items() if k != "cells"}, indent=2))
(ROOT / "analysis.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ["adult_best_split_per_eps", "rule_points_eps_star_to_rho", "cardio_rule_choices",
                                         "cardio_mean_composite", "criterion"]}, indent=2))
print(json.dumps(ft, indent=2))
