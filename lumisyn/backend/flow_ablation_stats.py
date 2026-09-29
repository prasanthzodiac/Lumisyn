"""
Headline statistics for the negative result: full pipeline (share 0.9, flow copula) vs marginals only
(share 0.9, independent copula), paired by seed. "No difference" is tested as equivalence (paired TOST,
margin +-0.01 AUC and +-0.005 TVD), not as a non-significant difference.

    python lumisyn/backend/flow_ablation_stats.py   ->  runs/dapc_split/flow_ablation_stats.json
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2] / "runs" / "dapc_split"
MARGINS = {"tstr_auc": 0.01, "tvd_2way_all": 0.005, "tvd_2way_feature_feature": 0.005}


def tost(d: np.ndarray, margin: float) -> float:
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    p_low = 1 - stats.t.cdf((m + margin) / se, n - 1)   # H0: mean <= -margin
    p_high = stats.t.cdf((m - margin) / se, n - 1)       # H0: mean >= +margin
    return float(max(p_low, p_high))


def main():
    rows = [json.loads(line) for line in (ROOT / "results.jsonl").read_text().splitlines()]
    out = {}
    for ds in ("adult", "cardio"):
        full = {r["seed"]: r for r in rows if r["dataset"] == ds and r["config"] == "share_0.9" and r.get("target_eps", 1.0) == 1.0}
        marg = {r["seed"]: r for r in rows if r["dataset"] == ds and r["config"] == "share_0.9_marginals_only" and r.get("target_eps", 1.0) == 1.0}
        seeds = sorted(set(full) & set(marg))
        out[ds] = {"n_seeds": len(seeds)}
        for metric, margin in MARGINS.items():
            a = np.array([full[s][metric] for s in seeds])
            b = np.array([marg[s][metric] for s in seeds])
            d = a - b
            ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
            out[ds][metric] = {"full_mean": float(a.mean()), "full_std": float(a.std(ddof=1)),
                               "marginals_only_mean": float(b.mean()), "marginals_only_std": float(b.std(ddof=1)),
                               "diff_mean": float(d.mean()), "diff_ci95": [float(ci[0]), float(ci[1])],
                               "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
                               "equivalence_margin": margin, "tost_p": tost(d, margin),
                               "diff_sd": float(d.std(ddof=1)), "df": len(d) - 1,
                               "tost_t_lower": float((d.mean() + margin) / (d.std(ddof=1) / np.sqrt(len(d)))),
                               "tost_t_upper": float((d.mean() - margin) / (d.std(ddof=1) / np.sqrt(len(d)))),
                               "per_seed": {int(s_): [float(x), float(y)] for s_, x, y in zip(seeds, a, b)}}
    (ROOT / "flow_ablation_stats.json").write_text(json.dumps(out, indent=2))
    for ds, r in out.items():
        for metric in MARGINS:
            m = r[metric]
            print(f"{ds:7s} {metric:25s} full {m['full_mean']:.4f}  marg-only {m['marginals_only_mean']:.4f}  "
                  f"diff {m['diff_mean']:+.4f} CI [{m['diff_ci95'][0]:+.4f},{m['diff_ci95'][1]:+.4f}]  "
                  f"Wilcoxon p={m['wilcoxon_p']:.3f}  TOST p={m['tost_p']:.4f} (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
