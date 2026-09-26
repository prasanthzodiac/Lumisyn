"""
DAPC robustness analysis (paper §DAPC). DAPC is a deterministic function of public metadata,
so it can be evaluated exactly, without training:

1. nominal eps* for the 11 stress-test datasets (same public inputs the pipeline passes);
2. Monte-Carlo robustness to the hand-set constants: weights ~ Dirichlet(50 w), gammas x U(0.8, 1.2),
   beta ~ U(2, 3); report the eps* spread per dataset and Kendall tau of the dataset ranking;
3. one-at-a-time elasticities d eps* / d w_k (renormalised weights).

Run: python lumisyn/backend/dapc_sensitivity.py   ->  runs/dapc_sensitivity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_adaptive_stress_test as ast_  # noqa: E402  (sets sys.path for acfm_net)
from acfm_net.governance import ClinicalDataGovernanceLayer  # noqa: E402
from acfm_net.preparation import AdaptiveImputer, DistributionPreservingEncoder  # noqa: E402
from acfm_net.privacy.budget_allocator import AdaptiveBudgetAllocator  # noqa: E402
from acfm_net.intelligence.sensitive import SensitiveAttributeDetector  # noqa: E402


def public_inputs(df: pd.DataFrame, target: str, schema) -> dict:
    """Exactly the public metadata fit_generate passes to analyze_dataset_dapc."""
    _, _, dd = ClinicalDataGovernanceLayer().process_governance(df, protocol_id="DAPC", run_deidentification=True)
    di, _ = AdaptiveImputer().fit_transform(dd, {}, privacy_mode=True, public_schema=schema)
    _, info = DistributionPreservingEncoder().fit_transform(di, privacy_schema=schema)
    s = SensitiveAttributeDetector().detect(dd.head(1), use_data_dependent_risk=False)["sensitivity_vector"]
    return {"sensitivity_vector": s, "raw_n": len(dd), "num_samples": int(round(len(dd) / 50.0) * 50),
            "num_features": len(info["encoded_column_names"]),
            "is_high_stakes": any(k in target.lower() for k in ["mortality", "death", "cancer", "icu"])}


def eps_star(alloc: AdaptiveBudgetAllocator, p: dict) -> float:
    r = alloc.compute_risk_vector(p["sensitivity_vector"], p["num_samples"], p["num_features"], p["is_high_stakes"])
    return alloc.select_policy_bounded_epsilon(alloc.aggregate_risk(r))


def main():
    inputs = {}
    for ds in ast_.KNOWN_DATASETS:
        df = pd.read_csv(ds["file"]).drop(columns=[c for c in ds["drop_cols"]], errors="ignore")
        if ds["sample_n"] is not None and len(df) > ds["sample_n"]:
            df = df.sample(n=ds["sample_n"], random_state=42).reset_index(drop=True)
        inputs[ds["name"]] = public_inputs(df, ds["target"], ds["schema_fn"](df))
    for spec in ast_.SYNTHETIC_SPECS:
        df = ast_._make_synthetic(spec["n"], spec["d"], spec["task"], seed=42)
        yb = (float(df["target"].min()) - 50, float(df["target"].max()) + 50) if spec["task"] == "regression" else (0, 1)
        inputs[spec["name"]] = public_inputs(df, "target", ast_._synthetic_schema(spec["d"], spec["task"], yb))
    inputs = {k: v for k, v in inputs.items() if v["raw_n"] >= AdaptiveBudgetAllocator.MIN_COHORT_SIZE}  # gate uses raw n

    nominal_alloc = AdaptiveBudgetAllocator(target_epsilon=1.0)
    names = list(inputs)
    nominal = np.array([eps_star(nominal_alloc, inputs[n]) for n in names])

    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        a = AdaptiveBudgetAllocator(target_epsilon=1.0, beta=float(rng.uniform(2.0, 3.0)))
        a.weights = rng.dirichlet(50 * nominal_alloc.weights)
        a.gammas = nominal_alloc.gammas * rng.uniform(0.8, 1.2, size=4)
        draws.append([eps_star(a, inputs[n]) for n in names])
    draws = np.array(draws)
    taus = [kendalltau(nominal, d).statistic for d in draws]

    elasticity = {}
    for k, label in enumerate(["w_sens", "w_scale", "w_dim", "w_task"]):
        a = AdaptiveBudgetAllocator(target_epsilon=1.0)
        w = nominal_alloc.weights.copy()
        w[k] *= 1.10
        a.weights = w / w.sum()
        elasticity[label] = float(np.max(np.abs(np.array([eps_star(a, inputs[n]) for n in names]) - nominal)))

    out = {
        "datasets": {n: {"rounded_n": inputs[n]["num_samples"], "d_encoded": inputs[n]["num_features"],
                         "eps_nominal": round(float(nominal[i]), 4),
                         "eps_p05": round(float(np.percentile(draws[:, i], 5)), 4),
                         "eps_p95": round(float(np.percentile(draws[:, i], 95)), 4),
                         "max_abs_dev": round(float(np.max(np.abs(draws[:, i] - nominal[i]))), 4)}
                     for i, n in enumerate(names)},
        "perturbation": "weights ~ Dirichlet(50 w0), gammas x U(0.8,1.2), beta ~ U(2,3); 2000 draws",
        "kendall_tau_median": float(np.median(taus)), "kendall_tau_p05": float(np.percentile(taus, 5)),
        "max_eps_change_for_10pct_weight_bump": elasticity,
    }
    for n, row in out["datasets"].items():
        print(f"{n:42s} n={row['rounded_n']:6d} d={row['d_encoded']:3d}  eps*={row['eps_nominal']:.3f} "
              f"[{row['eps_p05']:.3f}, {row['eps_p95']:.3f}]")
    print(f"Kendall tau vs nominal ranking: median {out['kendall_tau_median']:.3f}, 5th pct {out['kendall_tau_p05']:.3f}")
    print("Max |d eps*| for +10% on one weight:", elasticity)
    path = ast_.REPO_ROOT / "runs" / "dapc_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
