"""
Empirical privacy audit for the paper (UCI Adult, public schema, d=78).

Part A — stronger adversary. Extends verify_icfm_sensitivity.worst_case_sensitivity: after
screening candidate replacement records, hill-climb the replacement over VALID encoded records
(one-hot blocks stay {0,1} one-hot, binary flags stay +-1, numerics stay in public bounds) to maximise
the exact clipped-gradient-sum sensitivity ||S(B) - S(B')||_2 / 2C.

Part B — white-box step audit (Nasr et al. 2021, 2023). For the worst (B, B') pair found in A,
run the actual Gaussian mechanism M(B) = S(B) + N(0, sigma^2 C^2 I) many times under each
hypothesis, apply the Neyman-Pearson test (projection on S(B') - S(B)), and turn Clopper-Pearson
bounds on FPR/FNR into a 95%-confidence lower bound on the Gaussian-DP parameter mu. DP-SGD's
analysis claims mu <= 2/sigma for a replace-one step; mu_lb > 2/sigma is a statistically
certified violation.

Part C — end-to-end black-box one-run canary audit (Steinke, Nasr, Jagielski, NeurIPS 2023).
m random domain records ("canaries") are each included in training with prob 1/2; the auditor
sees only the synthetic data, scores each canary by -distance to its nearest synthetic record,
and guesses IN/OUT for the k highest/lowest scores. The number of correct guesses gives a
lower bound on epsilon (pure-DP form of their Theorem; delta=1e-5 with m=1000 changes it
negligibly).

Run:  python lumisyn/backend/audit_privacy.py [--part A|C|all]   (B always follows A)
Out:  runs/privacy_audit.json
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_dist, binom, norm
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_icfm_sensitivity as v  # noqa: E402  (sets sys.path for acfm_net)

from acfm_net.pipeline import ACFMNetV5Pipeline  # noqa: E402
from acfm_net.generative.ot_cfm import ContinuousOTCFM  # noqa: E402
from acfm_net.governance import ClinicalDataGovernanceLayer  # noqa: E402
from acfm_net.preparation import AdaptiveImputer, DistributionPreservingEncoder, AdaptiveFeatureScaler  # noqa: E402

OUT = v.REPO_ROOT / "runs" / "privacy_audit.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adult training configuration used throughout the paper (identity mode, 5 epochs, batch 512).
N_TRAIN, BATCH_TRAIN, EPOCHS_TRAIN, DELTA_TRAIN, EPS_TRAIN = 12000, 512, 5, 8e-6, 0.741


# ---------------------------------------------------------------- encoding helpers
def encode(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[torch.Tensor, List[Tuple[int, int, str]]]:
    """Row-local public-schema encoding (same as the pipeline) plus the block layout."""
    _, _, dd = ClinicalDataGovernanceLayer().process_governance(df, protocol_id="AUDIT", run_deidentification=True)
    di, _ = AdaptiveImputer().fit_transform(dd, {}, privacy_mode=True, public_schema=schema)
    enc = DistributionPreservingEncoder()
    X, info = enc.fit_transform(di, privacy_schema=schema)
    S = AdaptiveFeatureScaler().fit_transform(X, info["feature_specs"], privacy_mode=True,
                                              encoded_column_names=info["encoded_column_names"])
    blocks, s = [], 0
    for col in enc.column_order:
        spec = enc.feature_specs[col]
        k = spec.get("dim", 1)
        typ = "ohe" if spec.get("type") == "categorical_ohe" and k > 1 else ("binary" if spec.get("kind") == "binary" else "num")
        blocks.append((s, k, typ))
        s += k
    return torch.tensor(S, dtype=torch.float32), blocks


def project(x: torch.Tensor, blocks) -> torch.Tensor:
    """Snap an arbitrary vector onto the valid encoded domain."""
    x = x.clamp(-1.0, 1.0).clone()
    for s, k, typ in blocks:
        if typ == "ohe":
            i = int(x[s:s + k].argmax())
            x[s:s + k] = 0.0
            x[s + i] = 1.0
        elif typ == "binary":
            x[s] = 1.0 if x[s] >= 0 else -1.0
    return x


def mutate(x: torch.Tensor, blocks, rng) -> torch.Tensor:
    y = x.clone()
    for _ in range(int(rng.integers(1, 4))):
        s, k, typ = blocks[int(rng.integers(len(blocks)))]
        if typ == "ohe":
            y[s:s + k] = 0.0
            y[s + int(rng.integers(k))] = 1.0
        elif typ == "binary":
            y[s] = -y[s]
        else:
            y[s] = float(np.clip(float(y[s]) + rng.normal(0, 0.5), -1.0, 1.0))
    return y


# ---------------------------------------------------------------- Part A
def stronger_adversary(X: torch.Tensor, blocks, climb_steps: int = 150, screen: int = 200) -> Dict[str, Any]:
    d, B = X.shape[1], v.BATCH
    res = {"batch_ot": [], "i_cfm": []}
    worst = {"batch_ot": (0.0, None, None), "i_cfm": (0.0, None, None)}  # (ratio, diff, C)
    for seed in range(v.SENS_SEEDS):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ContinuousOTCFM(continuous_dim=d, cond_dim=0, hidden_dim=128)
        params = [p for p in model.velocity_net.parameters() if p.requires_grad]
        for _ in range(v.TRIALS_PER_SEED):
            idx = rng.choice(len(X), size=B, replace=False)
            z1, z0, t = X[idx].clone(), torch.randn(B, d), torch.rand(B, 1)
            j = int(rng.integers(B))
            g = lambda i, rec, C: v._clipped_grad(model, params, z0[i:i+1], rec.unsqueeze(0), t[i:i+1], C)  # noqa: E731

            _, z1c = model.solve_minibatch_ot(z0, z1)
            C = 0.5 * min(g(i, z1c[i], 1e9)[1] for i in range(B))  # all-clipped regime
            base_ot = [g(i, z1c[i], C)[0] for i in range(B)]
            base_j = g(j, z1[j], C)[0]

            def f_ot(cand):
                z1p = z1.clone()
                z1p[j] = cand
                _, z1cp = model.solve_minibatch_ot(z0, z1p)
                diff = torch.zeros_like(base_ot[0])
                for i in range(B):
                    if not torch.equal(z1c[i], z1cp[i]):
                        diff += g(i, z1cp[i], C)[0] - base_ot[i]
                return float(diff.norm()) / (2 * C), diff

            def f_icfm(cand):
                diff = g(j, cand, C)[0] - base_j
                return float(diff.norm()) / (2 * C), diff

            # Screen valid candidates (real records, projected prior draws, random valid records)
            # by how many other records they reassign; climb from the best few.
            real = np.setdiff1d(rng.choice(len(X), size=screen + B, replace=False), idx)[:screen]
            pool = [X[r] for r in real] + [project(z0[k], blocks) for k in range(B) if k != j]
            pool += [project(torch.randn(d), blocks) for _ in range(screen // 4)]

            def n_reassigned(cand):
                z1p = z1.clone()
                z1p[j] = cand
                return sum(not torch.equal(a, b) for a, b in zip(z1c, model.solve_minibatch_ot(z0, z1p)[1]))

            starts = sorted(pool, key=lambda c: -n_reassigned(c))[:3]
            for mode, f in (("batch_ot", f_ot), ("i_cfm", f_icfm)):
                best_val, best_diff, best_x = max((f(c) + (c,) for c in starts), key=lambda r: r[0])
                for _ in range(climb_steps):
                    y = mutate(best_x, blocks, rng)
                    val, diff = f(y)
                    if val > best_val:
                        best_val, best_diff, best_x = val, diff, y
                res[mode].append(best_val)
                if best_val > worst[mode][0]:
                    worst[mode] = (best_val, best_diff.detach().clone(), C)
        print(f"  [A] seed {seed}: OT max so far {max(res['batch_ot']):.3f}, I-CFM {max(res['i_cfm']):.3f}", flush=True)

    out = {"adversary": f"top-3 of {screen} real + {B-1} projected-prior + {screen // 4} random valid records by "
                        f"reassignment count, then {climb_steps}-step hill climb over valid records",
           "n_trials": len(res["i_cfm"])}
    for mode, vals in res.items():
        a = np.array(vals)
        out[mode] = {"sensitivity_over_2C_mean": float(a.mean()), "sensitivity_over_2C_std": float(a.std()),
                     "sensitivity_over_2C_max": float(a.max()), "fraction_exceeding_bound": float(np.mean(a > 1 + 1e-6))}
    return out, worst


# ---------------------------------------------------------------- Part B
def training_sigma() -> float:
    from opacus.accountants.utils import get_noise_multiplier
    q = BATCH_TRAIN / N_TRAIN
    steps = EPOCHS_TRAIN * math.ceil(N_TRAIN / BATCH_TRAIN)
    return float(get_noise_multiplier(target_epsilon=EPS_TRAIN, target_delta=DELTA_TRAIN, sample_rate=q,
                                      steps=steps, accountant="prv"))


def gdp_eps(mu: float, delta: float) -> float:
    """Smallest eps with delta(eps) <= delta for mu-GDP (Dong, Roth, Su 2022, Cor. 2.13)."""
    dl = lambda e: norm.cdf(-e / mu + mu / 2) - math.exp(e) * norm.cdf(-e / mu - mu / 2)  # noqa: E731
    lo, hi = 0.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dl(mid) > delta else (lo, mid)
    return hi


def full_run_eps(sigma: float) -> float:
    from opacus.accountants import PRVAccountant
    acc = PRVAccountant()
    acc.history = [(sigma, BATCH_TRAIN / N_TRAIN, EPOCHS_TRAIN * math.ceil(N_TRAIN / BATCH_TRAIN))]
    return float(acc.get_epsilon(delta=DELTA_TRAIN))


def whitebox_step_audit(diff: torch.Tensor, C: float, sigma: float, n: int = 1_000_000, chunk: int = 4000) -> Dict[str, Any]:
    D = diff.to(DEV)
    delta_norm = float(D.norm())
    u = D / delta_norm
    tau = delta_norm / 2
    fp = fn = 0
    for _ in range(n // chunk):  # real full-dimensional Gaussian noise, as the mechanism draws it
        fp += int(((torch.randn(chunk, D.numel(), device=DEV) * sigma * C) @ u > tau).sum())
        fn += int((delta_norm + (torch.randn(chunk, D.numel(), device=DEV) * sigma * C) @ u <= tau).sum())
    n = (n // chunk) * chunk
    fpr_ub = float(beta_dist.ppf(0.975, fp + 1, n - fp))  # Clopper-Pearson, Bonferroni 2 x 2.5%
    fnr_ub = float(beta_dist.ppf(0.975, fn + 1, n - fn))
    mu_lb = float(norm.ppf(1 - fpr_ub) - norm.ppf(fnr_ub))
    mu_claim = 2.0 / sigma
    ratio_lb = mu_lb / mu_claim
    return {"n_per_hypothesis": n, "sigma": sigma, "sensitivity_over_2C": delta_norm / (2 * C),
            "fpr": fp / n, "fnr": fn / n, "mu_claimed": mu_claim, "mu_lower_bound_95": mu_lb,
            "violation_certified": bool(mu_lb > mu_claim),
            "per_step_eps_claimed_delta1e-5": gdp_eps(mu_claim, 1e-5),
            "per_step_eps_lower_bound_delta1e-5": gdp_eps(max(mu_lb, 1e-6), 1e-5),
            "full_run_eps_reported": full_run_eps(sigma),
            "full_run_eps_min_valid_certificate": full_run_eps(sigma / max(ratio_lb, 1.0))}


# ---------------------------------------------------------------- Part C
def eps_lower_bound_one_run(r: int, correct: int, beta: float = 0.05) -> float:
    """Largest eps rejected at level beta: P[Binom(r, e^eps/(1+e^eps)) >= correct] <= beta."""
    if r == 0 or binom.sf(correct - 1, r, 0.5) > beta:
        return 0.0
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = math.exp(mid) / (1 + math.exp(mid))
        lo, hi = (mid, hi) if binom.sf(correct - 1, r, p) <= beta else (lo, mid)
    return lo


def make_canaries(schema: Dict[str, Any], columns: List[str], m: int, rng) -> pd.DataFrame:
    rows = {}
    for col in columns:
        spec = schema[col]
        if spec.get("type") == "categorical":
            rows[col] = rng.choice(spec["categories"], size=m)
        else:
            lo, hi = spec["bounds"]
            vals = rng.uniform(lo, hi, size=m)
            rows[col] = np.round(vals) if spec.get("kind") in ("integer", "count", "binary", "binary_int") else vals
    return pd.DataFrame(rows)[columns]


def canary_audit(df: pd.DataFrame, schema: Dict[str, Any], m: int = 1000, k_primary: int = 100) -> Dict[str, Any]:
    tr, _ = train_test_split(df, test_size=0.2, random_state=42)
    tr = tr.reset_index(drop=True)
    configs = [("i_cfm", s, dict(use_ot_coupling=False), 1.0, 5) for s in range(3)]
    configs += [("batch_ot", s, dict(use_ot_coupling=True, allow_nonprivate_ot=True), 1.0, 5) for s in range(3)]
    configs += [("near_nonprivate_i_cfm", 0, dict(use_ot_coupling=False), 50.0, 30)]  # audit-power check
    out = []
    for mode, seed, kw, eps_max, epochs in configs:
        rng = np.random.default_rng(1000 + seed)
        can = make_canaries(schema, list(tr.columns), m, rng)
        member = rng.random(m) < 0.5
        train = pd.concat([tr, can[member]], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        pipe = ACFMNetV5Pipeline(target_epsilon=eps_max, target_delta=1e-5, latent_mode="identity",
                                 public_schema=schema, enable_dp_recalibration=False, **kw)
        syn, rep = pipe.fit_generate(df_raw=train, num_samples=len(tr), target_col="class", epochs=epochs,
                                     protocol_id=f"CANARY-{mode}-{seed}")
        Xall, _ = encode(pd.concat([can, syn[can.columns]], ignore_index=True), schema)
        dist = torch.cdist(Xall[:m].to(DEV), Xall[m:].to(DEV)).min(dim=1).values.cpu().numpy()
        order = np.argsort(dist)  # closest to a synthetic record = most likely IN
        row = {"mode": mode, "seed": seed, "epsilon_reported": rep["privacy_summary"]["prv_upper_with_laplace"],
               "time_s": round(time.time() - t0, 1), "by_k": {}}
        for k in (25, 50, 100, 200, 500):
            guess_in, guess_out = order[:k], order[-k:]
            correct = int(member[guess_in].sum() + (~member[guess_out]).sum())
            row["by_k"][k] = {"guesses": 2 * k, "correct": correct,
                              "eps_lb_95": eps_lower_bound_one_run(2 * k, correct)}
        row["eps_lb_95_primary"] = row["by_k"][k_primary]["eps_lb_95"]
        print(f"  [C] {mode} seed={seed} eps={row['epsilon_reported']:.3f} "
              f"correct@k={k_primary}: {row['by_k'][k_primary]['correct']}/{2*k_primary} "
              f"eps_lb={row['eps_lb_95_primary']:.3f}", flush=True)
        out.append(row)
    return {"m_canaries": m, "k_primary_fixed_a_priori": k_primary, "score": "-L2 distance to nearest synthetic record (encoded space)",
            "bound": "Steinke et al. 2023, pure-DP form, 95% confidence", "runs": out}


def main():
    global N_TRAIN, EPS_TRAIN
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
    dataset = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else "adult"
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    if dataset == "cardio":  # second dataset for Part A/B only; config of runs/dapc_split (eps* 0.849)
        import dapc_split_eval
        df, schema, _, _ = dapc_split_eval.load("cardio")
        N_TRAIN, EPS_TRAIN = 16000, 0.849
    else:
        df, schema = v.load_adult(v.N_ROWS)
    key = "stronger_adversary" + ("" if dataset == "adult" else f"_{dataset}")
    X, blocks = encode(df, schema)
    print(f"Adult encoded: N={len(X)}, d={X.shape[1]}, blocks={len(blocks)}", flush=True)

    if part in ("A", "all"):
        t0 = time.time()
        adv, worst = stronger_adversary(X, blocks)
        adv["time_s"] = round(time.time() - t0, 1)
        sigma = training_sigma()
        adv["whitebox_step_audit"] = {mode: whitebox_step_audit(diff, C, sigma) for mode, (_, diff, C) in worst.items()}
        results[key] = adv
        print(json.dumps(adv, indent=2), flush=True)
        OUT.write_text(json.dumps(results, indent=2))

    if part in ("C", "all") and dataset == "adult":
        results["canary_audit"] = canary_audit(df, schema)
        OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
