"""
Rigorous verification of the paper's core claim: batch-Hungarian OT coupling breaks the
per-record sensitivity bound DP-SGD's accountant assumes, and I-CFM restores it — on real
data (UCI Adult, encoded with the declared public schema, d=78), with proper statistics.

Part 1 — the quantity DP-SGD's guarantee actually depends on: the L2 sensitivity of the SUM
of clipped per-sample gradients between replace-one neighboring batches B, B'. The Gaussian
mechanism is calibrated to Delta = 2C. We measure ||S(B) - S(B')||_2 directly, sharing the
prior draws z0 and times t between B and B' so only the coupling can differ.

Part 2 — utility cost of the fix: full pipeline, I-CFM vs batch-OT at the same target epsilon,
10 seeds each, paired Wilcoxon signed-rank tests.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "backend"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.generative.ot_cfm import ContinuousOTCFM
from acfm_net.governance import ClinicalDataGovernanceLayer
from acfm_net.preparation import AdaptiveImputer, DistributionPreservingEncoder, AdaptiveFeatureScaler
from eval_utils import evaluate_tstr, compute_anti_memorization_dcr, correlation_frobenius_error

N_ROWS = 15000
CLIP_C = 1.25
SENS_SEEDS = 10
TRIALS_PER_SEED = 10
BATCH = 64
UTIL_SEEDS = 10


def load_adult(n_rows: int):
    df = pd.read_csv(REPO_ROOT / "lumisyn" / "data" / "kaggle_datasets" / "adult.csv")
    df = df.sample(n=min(n_rows, len(df)), random_state=42).reset_index(drop=True)
    with open(REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml") as f:
        schema = yaml.safe_load(f)["columns"]
    return df, schema


def encode_adult(df: pd.DataFrame, schema: Dict[str, Any]) -> torch.Tensor:
    """Same Stage 0-2 preprocessing the pipeline uses, public schema only."""
    _, _, df_deid = ClinicalDataGovernanceLayer().process_governance(df, protocol_id="SENS", run_deidentification=True)
    df_imp, _ = AdaptiveImputer().fit_transform(df_deid, {}, privacy_mode=True, public_schema=schema)
    enc_X, enc_info = DistributionPreservingEncoder().fit_transform(df_imp, privacy_schema=schema)
    scaled = AdaptiveFeatureScaler().fit_transform(
        enc_X, enc_info["feature_specs"], privacy_mode=True, encoded_column_names=enc_info["encoded_column_names"]
    )
    return torch.tensor(scaled, dtype=torch.float32)


def clipped_grad_sum(model, z0, z1c, t, params, C):
    """Sum over the batch of per-sample gradients clipped to norm C (exact DP-SGD step, pre-noise)."""
    total = [torch.zeros_like(p) for p in params]
    for i in range(z0.shape[0]):
        zt = (1.0 - t[i:i+1]) * z0[i:i+1] + t[i:i+1] * z1c[i:i+1]
        loss = torch.nn.functional.mse_loss(model.velocity_net(zt, t[i:i+1]), z1c[i:i+1] - z0[i:i+1], reduction="none").mean()
        grads = torch.autograd.grad(loss, params)
        norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
        coef = min(1.0, C / max(float(norm), 1e-12))
        for acc, g in zip(total, grads):
            acc.add_(g * coef)
    return total


def sensitivity_experiment(X: torch.Tensor) -> Dict[str, Any]:
    d = X.shape[1]
    rows = {"batch_ot": [], "i_cfm": []}
    reassigned = []
    for seed in range(SENS_SEEDS):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ContinuousOTCFM(continuous_dim=d, cond_dim=0, hidden_dim=128)
        params = [p for p in model.velocity_net.parameters() if p.requires_grad]
        for _ in range(TRIALS_PER_SEED):
            idx = rng.choice(len(X), size=BATCH + 1, replace=False)
            z1 = X[idx[:BATCH]].clone()
            z1_prime = z1.clone()
            j = int(rng.integers(BATCH))
            z1_prime[j] = X[idx[BATCH]]  # replace-one neighbor: a different real record
            z0 = torch.randn(BATCH, d)
            t = torch.rand(BATCH, 1)

            # Batch-Hungarian OT
            _, z1c = model.solve_minibatch_ot(z0, z1)
            _, z1c_p = model.solve_minibatch_ot(z0, z1_prime)
            others = [i for i in range(BATCH) if i != j]
            reassigned.append(int(sum(not torch.equal(z1c[i], z1c_p[i]) for i in others)))
            s, s_p = clipped_grad_sum(model, z0, z1c, t, params, CLIP_C), clipped_grad_sum(model, z0, z1c_p, t, params, CLIP_C)
            rows["batch_ot"].append(float(torch.sqrt(sum((a - b).pow(2).sum() for a, b in zip(s, s_p)))))

            # I-CFM: identity pairing
            s, s_p = clipped_grad_sum(model, z0, z1, t, params, CLIP_C), clipped_grad_sum(model, z0, z1_prime, t, params, CLIP_C)
            rows["i_cfm"].append(float(torch.sqrt(sum((a - b).pow(2).sum() for a, b in zip(s, s_p)))))

    bound = 2.0 * CLIP_C
    out = {"clip_C": CLIP_C, "replace_one_bound_2C": bound, "batch_size": BATCH, "dim": d,
           "n_trials": len(rows["i_cfm"]), "ot_reassigned_other_records_mean": float(np.mean(reassigned)),
           "ot_reassigned_other_records_max": int(np.max(reassigned))}
    for mode, vals in rows.items():
        v = np.array(vals)
        out[mode] = {
            "sensitivity_mean": float(v.mean()), "sensitivity_std": float(v.std()),
            "sensitivity_max": float(v.max()), "ratio_to_bound_max": float(v.max() / bound),
            "fraction_exceeding_bound": float(np.mean(v > bound + 1e-6)),
        }
    return out


def _clipped_grad(model, params, z0_i, z1_i, t_i, C):
    zt = (1.0 - t_i) * z0_i + t_i * z1_i
    loss = torch.nn.functional.mse_loss(model.velocity_net(zt, t_i), z1_i - z0_i, reduction="none").mean()
    g = torch.cat([x.flatten() for x in torch.autograd.grad(loss, params)])
    n = float(g.norm())
    return g * min(1.0, C / max(n, 1e-12)), n


def worst_case_sensitivity(X: torch.Tensor, candidates: int = 200, top_k: int = 5) -> Dict[str, Any]:
    """
    DP is a worst-case guarantee, so search for an adversarial replace-one neighbor.
    Regime: C set so every per-sample gradient is clipped (matches the 100% clip fraction
    in every training log). Hungarian is cheap, so pre-screen `candidates` replacement
    records by how many OTHER records they reassign, then compute exact clipped-sum
    sensitivity only for the top_k. Only records whose pairing changed are recomputed.
    """
    d = X.shape[1]
    res = {"batch_ot": [], "i_cfm": []}
    reassigned = []
    for seed in range(SENS_SEEDS):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ContinuousOTCFM(continuous_dim=d, cond_dim=0, hidden_dim=128)
        params = [p for p in model.velocity_net.parameters() if p.requires_grad]
        for _ in range(TRIALS_PER_SEED):
            idx = rng.choice(len(X), size=BATCH, replace=False)
            z1 = X[idx].clone()
            z0 = torch.randn(BATCH, d)
            t = torch.rand(BATCH, 1)
            j = int(rng.integers(BATCH))

            _, z1c = model.solve_minibatch_ot(z0, z1)
            norms = [_clipped_grad(model, params, z0[i:i+1], z1c[i:i+1], t[i:i+1], 1e9)[1] for i in range(BATCH)]
            C = 0.5 * min(norms)  # all-clipped regime
            base_ot = [_clipped_grad(model, params, z0[i:i+1], z1c[i:i+1], t[i:i+1], C)[0] for i in range(BATCH)]
            base_j_icfm = _clipped_grad(model, params, z0[j:j+1], z1[j:j+1], t[j:j+1], C)[0]

            # DP neighbors may differ in ANY domain record, not just real ones: mix real records,
            # domain corners, and points placed on other records' prior draws (to force cascades).
            real = np.setdiff1d(rng.choice(len(X), size=candidates + BATCH, replace=False), idx)[:candidates]
            pool = [X[r] for r in real]
            pool += [torch.clamp(z0[k], -1.0, 1.0) for k in range(BATCH) if k != j]
            pool += [torch.sign(torch.randn(d)) for _ in range(candidates // 4)]
            screened = []
            for r, cand in enumerate(pool):
                z1p = z1.clone()
                z1p[j] = cand
                _, z1cp = model.solve_minibatch_ot(z0, z1p)
                changed = [i for i in range(BATCH) if not torch.equal(z1c[i], z1cp[i])]
                screened.append((len(changed), int(r), changed, z1cp))
            screened.sort(key=lambda s: -s[0])

            best_ot, best_icfm = 0.0, 0.0
            for n_changed, r, changed, z1cp in screened[:top_k]:
                diff = torch.zeros_like(base_ot[0])
                for i in changed:
                    diff += _clipped_grad(model, params, z0[i:i+1], z1cp[i:i+1], t[i:i+1], C)[0] - base_ot[i]
                best_ot = max(best_ot, float(diff.norm()) / (2 * C))
                g_new = _clipped_grad(model, params, z0[j:j+1], pool[r].unsqueeze(0), t[j:j+1], C)[0]
                best_icfm = max(best_icfm, float((g_new - base_j_icfm).norm()) / (2 * C))
            reassigned.append(screened[0][0] - 1)
            res["batch_ot"].append(best_ot)
            res["i_cfm"].append(best_icfm)

    out = {"regime": "all per-sample gradients clipped (C = 0.5 * min grad norm)",
           "adversary": f"best of top-{top_k} by reassignment count over {candidates} real + {BATCH-1} prior-anchored + {candidates // 4} domain-corner candidates per trial",
           "n_trials": len(res["i_cfm"]),
           "ot_reassigned_other_records_mean": float(np.mean(reassigned)),
           "ot_reassigned_other_records_max": int(np.max(reassigned))}
    for mode, vals in res.items():
        v = np.array(vals)
        out[mode] = {"sensitivity_over_2C_mean": float(v.mean()), "sensitivity_over_2C_std": float(v.std()),
                     "sensitivity_over_2C_max": float(v.max()), "fraction_exceeding_bound": float(np.mean(v > 1.0 + 1e-6))}
    return out


def utility_experiment(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    results = {"i_cfm": [], "batch_ot": []}
    for seed in range(UTIL_SEEDS):
        for mode, kwargs in [("i_cfm", dict(use_ot_coupling=False)),
                             ("batch_ot", dict(use_ot_coupling=True, allow_nonprivate_ot=True))]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            pipe = ACFMNetV5Pipeline(target_epsilon=1.0, target_delta=1e-5, latent_mode="identity",
                                     public_schema=schema, enable_dp_recalibration=False, **kwargs)
            df_syn, report = pipe.fit_generate(df_raw=df_train, num_samples=len(df_train), target_col="class",
                                               epochs=5, protocol_id=f"ICFM-{mode}-{seed}")
            util = evaluate_tstr(df_train, df_test, df_syn, "class", "classification")
            row = {"seed": seed, "time_s": time.time() - t0,
                   "corr_err": correlation_frobenius_error(df_train, df_syn),
                   "mean_dcr": compute_anti_memorization_dcr(df_train, df_syn)[0],
                   "tstr_auc": util["tstr_auc"], "tstr_acc": util["tstr_acc"], "trtr_auc": util["trtr_auc"],
                   "reported_epsilon": report["privacy_summary"]["prv_upper_with_laplace"]}
            results[mode].append(row)
            print(f"  [{mode} seed={seed}] {row['time_s']:.1f}s AUC={row['tstr_auc']:.3f} corr={row['corr_err']:.4f} eps={row['reported_epsilon']:.4f}", flush=True)
    return results


def paired_stats(util: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out = {}
    for metric in ("tstr_auc", "corr_err", "mean_dcr"):
        a = np.array([r[metric] for r in util["i_cfm"]])
        b = np.array([r[metric] for r in util["batch_ot"]])
        p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
        out[metric] = {"i_cfm_mean": float(a.mean()), "i_cfm_std": float(a.std(ddof=1)),
                       "batch_ot_mean": float(b.mean()), "batch_ot_std": float(b.std(ddof=1)),
                       "wilcoxon_p": p}
    return out


def main():
    df, schema = load_adult(N_ROWS)
    X = encode_adult(df, schema)
    print(f"Adult encoded: N={len(X)}, d={X.shape[1]}", flush=True)

    if "--worst-case-only" in sys.argv:
        wc = worst_case_sensitivity(X)
        print(json.dumps(wc, indent=2), flush=True)
        out_path = REPO_ROOT / "runs" / "icfm_worst_case_sensitivity.json"
        out_path.write_text(json.dumps(wc, indent=2))
        print(f"Saved to {out_path}")
        return

    out_path = REPO_ROOT / "runs" / "icfm_sensitivity_verification.json"
    if "--utility-only" in sys.argv:  # Part 1 does not depend on the pipeline; reuse it
        sens = json.loads(out_path.read_text())["sensitivity"]
    else:
        print(f"\n--- Part 1: clipped-sum sensitivity, {SENS_SEEDS} seeds x {TRIALS_PER_SEED} trials ---", flush=True)
        sens = sensitivity_experiment(X)
        print(json.dumps(sens, indent=2), flush=True)

    print(f"\n--- Part 2: utility, {UTIL_SEEDS} seeds x 2 couplings ---", flush=True)
    util = utility_experiment(df, schema)
    stats = paired_stats(util)
    print(json.dumps(stats, indent=2), flush=True)

    out_path.write_text(json.dumps({"sensitivity": sens, "utility": util, "stats": stats}, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
