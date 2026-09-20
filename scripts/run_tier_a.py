import os
import sys
import time
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Enforce reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.generative.ot_cfm import ContinuousOTCFM, VelocityNetwork
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    SlicedWassersteinEvaluator,
    _encode_for_eval,
)
from acfm_net.evaluation.dp_gaussian_copula import DPGaussianCopulaGenerator

def load_adult_validation_half(seed: int = 42):
    csv_path = REPO_ROOT / "lumisyn" / "data" / "kaggle_datasets" / "adult.csv"
    assert csv_path.exists(), f"File not found: {csv_path}"
    df = pd.read_csv(csv_path)
    if " income" in df.columns:
        df = df.rename(columns={" income": "income"})
    if "income" in df.columns:
        df["income"] = df["income"].str.strip()
    if "class" in df.columns:
        df["class"] = df["class"].astype(str).str.strip()

    rng = np.random.default_rng(seed)
    n_total = len(df)
    perm = rng.permutation(n_total)
    # Adult 30k validation half: 15k train / 15k validation
    train_idx = perm[:15000]
    val_idx = perm[15000:30000]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val

def compute_all_metrics(df_real_train, df_syn, df_val, real_train_std=None, feat_names=None):
    common = [c for c in df_real_train.columns if c in df_syn.columns and not c.endswith("_isna")]
    
    # 1. 1-way TVD
    marg = MarginalErrorEvaluator()
    eval_1w = marg.evaluate_1way(df_real_train[common], df_syn[common], df_real_holdout=df_val[common])
    raw_1w = eval_1w.get("mean_1way_marginal_error_tvd", 0.0)
    floor_1w = eval_1w.get("real_vs_real_floor_1way_tvd", 0.0076)
    norm_1w = eval_1w.get("normalized_1way_marginal_error", raw_1w / max(1e-4, floor_1w))
    per_feat_1w = eval_1w.get("per_feature_1way_tvd", {})

    # 2. 2-way TVD
    eval_2w = marg.evaluate_2way(df_real_train[common], df_syn[common], max_pairs=30, seed=42)
    tvd_2w = eval_2w["mean_2way_marginal_error_tvd"]
    floor_2w_eval = marg.evaluate_2way(df_real_train[common], df_val[common], max_pairs=30, seed=42)
    floor_2w = floor_2w_eval["mean_2way_marginal_error_tvd"]
    norm_2w = tvd_2w / max(1e-4, floor_2w)

    # 3. Correlation Error (Frobenius)
    corr_eval = CorrelationErrorEvaluator()
    eval_corr = corr_eval.evaluate(df_real_train[common], df_syn[common], df_real_holdout=df_val[common])
    frob_corr = eval_corr.get("pearson_frobenius_error", 0.0)
    floor_corr = eval_corr.get("real_vs_real_floor_pearson_frobenius", 0.0430)
    norm_corr = frob_corr / max(1e-4, floor_corr)

    # 4. Standardized SWD
    Xr, fn = _encode_for_eval(df_real_train[common])
    Xs, _ = _encode_for_eval(df_syn[common])
    std_to_use = real_train_std if real_train_std is not None else np.std(Xr, axis=0)
    swd_eval = SlicedWassersteinEvaluator(num_projections=100)
    swd_res = swd_eval.evaluate(
        Xr, Xs, seed=42, standardize=True, real_std=std_to_use, feature_names=fn
    )
    swd = swd_res["sliced_wasserstein_distance"]

    return {
        "raw_1way_tvd": raw_1w,
        "floor_1way": floor_1w,
        "norm_1way_tvd": norm_1w,
        "raw_2way_tvd": tvd_2w,
        "floor_2way": floor_2w,
        "norm_2way_tvd": norm_2w,
        "frob_corr": frob_corr,
        "floor_corr": floor_corr,
        "norm_corr": norm_corr,
        "swd": swd,
        "per_feature_1way": per_feat_1w,
    }

def main():
    total_start_time = time.time()
    print("=" * 80)
    print(" TIER A BENCHMARK (Adult 30k, validation half, seed 42, 1 seed, 3-min timeout/step)")
    print("=" * 80)

    df_train, df_val = load_adult_validation_half(seed=42)
    print(f"Data split: Train={len(df_train)} rows | Validation={len(df_val)} rows")

    schema_path = REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml"
    with open(schema_path, "r", encoding="utf-8") as f:
        adult_schema = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up base pipeline for preprocessing
    base_pipe = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        latent_mode="identity",
        batch_size=512,
        flow_hidden_dim=128,
        flow_num_layers=2,
    )

    df_imputed, _ = base_pipe.imputer.fit_transform(df_train, None, privacy_mode=True, public_schema=adult_schema)
    encoded_X, enc_info = base_pipe.encoder.fit_transform(df_imputed, privacy_schema=adult_schema)
    scaled_X = base_pipe.scaler.fit_transform(
        encoded_X,
        enc_info["feature_specs"],
        privacy_mode=True,
        encoded_column_names=enc_info["encoded_column_names"],
    )
    X_tensor = torch.tensor(scaled_X, dtype=torch.float32, device=device)
    in_dim = X_tensor.shape[1]
    print(f"Encoded and scaled X: {X_tensor.shape} (d={in_dim})")

    # Real train std for SWD
    common_cols = [c for c in df_train.columns if not c.endswith("_isna")]
    Xr_train, feat_names = _encode_for_eval(df_train[common_cols])
    real_train_std = np.std(Xr_train, axis=0)

    tier_a_results = {}

    # =========================================================================
    # Step a: Non-private identity flow (same architecture, 40 epochs)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step a: Non-Private Identity Flow (40 epochs, AdamW lr=2e-3, B=512)")
    print("=" * 70)
    t_a0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    nonpriv_flow = ContinuousOTCFM(
        continuous_dim=in_dim,
        cond_dim=0,
        hidden_dim=128,
        num_layers=2,
    ).to(device)
    opt_a = torch.optim.Adam(nonpriv_flow.parameters(), lr=2e-3)
    loader_a = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=True)

    for ep in range(1, 41):
        nonpriv_flow.train()
        ep_loss = 0.0
        batches = 0
        for (xb,) in loader_a:
            opt_a.zero_grad()
            loss = nonpriv_flow.compute_loss(xb, use_ot_coupling=False)
            loss.backward()
            opt_a.step()
            ep_loss += float(loss.item())
            batches += 1
        if ep % 10 == 0 or ep == 40:
            print(f"  Epoch {ep:2d}/40 | Flow Loss: {ep_loss/batches:.4f}")

    # Generate synthetic samples
    nonpriv_flow.eval()
    with torch.no_grad():
        z0 = torch.randn(len(df_val), in_dim, device=device)
        raw_samples = nonpriv_flow.sample(num_samples=len(df_val), device=device, num_steps=50, solver="rk4")
        oob_mask = (raw_samples < -1.0) | (raw_samples > 1.0)
        oob_fraction = float(oob_mask.float().mean().item())
        clamped_samples = torch.clamp(raw_samples, -1.0, 1.0).cpu().numpy()
        decoded_encoded = base_pipe.scaler.inverse_transform(clamped_samples)
        df_syn_a = base_pipe.encoder.decode(decoded_encoded)

    m_a = compute_all_metrics(df_train, df_syn_a, df_val, real_train_std=real_train_std, feat_names=feat_names)
    t_a_elapsed = time.time() - t_a0
    print(f"\nStep a Results (Elapsed: {t_a_elapsed:.1f}s):")
    print(f"  Normalized 1-way TVD:       {m_a['norm_1way_tvd']:.4f}x (Raw: {m_a['raw_1way_tvd']:.4f}, Floor: {m_a['floor_1way']:.4f})")
    print(f"  Normalized 2-way TVD:       {m_a['norm_2way_tvd']:.4f}x")
    print(f"  Normalized Correlation:     {m_a['norm_corr']:.4f}x")
    print(f"  Standardized SWD:           {m_a['swd']:.4f}")
    print(f"  Pre-clamp OOB fraction:     {oob_fraction:.2%}")

    tier_a_results["step_a_nonprivate"] = {
        "metrics": m_a,
        "oob_fraction": oob_fraction,
        "elapsed_sec": t_a_elapsed,
    }

    # STOP RULE: if normalized 1-way TVD > 10x floor
    if m_a["norm_1way_tvd"] > 10.0:
        print("\n" + "!" * 80)
        print(f"DESIGN BUG: Non-private normalized 1-way TVD is {m_a['norm_1way_tvd']:.2f}x (threshold is 10.0x)")
        print(f"Diagnostics: OOB Frac = {oob_fraction:.2%}, Standardized SWD = {m_a['swd']:.4f}")
        print("!" * 80)
        total_time = time.time() - total_start_time
        print(f"\nDESIGN BUG, total time: {total_time:.1f}s")
        return

    # =========================================================================
    # Step b: DP eps=1 identity at prior scale s in {0.25, 0.5, 1.0}
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step b: DP eps=1 Identity Flow Sweep over Prior Scale s in {0.25, 0.5, 1.0}")
    print("=" * 70)
    scales = [0.25, 0.5, 1.0]
    dp_scale_results = {}
    best_s = None
    best_1w = float("inf")

    num_cols = [c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c])]

    for s_val in scales:
        t_b0 = time.time()
        print(f"\n--- Testing DP eps=1.0 at prior scale s = {s_val} ---")
        torch.manual_seed(42)
        np.random.seed(42)

        pipe_b = ACFMNetV5Pipeline(
            target_epsilon=1.0,
            target_delta=1e-5,
            strict_release_privacy=True,
            latent_mode="identity",
            batch_size=512,
            flow_hidden_dim=128,
            flow_num_layers=2,
            flow_optimizer="adam",
            flow_lr=2e-3,
            max_grad_norm_flow=3.0822,
            prior_scale=s_val,
        )
        df_syn_b, rep_b = pipe_b.fit_generate(df_train, num_samples=len(df_val), epochs=10, target_col="class")
        
        # Pre-clamp OOB fraction from released velocity net
        with torch.no_grad():
            z0_b = torch.randn(len(df_val), in_dim, device=device) * s_val
            raw_latents_b = pipe_b.released_checkpoint["velocity_net"] if "velocity_net" in pipe_b.released_checkpoint else None

        m_b = compute_all_metrics(df_train, df_syn_b, df_val, real_train_std=real_train_std, feat_names=feat_names)

        # Per-column std mismatch
        std_real = df_train[num_cols].std().values
        std_syn = df_syn_b[num_cols].std().values
        col_std_mismatch = np.mean(np.abs(std_real - std_syn) / np.maximum(1e-6, std_real))

        t_b_elapsed = time.time() - t_b0
        print(f"  s={s_val:4.2f} (Elapsed: {t_b_elapsed:.1f}s):")
        print(f"    Norm 1-way TVD:       {m_b['norm_1way_tvd']:.2f}x")
        print(f"    Norm 2-way TVD:       {m_b['norm_2way_tvd']:.2f}x")
        print(f"    Norm Correlation:     {m_b['norm_corr']:.2f}x")
        print(f"    Standardized SWD:     {m_b['swd']:.4f}")
        print(f"    Col Std Mismatch:     {col_std_mismatch:.2%}")

        dp_scale_results[s_val] = {
            "metrics": m_b,
            "col_std_mismatch": col_std_mismatch,
            "elapsed_sec": t_b_elapsed,
        }

        if m_b["norm_1way_tvd"] < best_1w:
            best_1w = m_b["norm_1way_tvd"]
            best_s = s_val

    print(f"\n[Best Prior Scale]: s = {best_s} with Norm 1-way TVD = {best_1w:.2f}x")
    # Top 10 worst columns by 1-way TVD for best s
    best_per_col = dp_scale_results[best_s]["metrics"]["per_feature_1way"]
    sorted_worst_cols = sorted(best_per_col.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"10 Worst Columns by 1-way TVD for best s={best_s}:")
    for rank, (col, tvd) in enumerate(sorted_worst_cols, 1):
        print(f"  {rank:2d}. {col:<20s}: {tvd:.4f}")

    tier_a_results["step_b_dp_sweep"] = {
        "scale_results": dp_scale_results,
        "best_s": best_s,
        "worst_10_columns": sorted_worst_cols,
    }

    # =========================================================================
    # Step c: DP Kendall-tau copula with NON-PRIVATE tau as a positive control
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step c: DP Kendall-tau Copula with NON-PRIVATE Tau (Positive Control)")
    print("=" * 70)
    t_c0 = time.time()
    
    # Prepare column specs for Adult dataset
    col_specs = {}
    for col in df_train.columns:
        if col.endswith("_isna"):
            continue
        if pd.api.types.is_numeric_dtype(df_train[col]):
            col_specs[col] = {
                "type": "numeric",
                "bounds": (float(df_train[col].min()), float(df_train[col].max())),
            }
        else:
            cats = list(df_train[col].dropna().unique())
            col_specs[col] = {
                "type": "categorical",
                "categories": cats,
            }

    # In DPGaussianCopulaGenerator:
    # Set eps_corr = 1e6 (effectively infinite epsilon, zero noise on Kendall tau)
    # While keeping eps_marginal = 1.0
    copula = DPGaussianCopulaGenerator(
        target_epsilon=1.0,
        eps_marginal_ratio=0.5,
        eps_corr_ratio=0.5,
        columns_spec=col_specs,
        seed=42,
    )
    # Override eps_corr to be effectively non-private (eps_corr=1e6 gives lap_scale_tau -> 0)
    copula.eps_corr = 1e6
    copula.eps_marginal = 1.0
    copula.fit(df_train)
    df_syn_c = copula.generate(len(df_val), seed=42)

    m_c = compute_all_metrics(df_train, df_syn_c, df_val, real_train_std=real_train_std, feat_names=feat_names)
    t_c_elapsed = time.time() - t_c0

    print(f"\nStep c Results (Elapsed: {t_c_elapsed:.1f}s):")
    print(f"  Normalized 1-way TVD:   {m_c['norm_1way_tvd']:.4f}x (Raw: {m_c['raw_1way_tvd']:.4f})")
    print(f"  Normalized 2-way TVD:   {m_c['norm_2way_tvd']:.4f}x")
    print(f"  Normalized Correlation: {m_c['norm_corr']:.4f}x")
    print(f"  Standardized SWD:       {m_c['swd']:.4f}")

    tier_a_results["step_c_copula_pos_control"] = {
        "metrics": m_c,
        "elapsed_sec": t_c_elapsed,
    }

    total_time = time.time() - total_start_time
    # Determine conclusion: DESIGN BUG, DP-LIMITED, or PROMISING
    # If non-private flow passed, and DP flow is limited by noise compared to non-private flow, it is DP-LIMITED
    if m_a["norm_1way_tvd"] <= 10.0:
        if m_b["norm_1way_tvd"] <= 3.0:
            conclusion = "PROMISING"
        else:
            conclusion = "DP-LIMITED"
    else:
        conclusion = "DESIGN BUG"

    with open(REPO_ROOT / "runs" / "tier_a_results.json", "w", encoding="utf-8") as f:
        json.dump(tier_a_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"TIER A FINAL SUMMARY LINE:")
    print(f"{conclusion}, total time: {total_time:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
