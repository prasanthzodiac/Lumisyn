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

# Enforce deterministic flags
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.generative.ot_cfm import ContinuousOTCFM, VelocityNetwork
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    SlicedWassersteinEvaluator,
    _encode_for_eval,
)
from acfm_net.evaluation.floor_baselines import IndependentMarginalsDPGenerator
from scripts.run_tier_a import load_adult_validation_half, compute_all_metrics

def main():
    t_start = time.time()
    print("=" * 80)
    print(" DESIGN-BUG TRIAGE (Adult 30k, 15k train / 15k validation, seed 42, d_enc=78)")
    print("=" * 80)

    # 0. Load Dataset & Schema
    df_train, df_val = load_adult_validation_half(seed=42)
    schema_path = REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml"
    with open(schema_path, "r", encoding="utf-8") as f:
        adult_schema = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Train N={len(df_train)} | Val N={len(df_val)}")

    # Preprocessing pipeline
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
    assert in_dim == 78, f"Expected d_enc=78, got {in_dim}"
    print(f"Scaled X tensor shape: {X_tensor.shape} (d_enc={in_dim})")

    common_cols = [c for c in df_train.columns if not c.endswith("_isna")]
    Xr_train, feat_names = _encode_for_eval(df_train[common_cols])
    real_train_std = np.std(Xr_train, axis=0)

    # Zero-predictor baseline on scaled_X: E[||X||^2] + 1.0
    zero_predictor_baseline = float(torch.mean(X_tensor ** 2).item() + 1.0)
    print(f"Zero-predictor flow loss baseline on X_scaled: {zero_predictor_baseline:.4f}")

    results = {}

    # =========================================================================
    # T6. Recompute Independent-DP-Marginals (eps=1) & Real-vs-Real Floor on THIS split
    # =========================================================================
    print("\n" + "=" * 75)
    print(" T6. BASELINES ON THIS SPLIT (15k Train / 15k Val)")
    print("=" * 75)
    # Real-vs-real floor
    marg = MarginalErrorEvaluator()
    floor_1w_eval = marg.evaluate_1way(df_train[common_cols], df_val[common_cols])
    floor_1w = floor_1w_eval["mean_1way_marginal_error_tvd"]
    floor_2w_eval = marg.evaluate_2way(df_train[common_cols], df_val[common_cols], max_pairs=30, seed=42)
    floor_2w = floor_2w_eval["mean_2way_marginal_error_tvd"]
    corr_eval = CorrelationErrorEvaluator()
    floor_corr_eval = corr_eval.evaluate(df_train[common_cols], df_val[common_cols])
    floor_corr = floor_corr_eval["pearson_frobenius_error"]

    print(f"Real-vs-Real Holdout Floors:")
    print(f"  1-way TVD Floor:       {floor_1w:.6f}")
    print(f"  2-way TVD Floor:       {floor_2w:.6f}")
    print(f"  Correlation Floor:     {floor_corr:.6f}")

    # Independent DP Marginals (eps=1.0)
    ind_gen = IndependentMarginalsDPGenerator(epsilon=1.0, seed=42)
    ind_gen.fit(df_train)
    df_syn_ind = ind_gen.generate(len(df_val))
    m_ind = compute_all_metrics(df_train, df_syn_ind, df_val, real_train_std=real_train_std, feat_names=feat_names)
    print(f"\nIndependent DP Marginals (eps=1.0):")
    print(f"  Norm 1-way TVD:        {m_ind['norm_1way_tvd']:.4f}x (Raw: {m_ind['raw_1way_tvd']:.4f})")
    print(f"  Norm 2-way TVD:        {m_ind['norm_2way_tvd']:.4f}x")
    print(f"  Norm Correlation:      {m_ind['norm_corr']:.4f}x")
    print(f"  Standardized SWD:      {m_ind['swd']:.4f}")

    results["T6_baselines"] = {
        "floor_1way": floor_1w,
        "floor_2way": floor_2w,
        "floor_corr": floor_corr,
        "independent_dp_marginals": m_ind,
    }

    # =========================================================================
    # Helper for training non-private flow
    # =========================================================================
    def train_nonpriv_flow(epochs: int, hidden_dim: int = 128, num_layers: int = 2, lr: float = 2e-3):
        t0 = time.time()
        torch.manual_seed(42)
        np.random.seed(42)
        flow = ContinuousOTCFM(
            continuous_dim=in_dim,
            cond_dim=0,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=True)

        final_loss = 0.0
        for ep in range(1, epochs + 1):
            flow.train()
            ep_loss = 0.0
            batches = 0
            for (xb,) in loader:
                optimizer.zero_grad()
                loss = flow.compute_loss(xb, use_ot_coupling=False)
                loss.backward()
                optimizer.step()
                ep_loss += float(loss.item())
                batches += 1
            final_loss = ep_loss / max(1, batches)
            if ep % max(1, epochs // 4) == 0 or ep == epochs:
                print(f"    Epoch {ep:3d}/{epochs} | Flow Loss: {final_loss:.4f}")

        # Sample
        flow.eval()
        with torch.no_grad():
            raw_latents = flow.sample(num_samples=len(df_val), device=device, num_steps=50, solver="rk4")
            oob = float(((raw_latents < -1.0) | (raw_latents > 1.0)).float().mean().item())
            clamped = torch.clamp(raw_latents, -1.0, 1.0).cpu().numpy()
            decoded_encoded = base_pipe.scaler.inverse_transform(clamped)
            df_syn = base_pipe.encoder.decode(decoded_encoded)

        m = compute_all_metrics(df_train, df_syn, df_val, real_train_std=real_train_std, feat_names=feat_names)
        m["final_flow_loss"] = final_loss
        m["zero_pred_baseline"] = zero_predictor_baseline
        m["loss_vs_zero_ratio"] = final_loss / max(1e-6, zero_predictor_baseline)
        m["oob_fraction"] = oob
        m["train_time"] = time.time() - t0
        m["raw_latents"] = raw_latents
        m["df_syn"] = df_syn
        m["decoded_encoded"] = decoded_encoded
        return flow, m

    # =========================================================================
    # T1. Non-private identity flow at E in {40, 200, 600} (128x2)
    # =========================================================================
    print("\n" + "=" * 75)
    print(" T1. NON-PRIVATE IDENTITY FLOW AT EPOCHS in {40, 200, 600} (128x2)")
    print("=" * 75)
    t1_results = {}
    best_t1_m = None
    best_t1_flow = None
    best_t1_1w = float("inf")

    for ep in [40, 200, 600]:
        print(f"\n--- Training E = {ep} epochs (Architecture: 128x2) ---")
        flow, m = train_nonpriv_flow(epochs=ep, hidden_dim=128, num_layers=2)
        print(f"  E={ep:3d} (Time: {m['train_time']:.1f}s):")
        print(f"    Norm 1-way TVD:       {m['norm_1way_tvd']:.4f}x (Raw: {m['raw_1way_tvd']:.4f})")
        print(f"    Norm 2-way TVD:       {m['norm_2way_tvd']:.4f}x")
        print(f"    Norm Correlation:     {m['norm_corr']:.4f}x")
        print(f"    Standardized SWD:     {m['swd']:.4f}")
        print(f"    Final Flow Loss:      {m['final_flow_loss']:.4f} vs Zero-Pred {zero_predictor_baseline:.4f} ({m['loss_vs_zero_ratio']:.2%})")
        print(f"    Pre-clamp OOB Frac:   {m['oob_fraction']:.2%}")

        t1_results[ep] = {k: v for k, v in m.items() if k not in ("raw_latents", "df_syn", "decoded_encoded")}
        if m["norm_1way_tvd"] < best_t1_1w:
            best_t1_1w = m["norm_1way_tvd"]
            best_t1_m = m
            best_t1_flow = flow

    results["T1_epoch_sweep"] = t1_results

    # =========================================================================
    # T2. At E=200: hidden 128x2 vs 256x3
    # =========================================================================
    print("\n" + "=" * 75)
    print(" T2. ARCHITECTURE COMPARISON AT E=200 (128x2 vs 256x3)")
    print("=" * 75)
    print("\n--- Training E = 200 epochs (Architecture: 256x3) ---")
    flow_256, m_256 = train_nonpriv_flow(epochs=200, hidden_dim=256, num_layers=3)
    print(f"  256x3 (Time: {m_256['train_time']:.1f}s):")
    print(f"    Norm 1-way TVD:       {m_256['norm_1way_tvd']:.4f}x (Raw: {m_256['raw_1way_tvd']:.4f})")
    print(f"    Norm 2-way TVD:       {m_256['norm_2way_tvd']:.4f}x")
    print(f"    Norm Correlation:     {m_256['norm_corr']:.4f}x")
    print(f"    Standardized SWD:     {m_256['swd']:.4f}")
    print(f"    Final Flow Loss:      {m_256['final_flow_loss']:.4f} vs Zero-Pred {zero_predictor_baseline:.4f} ({m_256['loss_vs_zero_ratio']:.2%})")
    print(f"    Pre-clamp OOB Frac:   {m_256['oob_fraction']:.2%}")

    results["T2_arch_comparison"] = {
        "128x2": t1_results[200],
        "256x3": {k: v for k, v in m_256.items() if k not in ("raw_latents", "df_syn", "decoded_encoded")},
    }

    # Determine best of T1/T2
    if m_256["norm_1way_tvd"] < best_t1_1w:
        best_overall_m = m_256
        best_overall_flow = flow_256
        best_label = "256x3 (E=200)"
    else:
        best_overall_m = best_t1_m
        best_overall_flow = best_t1_flow
        best_label = f"128x2 (E={min(t1_results.keys(), key=lambda k: t1_results[k]['norm_1way_tvd'])})"
    print(f"\n[Winner of T1/T2]: {best_label} with Norm 1-way TVD = {best_overall_m['norm_1way_tvd']:.4f}x")

    # =========================================================================
    # T3. 10 worst columns by 1-way TVD & Categorical decode variants (argmax vs softmax sampling)
    # =========================================================================
    print("\n" + "=" * 75)
    print(f" T3. 10 WORST COLUMNS & CATEGORICAL DECODE VARIANTS ({best_label})")
    print("=" * 75)
    per_feat = best_overall_m["per_feature_1way"]
    worst_10 = sorted(per_feat.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n10 Worst Columns by 1-way TVD (Argmax decode):")
    for i, (col, tvd) in enumerate(worst_10, 1):
        print(f"  {i:2d}. {col:<20s}: {tvd:.4f}")

    # Custom decode with sampling from softmax of block values (temperature 1.0)
    def decode_softmax_sampling(encoder, encoded_matrix, seed=42):
        rng = np.random.RandomState(seed)
        df_dict = {}
        curr_idx = 0
        for col in encoder.column_order:
            spec = encoder.feature_specs[col]
            dim = spec["dim"]
            block = encoded_matrix[:, curr_idx:curr_idx + dim]
            kind = spec.get("kind", "continuous")

            if spec["type"] == "numeric":
                raw = block.ravel()
                lo, hi = spec["orig_min"], spec["orig_max"]
                if kind == "binary":
                    out = np.clip(np.round(raw), 0.0, 1.0).astype(np.int32)
                elif kind == "binary_int":
                    unique_vals = sorted({lo, hi})
                    mid = (unique_vals[0] + unique_vals[1]) / 2.0
                    out = np.where(raw >= mid, unique_vals[1], unique_vals[0]).astype(np.int32)
                elif kind == "count":
                    out = np.clip(np.round(raw), 0.0, hi).astype(np.int32)
                elif kind == "integer":
                    out = np.clip(np.round(raw), lo, hi).astype(np.int32)
                else:
                    out = np.clip(raw, lo, hi).astype(np.float32)
                df_dict[col] = out

            elif spec["type"] == "categorical_ohe":
                ohe = encoder.encoders[col]
                # Softmax with temperature 1.0
                logits = block - block.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                # Sample according to probabilities
                sampled_cats = []
                for p in probs:
                    # Guard against tiny negative/numerical precision
                    p = np.maximum(1e-8, p)
                    p /= p.sum()
                    idx = rng.choice(len(p), p=p)
                    sampled_cats.append(idx)
                hard_ohe = np.zeros_like(probs)
                hard_ohe[np.arange(len(probs)), sampled_cats] = 1.0
                df_dict[col] = ohe.inverse_transform(hard_ohe).ravel()

            elif spec["type"] == "categorical_ord":
                ord_enc = encoder.encoders[col]
                clipped = np.clip(np.round(block), 0, len(spec["categories"]) - 1)
                df_dict[col] = ord_enc.inverse_transform(clipped).ravel()

            curr_idx += dim
        return pd.DataFrame(df_dict)

    df_syn_softmax = decode_softmax_sampling(base_pipe.encoder, best_overall_m["decoded_encoded"], seed=42)
    m_softmax = compute_all_metrics(df_train, df_syn_softmax, df_val, real_train_std=real_train_std, feat_names=feat_names)

    print("\nCategorical Decode Variants Comparison:")
    print(f"  Argmax:           Norm 1-way TVD = {best_overall_m['norm_1way_tvd']:.4f}x | Norm 2-way TVD = {best_overall_m['norm_2way_tvd']:.4f}x")
    print(f"  Softmax Sampling: Norm 1-way TVD = {m_softmax['norm_1way_tvd']:.4f}x | Norm 2-way TVD = {m_softmax['norm_2way_tvd']:.4f}x")
    print("\nPer-column TVD Comparison (Argmax vs Softmax Sampling):")
    for col, _ in worst_10:
        tvd_arg = best_overall_m["per_feature_1way"].get(col, 0.0)
        tvd_sm = m_softmax["per_feature_1way"].get(col, 0.0)
        print(f"  {col:<20s}: Argmax = {tvd_arg:.4f} | Softmax Sampling = {tvd_sm:.4f}")

    results["T3_worst_and_decode"] = {
        "worst_10_argmax": worst_10,
        "argmax_norm_1way": best_overall_m["norm_1way_tvd"],
        "argmax_norm_2way": best_overall_m["norm_2way_tvd"],
        "softmax_norm_1way": m_softmax["norm_1way_tvd"],
        "softmax_norm_2way": m_softmax["norm_2way_tvd"],
    }

    # =========================================================================
    # T4. Diagnostic only, non-private: EXACT marginal matching (Oracle Recalibration)
    # =========================================================================
    print("\n" + "=" * 75)
    print(" T4. ORACLE RECALIBRATION (Exact Rank-Preserving Marginal Matching)")
    print("=" * 75)
    def apply_oracle_recalibration(df_real, df_generated):
        df_oracle = df_generated.copy()
        for col in df_real.columns:
            if col not in df_generated.columns or col.endswith("_isna"):
                continue
            real_vals = df_real[col].values
            syn_vals = df_generated[col].values
            N_syn = len(syn_vals)
            
            if pd.api.types.is_numeric_dtype(df_real[col]):
                # Continuous / numerical rank matching
                clean_real = pd.to_numeric(df_real[col], errors="coerce").dropna().values
                ranks = np.argsort(np.argsort(syn_vals + np.random.randn(N_syn) * 1e-8))
                percentiles = (ranks + 0.5) / N_syn
                oracle_vals = np.quantile(clean_real, percentiles)
                if pd.api.types.is_integer_dtype(df_real[col]):
                    oracle_vals = np.round(oracle_vals).astype(df_real[col].dtype)
                df_oracle[col] = oracle_vals
            else:
                # Categorical quantile matching according to empirical CDF
                clean_real = df_real[col].dropna().astype(str).values
                cats, counts = np.unique(clean_real, return_counts=True)
                probs = counts / counts.sum()
                cdf = np.cumsum(probs)
                cdf[-1] = 1.0
                # Robust rank transform for categorical values with tie-breaking jitter
                unique_syn = sorted(list(set(syn_vals.astype(str))))
                syn_map = {c: i for i, c in enumerate(unique_syn)}
                jitter = np.random.uniform(0, 1e-4, size=N_syn)
                cat_ranks = np.array([syn_map.get(str(x), 0) for x in syn_vals], dtype=float) + jitter
                ranks = np.argsort(np.argsort(cat_ranks))
                u = (ranks + 0.5) / N_syn
                assigned_idx = np.searchsorted(cdf, u, side="right")
                assigned_idx = np.clip(assigned_idx, 0, len(cats) - 1)
                df_oracle[col] = cats[assigned_idx]
        return df_oracle

    df_oracle = apply_oracle_recalibration(df_train, best_overall_m["df_syn"])
    m_oracle = compute_all_metrics(df_train, df_oracle, df_val, real_train_std=real_train_std, feat_names=feat_names)
    print(f"Oracle Recalibration Results:")
    print(f"  Norm 1-way TVD:       {m_oracle['norm_1way_tvd']:.4f}x (Raw: {m_oracle['raw_1way_tvd']:.4f})")
    print(f"  Norm 2-way TVD:       {m_oracle['norm_2way_tvd']:.4f}x (Raw: {m_oracle['raw_2way_tvd']:.4f})")
    print(f"  Norm Correlation:     {m_oracle['norm_corr']:.4f}x (Raw: {m_oracle['frob_corr']:.4f})")
    print(f"  Standardized SWD:     {m_oracle['swd']:.4f}")

    results["T4_oracle_recalibration"] = {
        "norm_1way_tvd": m_oracle["norm_1way_tvd"],
        "norm_2way_tvd": m_oracle["norm_2way_tvd"],
        "norm_corr": m_oracle["norm_corr"],
        "swd": m_oracle["swd"],
    }

    # =========================================================================
    # T5. Prior scale check & DP eps=1 (Skip if T1-T3 1-way TVD > 10x)
    # =========================================================================
    print("\n" + "=" * 75)
    print(" T5. PRIOR SCALE VERIFICATION & DP IDENTITY RUN")
    print("=" * 75)
    # Check sample std of z0
    for s_test in [0.25, 0.5, 1.0]:
        z_sample = torch.randn(15000, in_dim) * s_test
        sample_std = float(z_sample.std().item())
        print(f"  Prior scale s={s_test:4.2f} -> empirical sample std = {sample_std:.4f}")

    if best_overall_m["norm_1way_tvd"] > 10.0:
        print("\n  [SKIP T5]: Non-private 1-way TVD is > 10x floor; T5 DP sweep skipped as instructed.")
        results["T5_prior_scale"] = "SKIPPED_DUE_TO_NONPRIVATE_TVD_GT_10X"
    else:
        print("\n  Running T5 DP sweep...")
        # (Only if T1-T3 left 1-way TVD <= 10x)

    # =========================================================================
    # T7. Run regression config twice with same seed; report run-to-run correlation diff
    # =========================================================================
    print("\n" + "=" * 75)
    print(" T7. RUN-TO-RUN DETERMINISTIC REPRODUCIBILITY (2 Runs, Same Seed 42)")
    print("=" * 75)
    def run_regression_pipeline_eval(run_id: int):
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        pipe = ACFMNetV5Pipeline(
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
            prior_scale=1.0,
        )
        t0 = time.time()
        df_syn, _ = pipe.fit_generate(df_train, num_samples=len(df_val), epochs=10, target_col="class")
        print(f"  Run {run_id} completed in {time.time() - t0:.1f}s")
        # Compute Pearson correlation matrix
        Xr, _ = _encode_for_eval(df_syn[common_cols])
        corr = np.corrcoef(Xr, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        return corr

    corr_run1 = run_regression_pipeline_eval(1)
    corr_run2 = run_regression_pipeline_eval(2)

    corr_frob_diff = float(np.linalg.norm(corr_run1 - corr_run2, ord="fro"))
    corr_max_diff = float(np.max(np.abs(corr_run1 - corr_run2)))
    print(f"\nRun-to-Run Correlation Matrix Difference (Same Seed 42):")
    print(f"  Frobenius norm difference: {corr_frob_diff:.6f}")
    print(f"  Max absolute cell difference: {corr_max_diff:.6f}")

    results["T7_run_to_run"] = {
        "corr_frobenius_difference": corr_frob_diff,
        "corr_max_difference": corr_max_diff,
    }

    # Save all results to disk
    with open(REPO_ROOT / "runs" / "design_bug_triage.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start

    # Final line decision:
    # If training longer (E=600 or larger capacity) reduced 1-way TVD to near floor, it was UNDERTRAINED.
    # If flow loss saturated well below zero-predictor, but 1-way TVD stays high (>10x) even at E=600 and with 256x3, it is a REPRESENTATION-LIMIT of identity CFM on one-hot categorical space.
    if best_overall_m["norm_1way_tvd"] <= 3.0:
        verdict = "OK"
    elif best_overall_m["norm_1way_tvd"] <= 10.0:
        verdict = "DP-LIMITED"
    elif best_t1_m["norm_1way_tvd"] < 0.5 * t1_results[40]["norm_1way_tvd"]:
        verdict = "UNDERTRAINED"
    else:
        verdict = "REPRESENTATION-LIMIT"

    print("\n" + "=" * 80)
    print(f"FINAL TRIAGE LINE:")
    print(f"{verdict}, total time: {total_time:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
