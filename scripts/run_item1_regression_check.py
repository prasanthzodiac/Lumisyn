import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Deterministic flags
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.representation.hvae import HierarchicalVAE
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    SlicedWassersteinEvaluator,
    _encode_for_eval,
)

def load_adult_30k_and_holdout(seed: int = 42):
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
    train_idx = perm[:30000]
    holdout_idx = perm[30000:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)
    return df_train, df_holdout

def compute_metrics(df_real_train, df_syn, df_holdout):
    common = [c for c in df_real_train.columns if c in df_syn.columns and not c.endswith("_isna")]
    marg = MarginalErrorEvaluator()
    eval_1w = marg.evaluate_1way(df_real_train[common], df_syn[common], df_real_holdout=df_holdout[common])
    
    eval_2w = marg.evaluate_2way(df_real_train[common], df_syn[common], max_pairs=30, seed=42)
    tvd_2w = eval_2w["mean_2way_marginal_error_tvd"]
    floor_2w_eval = marg.evaluate_2way(df_real_train[common], df_holdout[common], max_pairs=30, seed=42)
    floor_2w = floor_2w_eval["mean_2way_marginal_error_tvd"]
    norm_tvd_2w = round(tvd_2w / max(1e-4, floor_2w), 4)

    corr_eval = CorrelationErrorEvaluator()
    eval_corr = corr_eval.evaluate(df_real_train[common], df_syn[common], df_real_holdout=df_holdout[common])
    frob_corr = eval_corr.get("pearson_frobenius_error", 0.0)
    floor_corr = eval_corr.get("real_vs_real_floor_pearson_frobenius", 0.0430)
    norm_corr = eval_corr.get("normalized_pearson_frobenius_error", frob_corr / max(1e-4, floor_corr))

    return {
        "norm_1way_tvd": eval_1w.get("normalized_1way_marginal_error", 1.0),
        "norm_2way_tvd": norm_tvd_2w,
        "norm_frob_corr": norm_corr,
        "raw_1way_tvd": eval_1w.get("mean_1way_marginal_error_tvd", 0.0),
        "floor_1way": eval_1w.get("real_vs_real_floor_1way_tvd", 0.0076),
    }

def main():
    print("=" * 80)
    print(" REGRESSION CHECK 1: PIPELINE CONFIG RERUN (IDENTITY MODE, DP eps=1, seed 42)")
    print("=" * 80)
    df_train, df_holdout = load_adult_30k_and_holdout(seed=42)

    # 1. Rerun pipeline config
    torch.manual_seed(42)
    np.random.seed(42)
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
        max_grad_norm_flow=3.0822, # 1x median norm
        prior_scale=1.0,
    )
    t0 = time.time()
    df_syn, rep = pipe.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
    t_pipe = time.time() - t0
    print(f"Pipeline fit_generate completed in {t_pipe:.2f}s")

    metrics = compute_metrics(df_train, df_syn, df_holdout)
    print(f"Rerun Metrics (d={pipe.encoder.feature_dim if hasattr(pipe.encoder, 'feature_dim') else 'unknown'}):")
    print(f"  norm_1way_tvd:  {metrics['norm_1way_tvd']:.4f}x")
    print(f"  norm_2way_tvd:  {metrics['norm_2way_tvd']:.4f}x")
    print(f"  norm_frob_corr: {metrics['norm_frob_corr']:.4f}x")

    # Target from gate4d_results.jsonl row 6:
    # {"norm_1way_tvd": 69.26, "norm_2way_tvd": 34.03, "norm_frob_corr": 12.47}
    ref_1w = 69.26
    ref_2w = 34.03
    ref_corr = 12.47
    print("\nComparison to runs/gate4d_results.jsonl (Row 6, seed 42 A3 winner):")
    print(f"  1-way TVD:  Rerun={metrics['norm_1way_tvd']:.2f} vs Logged={ref_1w:.2f} (diff: {abs(metrics['norm_1way_tvd']-ref_1w)/ref_1w:.2%})")
    print(f"  2-way TVD:  Rerun={metrics['norm_2way_tvd']:.2f} vs Logged={ref_2w:.2f} (diff: {abs(metrics['norm_2way_tvd']-ref_2w)/ref_2w:.2%})")
    print(f"  Corr Frob:  Rerun={metrics['norm_frob_corr']:.2f} vs Logged={ref_corr:.2f} (diff: {abs(metrics['norm_frob_corr']-ref_corr)/ref_corr:.2%})")

    # 2. Non-private round trip at d=78 vs logged d=71 (0.001604)
    print("\n" + "=" * 80)
    print(" REGRESSION CHECK 2: NON-PRIVATE ROUND-TRIP AT d=78 (POST-ALIGNMENT)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema_path = REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml"
    import yaml
    with open(schema_path, "r", encoding="utf-8") as f:
        adult_schema = yaml.safe_load(f)

    # Impute, encode, scale with adult_schema
    df_imputed, _ = pipe.imputer.fit_transform(df_train, None, privacy_mode=True, public_schema=adult_schema)
    encoded_X, enc_info = pipe.encoder.fit_transform(df_imputed, privacy_schema=adult_schema)
    scaled_X = pipe.scaler.fit_transform(
        encoded_X,
        enc_info["feature_specs"],
        privacy_mode=True,
        encoded_column_names=enc_info["encoded_column_names"],
    )
    X_tensor = torch.tensor(scaled_X, dtype=torch.float32, device=device)
    in_dim = X_tensor.shape[1]
    print(f"Encoded and scaled X shape: {X_tensor.shape} (d={in_dim})")

    torch.manual_seed(42)
    hvae = HierarchicalVAE(
        in_dim=in_dim,
        global_dim=64,
        group_dim=32,
        num_groups=4,
        beta_kl=0.001,
        free_bits=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(hvae.parameters(), lr=2e-3, weight_decay=1e-5)
    dataloader = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=True)

    epochs = 15
    for ep in range(1, epochs + 1):
        hvae.train()
        for (xb,) in dataloader:
            optimizer.zero_grad()
            out = hvae(xb, target=xb, epoch=ep)
            loss = out["recon_loss"] + 0.001 * out["kl_loss"]
            loss.backward()
            optimizer.step()

    hvae.eval()
    with torch.no_grad():
        h = hvae.encoder_backbone(X_tensor)
        mu_g = hvae.fc_mu_g(h)
        mus = [mu_g]
        for head in hvae.group_heads:
            h_g = torch.cat([h, mu_g], dim=-1)
            out_head = head(h_g)
            mu_k, _ = torch.chunk(out_head, 2, dim=-1)
            mus.append(mu_k)
        z = torch.cat(mus, dim=-1)
        recon_scaled = hvae.decode(z)
        recon_scaled = torch.clamp(recon_scaled, -1.0, 1.0)
        recon_mse_d78 = float(F.mse_loss(recon_scaled, X_tensor).item())

    logged_mse_d71 = 0.001604
    print(f"Posterior-mean MSE at d={in_dim}: {recon_mse_d78:.6f} (logged at d=71: {logged_mse_d71:.6f})")

    results = {
        "identity_pipeline_rerun": {
            "d_enc": in_dim,
            "metrics": metrics,
            "ref_1w": ref_1w,
            "ref_2w": ref_2w,
            "ref_corr": ref_corr,
        },
        "nonprivate_roundtrip": {
            "d_enc": in_dim,
            "recon_mse": recon_mse_d78,
            "logged_mse_d71": logged_mse_d71,
        }
    }
    with open(REPO_ROOT / "runs" / "item1_regression_check.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to runs/item1_regression_check.json")

if __name__ == "__main__":
    main()
