import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.representation.hvae import HierarchicalVAE

def run_roundtrip_test():
    print("================================================================================")
    print(" ITEM 3 & 4: NON-PRIVATE ROUND-TRIP & LATENT-DEPENDENCE TEST (ADULT 30k)")
    print("================================================================================")

    data_path = "lumisyn/data/kaggle_datasets/adult.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    full_df = pd.read_csv(data_path)
    # Deduplicate / check 1 row per patient
    if "id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df = full_df.sample(n=30000, random_state=42).reset_index(drop=True)
    print(f"Dataset: Adult (sample N=30,000, raw cols={df.shape[1]})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipeline = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        allow_nonprivate_debug=True,
        batch_size=512,
        disable_gat=True,
    )

    # 1. Pipeline preparation to get X_scaled
    df_imputed, missing_info = pipeline.imputer.fit_transform(
        df, None, privacy_mode=True, public_schema=None
    )
    encoded_X, enc_info = pipeline.encoder.fit_transform(
        df_imputed, privacy_schema=None
    )
    scaled_X = pipeline.scaler.fit_transform(
        encoded_X,
        enc_info["feature_specs"],
        privacy_mode=True,
        encoded_column_names=enc_info["encoded_column_names"],
    )

    X_tensor = torch.tensor(scaled_X, dtype=torch.float32, device=device)
    in_dim = X_tensor.shape[1]
    N = X_tensor.shape[0]
    print(f"Encoded and scaled X: {X_tensor.shape}, range: [{X_tensor.min():.4f}, {X_tensor.max():.4f}]")

    # Baselines in X_scaled space
    with torch.no_grad():
        zero_pred_mse = float(torch.mean(X_tensor ** 2).item())
        mean_vec = X_tensor.mean(dim=0, keepdim=True)
        mean_pred_mse = float(torch.mean((X_tensor - mean_vec) ** 2).item())
    print(f"\n[Baselines in X_scaled space]")
    print(f"  Zero-predictor MSE: {zero_pred_mse:.6f}")
    print(f"  Mean-predictor MSE: {mean_pred_mse:.6f}")

    # 2. Non-private HVAE training on fixed X_scaled target
    hvae = HierarchicalVAE(
        in_dim=in_dim,
        global_dim=64,
        group_dim=32,
        num_groups=4,
        beta_kl=0.001,   # small beta for reconstruction fidelity
        free_bits=0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(hvae.parameters(), lr=2e-3, weight_decay=1e-5)
    dataloader = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=True)

    epochs = 15
    print(f"\n[Training HVAE Non-Privately (AdamW, lr=2e-3, {epochs} epochs, batch=512)]")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        hvae.train()
        ep_recon = 0.0
        ep_kl = 0.0
        batches = 0
        for (xb,) in dataloader:
            optimizer.zero_grad()
            out = hvae(xb, target=xb, epoch=ep)
            loss = out["recon_loss"] + 0.001 * out["kl_loss"]
            loss.backward()
            optimizer.step()
            ep_recon += float(out["recon_loss"].item())
            ep_kl += float(out["kl_loss"].item())
            batches += 1
        avg_recon = ep_recon / batches
        avg_kl = ep_kl / batches
        if ep % 3 == 0 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs} | Recon MSE={avg_recon:.6f} | KL={avg_kl:.6f}")
    print(f"Training completed in {time.time() - t0:.2f}s")

    # 3. Round-trip test: real X -> posterior-mean z -> decoder -> inverse transform -> compare
    hvae.eval()
    with torch.no_grad():
        # Encode to get posterior distributions
        h = hvae.encoder_backbone(X_tensor)
        mu_g = hvae.fc_mu_g(h)
        mus = [mu_g]
        for head in hvae.group_heads:
            h_g = torch.cat([h, mu_g], dim=-1)
            out_head = head(h_g)
            mu_k, _ = torch.chunk(out_head, 2, dim=-1)
            mus.append(mu_k)
        posterior_mean_z = torch.cat(mus, dim=-1)

        # Decode posterior-mean z
        X_recon = hvae.decode(posterior_mean_z)
        recon_mse = float(torch.mean((X_tensor - X_recon) ** 2).item())

    print("\n================================================================================")
    print(" ROUND-TRIP TEST RESULTS IN X_SCALED SPACE")
    print("================================================================================")
    print(f"  Zero-predictor MSE : {zero_pred_mse:.6f}")
    print(f"  Mean-predictor MSE : {mean_pred_mse:.6f}")
    print(f"  Posterior-mean MSE : {recon_mse:.6f}")
    print(f"  Recon vs Zero-pred : {recon_mse / zero_pred_mse:.4f}x ({recon_mse/zero_pred_mse*100:.1f}%)")
    print(f"  Recon vs Mean-pred : {recon_mse / mean_pred_mse:.4f}x (Ratio: {mean_pred_mse / max(1e-9, recon_mse):.2f}x better than mean predictor)")

    # Decode back to data space
    with torch.no_grad():
        clamped_recon = torch.clamp(X_recon, -1.0, 1.0).cpu().numpy()
        recon_encoded = pipeline.scaler.inverse_transform(clamped_recon)
        df_recon = pipeline.encoder.decode(recon_encoded)

    print(f"\n--- Original vs Reconstructed Rows Sample (First 3 rows) ---")
    cols_to_show = [c for c in df.columns if c in df_recon.columns][:8]
    print("ORIGINAL:")
    print(df[cols_to_show].head(3))
    print("\nRECONSTRUCTED (Round-trip):")
    print(df_recon[cols_to_show].head(3))

    # Feature-wise reconstruction error
    print("\n--- Feature-wise Reconstruction Accuracy / Error ---")
    for col in df.columns:
        if col not in df_recon.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mae = float(np.mean(np.abs(df[col].values - df_recon[col].values)))
            std = float(np.std(df[col].values))
            print(f"  [Numeric] {col:25s}: MAE = {mae:.4f} (std = {std:.4f}, rel = {mae/max(1e-5, std):.2%})")
        else:
            acc = float(np.mean(df[col].astype(str).values == df_recon[col].astype(str).values))
            print(f"  [Categorical] {col:21s}: Exact Match Acc = {acc:.2%}")

    # 4. Latent-Dependence Test in X_scaled space
    print("\n================================================================================")
    print(" LATENT-DEPENDENCE TEST IN X_SCALED SPACE (NON-PRIVATE)")
    print("================================================================================")
    with torch.no_grad():
        # (a) Posterior z
        latent_Z, total_mu, total_logvar = hvae.encode(X_tensor)
        x_recon_post = hvae.decode(latent_Z)
        post_mse = float(torch.mean((X_tensor - x_recon_post) ** 2).item())

        # (b) Shuffled z
        perm = torch.randperm(latent_Z.shape[0])
        z_shuf = latent_Z[perm]
        x_recon_shuf = hvae.decode(z_shuf)
        shuf_mse = float(torch.mean((X_tensor - x_recon_shuf) ** 2).item())

        # (c) Prior z ~ N(0, I)
        z_pri = torch.randn_like(latent_Z)
        x_recon_pri = hvae.decode(z_pri)
        pri_mse = float(torch.mean((X_tensor - x_recon_pri) ** 2).item())

        kl_elem = -0.5 * (1.0 + total_logvar - total_mu.pow(2) - total_logvar.exp())
        tot_kl = float(kl_elem.sum(dim=-1).mean().item())

    shuf_ratio = shuf_mse / max(1e-9, post_mse)
    mean_ratio = post_mse / max(1e-9, mean_pred_mse)
    pass_rule_1 = shuf_ratio >= 3.0
    pass_rule_2 = post_mse <= (mean_pred_mse / 3.0)
    ld_pass = pass_rule_1 and pass_rule_2

    print(f"  Posterior z MSE : {post_mse:.6f}")
    print(f"  Shuffled z MSE  : {shuf_mse:.6f} (ratio vs posterior: {shuf_ratio:.2f}x)")
    print(f"  Prior z MSE     : {pri_mse:.6f}")
    print(f"  Mean-pred MSE   : {mean_pred_mse:.6f}")
    print(f"  Total KL (nats) : {tot_kl:.4f}")
    print(f"  Rule 1 (shuffled >= 3x posterior): {pass_rule_1} ({shuf_ratio:.2f}x >= 3.0x)")
    print(f"  Rule 2 (posterior <= mean_pred/3): {pass_rule_2} ({post_mse:.6f} <= {mean_pred_mse/3.0:.6f})")
    print(f"  OVERALL PASS    : {ld_pass}")
    print("================================================================================\n")

if __name__ == "__main__":
    run_roundtrip_test()
