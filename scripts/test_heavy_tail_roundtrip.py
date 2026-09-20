import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.representation.hvae import HierarchicalVAE

def main():
    print("================================================================================")
    print(" GATE 4c ITEM 6: NON-PRIVATE ROUND-TRIP WITH SCHEMA-LEVEL LOG1P ON HEAVY TAILS")
    print("================================================================================")

    data_path = "lumisyn/data/kaggle_datasets/adult.csv"
    schema_path = "lumisyn/data/schemas/adult_schema.yaml"

    with open(schema_path, "r") as f:
        schema_dict = yaml.safe_load(f)["columns"]

    full_df = pd.read_csv(data_path)
    if "id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df = full_df.sample(n=30000, random_state=42).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Sample N={len(df)}")

    pipe = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        allow_nonprivate_debug=True,
        batch_size=512,
        disable_gat=True,
    )

    # 1. Pipeline preparation with public schema
    df_imp, _ = pipe.imputer.fit_transform(df, None, privacy_mode=True, public_schema=schema_dict)
    encoded_X, enc_info = pipe.encoder.fit_transform(df_imp, privacy_schema=schema_dict)
    scaled_X = pipe.scaler.fit_transform(
        encoded_X,
        enc_info["feature_specs"],
        privacy_mode=True,
        encoded_column_names=enc_info["encoded_column_names"],
    )

    X_tensor = torch.tensor(scaled_X, dtype=torch.float32, device=device)
    in_dim = X_tensor.shape[1]
    print(f"Encoded and scaled X shape: {X_tensor.shape}, in_dim={in_dim}")

    # 2. Train non-private HVAE (15 epochs)
    hvae = HierarchicalVAE(
        in_dim=in_dim,
        global_dim=64,
        group_dim=32,
        num_groups=4,
        beta_kl=0.001,
        free_bits=0.0,
    ).to(device)

    opt = torch.optim.AdamW(hvae.parameters(), lr=2e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=True)

    print("\n[Training Non-Private HVAE (15 epochs)]")
    for ep in range(1, 16):
        hvae.train()
        ep_loss = 0.0
        batches = 0
        for (xb,) in loader:
            opt.zero_grad()
            out = hvae(xb, target=xb, epoch=ep)
            loss = out["recon_loss"] + 0.001 * out["kl_loss"]
            loss.backward()
            opt.step()
            ep_loss += float(out["recon_loss"].item())
            batches += 1
        if ep % 5 == 0 or ep == 15:
            print(f"  Epoch {ep:2d}/15 | Recon MSE: {ep_loss/batches:.6f}")

    # 3. Round trip evaluation
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
        recon_mse = float(F.mse_loss(recon_scaled, X_tensor).item())

    clamped_np = recon_scaled.cpu().numpy()
    recon_encoded = pipe.scaler.inverse_transform(clamped_np)
    df_recon = pipe.encoder.decode(recon_encoded)

    print("\n================================================================================")
    print(" ROUND-TRIP RECONSTRUCTION FIDELITY COMPARISON (WITH LOG1P)")
    print("================================================================================")
    print(f"  Posterior Recon MSE in X_scaled space: {recon_mse:.6f}")

    # Continuous features evaluation
    num_cols = ["age", "fnlwgt", "hours-per-week", "capital-gain", "capital-loss"]
    for col in num_cols:
        orig = df[col].values
        recon = df_recon[col].values
        mae = float(np.mean(np.abs(orig - recon)))
        std = float(np.std(orig))
        rel_mae = (mae / std) * 100 if std > 0 else 0.0
        print(f"  {col:<16} | MAE: {mae:<10.2f} | Std: {std:<10.2f} | Rel MAE: {rel_mae:<6.2f}%")

if __name__ == "__main__":
    main()
