import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.representation.gat_encoder import GraphAttentionEncoder
from acfm_net.representation.cross_attention import GatedCrossAttention
from acfm_net.representation.hvae import HierarchicalVAE

def main():
    print("================================================================================")
    print(" ITEM 1: DIAGNOSE THE DECODE PATH")
    print("================================================================================")

    data_path = "lumisyn/data/kaggle_datasets/adult.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    df = pd.read_csv(data_path).head(5000)
    print(f"Loaded Adult subset: {df.shape}")

    pipeline = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        allow_nonprivate_debug=True,
        batch_size=256,
        disable_gat=False,
    )

    # Run up to Stage 2 to get X_scaled
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

    X_scaled = torch.tensor(scaled_X, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = X_scaled.to(device)
    in_dim = X_tensor.shape[1]

    # Graph
    enc_cols = enc_info["encoded_column_names"]
    d_enc = len(enc_cols)
    _, W_0, g_info = pipeline.graph_builder.build_graph(
        columns=enc_cols,
        dependency_matrix=np.eye(d_enc, dtype=np.float32),
        sensitivity_vector=np.ones(d_enc, dtype=np.float32) * 0.5,
        df_encoded=None,
        target_col=None,
    )
    adj_matrix = torch.tensor(W_0, dtype=torch.float32, device=device)

    # Components
    gat = GraphAttentionEncoder(in_features=1, hidden_dim=64).to(device)
    cross_attn = GatedCrossAttention(feature_dim=in_dim, graph_dim=64, hidden_dim=64).to(device)
    hvae = HierarchicalVAE(in_dim=in_dim, global_dim=64, group_dim=32, num_groups=4).to(device)

    with torch.no_grad():
        h_gat_all = gat(X_tensor, adj_matrix)
        X_gated_all = cross_attn(X_tensor, h_gat_all)
        
        # Pass X_gated_all through HVAE as was done originally
        hvae_out = hvae(X_gated_all)
        Z = hvae_out["Z"]
        decoder_output = hvae.decode(Z)

    def stats(name, t):
        t_np = t.detach().cpu().numpy()
        var_per_dim = t_np.var(axis=0)
        return {
            "name": name,
            "shape": list(t.shape),
            "min": float(t_np.min()),
            "max": float(t_np.max()),
            "mean": float(t_np.mean()),
            "total_var": float(t_np.var()),
            "min_dim_var": float(var_per_dim.min()),
            "max_dim_var": float(var_per_dim.max()),
            "mean_dim_var": float(var_per_dim.mean()),
        }

    s_scaled = stats("(a) X_scaled (public-bounds scaled input)", X_tensor)
    s_gated = stats("(b) X_gated_all (cross-attn output)", X_gated_all)
    s_dec = stats("(c) Decoder output hvae.decode(Z)", decoder_output)

    print("\n--- TENSOR DIAGNOSTICS ---")
    for s in [s_scaled, s_gated, s_dec]:
        print(f"\n{s['name']}:")
        print(f"  Shape       : {s['shape']}")
        print(f"  Range       : [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"  Mean        : {s['mean']:.4f}")
        print(f"  Variance    : {s['total_var']:.6f} (mean per-dim var: {s['mean_dim_var']:.6f}, range [{s['min_dim_var']:.6f}, {s['max_dim_var']:.6f}])")

    print("\n================================================================================")
    print(" SPACE MISMATCH DIAGNOSIS & SPECIFICATION")
    print("================================================================================")
    print("1. Space HVAE Decoder is TRAINED to reconstruct:")
    print("   -> X_gated_all space. In pipeline.py line 680, hvae(X_gated) was called, where")
    print("      hvae computes F.mse_loss(x_recon, x). Thus, x_recon targets X_gated_all.")
    print("   -> Notice X_gated_all has a compressed dynamic range and reduced variance compared to X_scaled.")
    print("\n2. Space Generation Path INVERSE-TRANSFORMS back to data:")
    print("   -> X_scaled space (in [-1, 1]^d). In pipeline.py lines 1122-1124:")
    print("        synth_np_scaled = repaired_synth_scaled.cpu().numpy()")
    print("        synth_np_encoded = self.scaler.inverse_transform(synth_np_scaled)")
    print("      AdaptiveFeatureScaler.inverse_transform expects inputs in [-1, 1] scaled space!")
    print("   -> Because the decoder was trained to predict X_gated_all rather than X_scaled,")
    print("      the decoder output does NOT represent X_scaled. When passed into inverse_transform,")
    print("      it produces invalid feature values, corrupting categorical probabilities and exploding")
    print("      continuous bounds, causing DCR to explode to ~1455.")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
