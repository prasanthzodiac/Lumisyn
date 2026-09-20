import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline

def main():
    print("================================================================================")
    print(" ITEM 4 & 7: DP EPS=1 LATENT-DEPENDENCE TEST & C SWEEP ON ADULT 30k")
    print("================================================================================")

    data_path = "lumisyn/data/kaggle_datasets/adult.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    full_df = pd.read_csv(data_path)
    if "id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df_30k = full_df.sample(n=30000, random_state=42).reset_index(drop=True)
    print(f"Dataset: Adult 30k random subsample, shape: {df_30k.shape}")

    # Sweep C around median norm (~0.32): C in [0.16, 0.32, 0.64]
    c_values = [0.16, 0.32, 0.64]
    results = []

    for C in c_values:
        print(f"\n========================================================================")
        print(f" RUNNING DP-SGD WITH C = {C:.2f} (C_mult = {C/0.32:.2f}x around 0.32)")
        print(f"========================================================================")
        torch.manual_seed(42)
        np.random.seed(42)

        pipe = ACFMNetV5Pipeline(
            target_epsilon=1.0,
            target_delta=1e-5,
            strict_release_privacy=True,
            allow_nonprivate_debug=False,
            batch_size=512,
            max_grad_norm_hvae=C,
            beta_kl=0.001,
            free_bits=0.0,
            disable_gat=True,
        )

        t0 = time.time()
        # Train representation & flow
        df_syn, report = pipe.fit_generate(
            df_30k.copy(),
            num_samples=1000,
            epochs=5,
            target_col="class"
        )
        runtime = time.time() - t0

        # Extract latent-dependence summary
        ld = report.get("latent_dependence_summary", {})
        hvae_log = report.get("hvae_stability_log", [])
        last_hvae = hvae_log[-1] if hvae_log else {}

        post_mse = ld.get("posterior_mse", float("nan"))
        shuf_mse = ld.get("shuffled_mse", float("nan"))
        prior_mse = ld.get("prior_mse", float("nan"))
        mean_pred = ld.get("mean_pred_mse", float("nan"))
        shuf_ratio = ld.get("shuffle_to_posterior_ratio", float("nan"))
        ld_pass = ld.get("pass_rule", False)

        avg_clipped = float(np.mean([ep.get("fraction_clipped", 0.0) for ep in hvae_log])) if hvae_log else 0.0
        avg_mean_norm = float(np.mean([ep.get("mean_grad_norm", 0.0) for ep in hvae_log])) if hvae_log else 0.0

        res = {
            "C": C,
            "C_mult": round(C / 0.32, 2),
            "avg_clipped_frac": round(avg_clipped, 4),
            "avg_mean_norm": round(avg_mean_norm, 4),
            "posterior_mse": round(post_mse, 6),
            "shuffled_mse": round(shuf_mse, 6),
            "prior_mse": round(prior_mse, 6),
            "mean_pred_mse": round(mean_pred, 6),
            "shuffle_ratio": round(shuf_ratio, 2),
            "ld_pass": ld_pass,
            "runtime_s": round(runtime, 1),
            "accounted_eps": report.get("privacy_summary", {}).get("prv_upper_with_laplace"),
        }
        results.append(res)

        print(f"\n--- Results for C = {C} ---")
        print(f"  Avg Clipped Frac   : {avg_clipped:.2%}")
        print(f"  Avg Mean Grad Norm : {avg_mean_norm:.4f}")
        print(f"  Posterior MSE      : {post_mse:.6f}")
        print(f"  Shuffled MSE       : {shuf_mse:.6f} (ratio: {shuf_ratio:.2f}x)")
        print(f"  Mean-pred MSE      : {mean_pred:.6f}")
        print(f"  LD Test Passed     : {ld_pass}")
        print(f"  Accounted Epsilon  : {res['accounted_eps']}")

    print("\n================================================================================")
    print(" DP-SGD LATENT-DEPENDENCE & C-SWEEP SUMMARY")
    print("================================================================================")
    print(f" {'C':<6s} | {'C_mult':<8s} | {'Clip%':<8s} | {'Norm':<8s} | {'Post MSE':<10s} | {'Shuf MSE':<10s} | {'Shuf/Post':<10s} | {'LD PASS':<8s}")
    print(" " + "-" * 78)
    for r in results:
        print(f" {r['C']:<6.2f} | {r['C_mult']:<8.2f} | {r['avg_clipped_frac']:<8.2%} | {r['avg_mean_norm']:<8.4f} | {r['posterior_mse']:<10.6f} | {r['shuffled_mse']:<10.6f} | {r['shuffle_ratio']:<10.2f}x | {str(r['ld_pass']):<8s}")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
