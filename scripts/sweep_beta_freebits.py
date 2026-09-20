import os
import sys
import time
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("lumisyn/framework"))
from acfm_net.pipeline import ACFMNetV5Pipeline

def main():
    print("================================================================================")
    print(" GATE 4b ITEM 2: SWEEP KL WEIGHT BETA AND FREE-BITS UNDER DP EPS=1.0")
    print("================================================================================")

    data_path = "lumisyn/data/kaggle_datasets/adult.csv"
    full_df = pd.read_csv(data_path)
    if "id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df_30k = full_df.sample(n=30000, random_state=42).reset_index(drop=True)

    beta_list = [0.001, 0.01, 0.1, 1.0]
    fb_list = [0.0, 0.1]

    results = []

    for fb in fb_list:
        for beta in beta_list:
            print(f"\n>>> Running free_bits={fb}, beta_kl={beta} (DP eps=1.0, N=30000)")
            pipe = ACFMNetV5Pipeline(
                target_epsilon=1.0,
                target_delta=1e-5,
                strict_release_privacy=True,
                batch_size=512,
                max_grad_norm_hvae=0.32,
                beta_kl=beta,
                free_bits=fb,
                disable_gat=True,
            )
            t0 = time.time()
            _, rep = pipe.fit_generate(
                df_30k.copy(),
                num_samples=100,
                epochs=5,
                target_col="class"
            )
            dur = time.time() - t0

            ld = rep.get("latent_dependence_summary", {})
            hvae_log = rep.get("hvae_stability_log", [])

            post_mse = ld.get("posterior_mse", float("nan"))
            shuf_mse = ld.get("shuffled_mse", float("nan"))
            prior_mse = ld.get("prior_mse", float("nan"))
            mean_pred = ld.get("mean_pred_mse", float("nan"))
            shuf_ratio = ld.get("shuffle_to_posterior_ratio", float("nan"))
            total_kl = ld.get("total_kl_nats", float("nan"))
            ld_pass = ld.get("pass_rule", False)

            avg_clipped = float(np.mean([ep.get("fraction_clipped", 0.0) for ep in hvae_log])) if hvae_log else 0.0
            avg_norm = float(np.mean([ep.get("mean_grad_norm", 0.0) for ep in hvae_log])) if hvae_log else 0.0

            res = {
                "free_bits": fb,
                "beta_kl": beta,
                "avg_clipped": round(avg_clipped, 4),
                "avg_norm": round(avg_norm, 4),
                "post_mse": round(post_mse, 6),
                "shuf_mse": round(shuf_mse, 6),
                "prior_mse": round(prior_mse, 6),
                "mean_pred": round(mean_pred, 6),
                "shuf_ratio": round(shuf_ratio, 2),
                "total_kl": round(total_kl, 4),
                "ld_pass": ld_pass,
                "dur_s": round(dur, 1),
            }
            results.append(res)
            print(f"Result: fb={fb}, beta={beta} -> Post MSE={post_mse:.6f}, Shuf MSE={shuf_mse:.6f}, Ratio={shuf_ratio:.2f}x, KL={total_kl:.4f} nats, PASS={ld_pass}")

    print("\n" + "=" * 80)
    print(" SUMMARY OF BETA x FREE-BITS SWEEP UNDER DP EPS=1.0")
    print("=" * 80)
    print(f"{'FB':<6} | {'Beta':<6} | {'Clip%':<7} | {'Norm':<7} | {'Post MSE':<10} | {'Shuf MSE':<10} | {'Ratio':<7} | {'KL (nats)':<10} | {'PASS':<5}")
    print("-" * 80)
    for r in results:
        print(f"{r['free_bits']:<6.1f} | {r['beta_kl']:<6.3f} | {r['avg_clipped']*100:<6.1f}% | {r['avg_norm']:<7.4f} | {r['post_mse']:<10.6f} | {r['shuf_mse']:<10.6f} | {r['shuf_ratio']:<7.2f} | {r['total_kl']:<10.4f} | {str(r['ld_pass']):<5}")
    print("=" * 80)

if __name__ == "__main__":
    main()
