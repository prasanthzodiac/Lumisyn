"""
Detailed Feature-by-Feature Comparison Script
Compares Original Real NHANES Data vs. ACFM-Net v5 Synthetic Data.
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from test_local_nhanes import load_and_merge_local_nhanes
from acfm_net.pipeline import ACFMNetV5Pipeline

def run_comparison():
    print("=" * 90)
    print("      ACFM-Net v5 — DETAILED ORIGINAL VS. SYNTHETIC COMPARISON")
    print("=" * 90)

    # 1. Load Original Data
    df_real = load_and_merge_local_nhanes(sample_size=1500)
    
    # 2. Run ACFM-Net v5 Pipeline to generate Synthetic Data
    pipeline = ACFMNetV5Pipeline(target_epsilon=1.0, target_delta=1e-5)
    df_syn, _ = pipeline.fit_generate(
        df_real,
        num_samples=1500,
        target_col="diabetes_outcome",
        epochs=10,
        protocol_id="IRB-NHANES-COMPARISON"
    )

    print("\n" + "=" * 90)
    print(" 1. CONTINUOUS FEATURE MARGINAL STATISTICS COMPARISON")
    print("=" * 90)

    continuous_features = ["age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol"]
    
    print(f"{'Feature':<15} | {'Real Mean (Std)':<22} | {'Syn Mean (Std)':<22} | {'Mean Diff':<10} | {'Wasserstein Dist':<16} | {'KS Stat (p-val)':<18}")
    print("-" * 115)

    for feat in continuous_features:
        if feat in df_real.columns and feat in df_syn.columns:
            r_vals = df_real[feat].dropna().values
            s_vals = df_syn[feat].dropna().values

            r_mean, r_std = np.mean(r_vals), np.std(r_vals)
            s_mean, s_std = np.mean(s_vals), np.std(s_vals)
            mean_diff = abs(r_mean - s_mean)
            
            w_dist = wasserstein_distance(r_vals, s_vals)
            ks_res = ks_2samp(r_vals, s_vals)

            r_str = f"{r_mean:.2f} (±{r_std:.2f})"
            s_str = f"{s_mean:.2f} (±{s_std:.2f})"
            ks_str = f"{ks_res.statistic:.3f} (p={ks_res.pvalue:.3f})"

            print(f"{feat:<15} | {r_str:<22} | {s_str:<22} | {mean_diff:<10.3f} | {w_dist:<16.3f} | {ks_str:<18}")

    print("\n" + "=" * 90)
    print(" 2. CATEGORICAL FEATURE FREQUENCY COMPARISON (%)")
    print("=" * 90)

    categorical_features = ["gender", "ethnicity", "diabetes_outcome"]
    for feat in categorical_features:
        if feat in df_real.columns and feat in df_syn.columns:
            print(f"\nFeature: [{feat.upper()}]")
            r_freq = df_real[feat].value_counts(normalize=True) * 100
            s_freq = df_syn[feat].value_counts(normalize=True) * 100
            
            all_cats = set(r_freq.index).union(set(s_freq.index))
            print(f"  {'Category':<22} | {'Real Freq (%)':<15} | {'Syn Freq (%)':<15} | {'Diff (%)':<10}")
            print("  " + "-" * 68)
            for cat in all_cats:
                rf = r_freq.get(cat, 0.0)
                sf = s_freq.get(cat, 0.0)
                print(f"  {str(cat):<22} | {rf:<15.2f} | {sf:<15.2f} | {abs(rf - sf):<10.2f}")

    print("\n" + "=" * 90)
    print(" 3. PAIRWISE CORRELATION MATRIX COMPARISON")
    print("=" * 90)

    num_cols = ["age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol"]
    r_corr = df_real[num_cols].dropna().corr()
    s_corr = df_syn[num_cols].dropna().corr()
    diff_corr = np.abs(r_corr - s_corr)

    print("\n  Pairwise Absolute Correlation Difference Matrix |Real - Synthetic|:")
    print(diff_corr.round(4).to_string())
    
    frob_norm = np.linalg.norm(r_corr.values - s_corr.values, 'fro')
    avg_corr_err = diff_corr.values[np.triu_indices_from(diff_corr, k=1)].mean()
    print(f"\n  • Frobenius Norm Error:              {frob_norm:.4f}")
    print(f"  • Average Upper-Tri Correlation Error: {avg_corr_err:.4f}")

    print("\n" + "=" * 90)
    print(" 4. CLINICAL DOMAIN CONSTRAINT VERIFICATION")
    print("=" * 90)

    r_sys_dia = (df_real["systolic_bp"] >= df_real["diastolic_bp"]).mean() * 100
    s_sys_dia = (df_syn["systolic_bp"] >= df_syn["diastolic_bp"]).mean() * 100

    print(f"  • Real Systolic >= Diastolic Compliance:      {r_sys_dia:.2f}%")
    print(f"  • Synthetic Systolic >= Diastolic Compliance: {s_sys_dia:.2f}%")

if __name__ == "__main__":
    run_comparison()
