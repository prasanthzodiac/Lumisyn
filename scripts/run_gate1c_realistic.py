import os
import sys
import time
import math
import subprocess
import numpy as np
import pandas as pd
import torch

# Ensure framework is in path
sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.evaluation.privacy import ClassifierTwoSampleTest, DuplicateMemorizationChecker

def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

def main():
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    git_hash = get_git_commit()
    seed = 2026
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("================================================================================")
    print(" GATE 1c REALISTIC BENCHMARK RUN: N = 5,000, q <= 0.05, DP-SGD Scaled")
    print(f" Timestamp: {timestamp}")
    print(f" Git Hash:  {git_hash}")
    print(f" Seed:      {seed}")
    print("================================================================================")

    # 1. Synthesize realistic clinical cohort: N=5,000 unique patients
    N = 5000
    print(f"\n[Data Prep] Synthesizing realistic cohort with N={N} unique patients (1 row per patient)...")
    patient_ids = [f"PT-{i:06d}" for i in range(1, N + 1)]
    age = np.random.normal(62.5, 14.2, size=N).clip(18.0, 95.0)
    bmi = np.random.normal(28.4, 5.8, size=N).clip(14.0, 55.0)
    systolic = np.random.normal(132.0, 18.5, size=N).clip(80.0, 210.0)
    diastolic = np.random.normal(82.0, 11.2, size=N).clip(50.0, 130.0)
    glucose = np.random.exponential(scale=35.0, size=N) + 85.0
    creatinine = np.random.lognormal(mean=0.1, sigma=0.4, size=N).clip(0.4, 12.0)
    admission_type = np.random.choice(["EMERGENCY", "ELECTIVE", "URGENT"], size=N, p=[0.60, 0.25, 0.15])
    gender = np.random.choice(["M", "F"], size=N, p=[0.52, 0.48])
    
    # Missingness in ~5% of BMI and ~8% of glucose to verify row-local indicator columns in raw space
    bmi[np.random.rand(N) < 0.05] = np.nan
    glucose[np.random.rand(N) < 0.08] = np.nan

    # Mortality outcome correlated with age, creatinine, and systolic BP
    logit = -3.5 + 0.03 * (np.nan_to_num(age, nan=62.5) - 60) + 0.25 * (np.nan_to_num(creatinine, nan=1.0) - 1.0) + 0.01 * (np.nan_to_num(systolic, nan=130) - 120)
    prob_mort = 1.0 / (1.0 + np.exp(-logit))
    mortality = (np.random.rand(N) < prob_mort).astype(int)

    df_cohort = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "bmi": bmi,
        "systolic": systolic,
        "diastolic": diastolic,
        "glucose": glucose,
        "creatinine": creatinine,
        "admission_type": admission_type,
        "gender": gender,
        "mortality": mortality,
    })

    # Verify patient uniqueness (1 row per patient, dedupe assert)
    assert df_cohort["patient_id"].nunique() == N, "Patient IDs must be strictly unique"
    print(f"[Data Prep] Patient uniqueness verified: {df_cohort['patient_id'].nunique()} unique IDs out of {len(df_cohort)} rows.")
    print(f"[Data Prep] Missing cells: bmi={df_cohort['bmi'].isna().sum()}, glucose={df_cohort['glucose'].isna().sum()}")
    print(f"[Data Prep] Mortality rate: {df_cohort['mortality'].mean():.4f} ({df_cohort['mortality'].sum()}/{N})")

    # 2. Configure Pipeline for Realistic DP-SGD (q <= 0.05)
    batch_size = 250  # 250 / 5000 = 0.05 <= 0.05
    q_target = batch_size / N
    print(f"\n[Config] Setting N={N}, batch_size={batch_size} => expected subsampling rate q={q_target:.4f} <= 0.05")

    pipeline = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        use_ot_coupling=False,   # I-CFM
        poisson_sampling=True,
        batch_size=batch_size,
    )

    epochs = 4  # Stage 8 flow epochs (Stage 4 rep epochs = 5)
    print(f"[Config] Stage 4 Representation Epochs: 5 | Stage 8 Flow Epochs: {epochs}")

    # 3. Fit & Generate
    print("\n--- BEGIN PIPELINE EXECUTION ---")
    df_syn, report = pipeline.fit_generate(
        df_raw=df_cohort,
        num_samples=500,
        target_col="mortality",
        epochs=epochs,
        protocol_id="IRB-GATE1C-REALISTIC",
    )
    print("--- END PIPELINE EXECUTION ---\n")

    # 4. Compile & Print Final Gate 1c Verification Report
    elapsed = time.time() - start_time
    pt = report["privacy_training"]
    pc = report["privacy_composition"]
    psum = report["privacy_summary"]
    fb = report.get("flow_baseline", {})
    hvae_logs = report.get("hvae_stability_log", [])

    print("================================================================================")
    print(" GATE 1c STABILITY & METRICS REPORT (N=5,000, q=0.05)")
    print("================================================================================")
    print("1. HVAE STABILITY PER EPOCH (Point 4):")
    print("  Epoch | Loss    | Recon Loss | KL Loss | Clipped Frac | Mean Norm | Max Norm | Acc DP eps")
    print("  ------+---------+------------+---------+--------------+-----------+----------+-----------")
    for log in hvae_logs:
        print(f"   {log['epoch']:2d}   | {log['loss']:7.4f} | {log['recon_loss']:10.4f} | {log['kl_loss']:7.4f} | {log['fraction_clipped']:11.2%}  | {log['mean_grad_norm']:9.4f} | {log['max_grad_norm']:8.4f} | {log['accumulated_epsilon']:9.4f}")

    print("\n2. FLOW LOSS BASELINE & LATENT SCALE (Point 5):")
    print(f"  Zero-Predictor Expected MSE Loss: {fb.get('zero_predictor_loss', 0.0):.4f}")
    print(f"  Latent Scale (Mean Per-Dim Var):  {fb.get('latent_scale_mean_variance', 0.0):.4f}")
    print(f"  Latent Standardization Applied:   {fb.get('latent_standardized', False)}")

    print("\n3. EPSILON SUMMARY (Point 9):")
    print(f"  Opacus PRV Upper Bound (with Laplace charges):    {psum['prv_upper_with_laplace']:.6f}  (<= {psum['release_target_epsilon']:.4f})")
    print(f"  Opacus PRV Upper Bound (without Laplace charges): {psum['prv_upper_without_laplace']:.6f}")
    print(f"  Formal Guarantee RDP   (with Laplace charges):    {psum['rdp_with_laplace']:.6f}")
    print(f"  Formal Guarantee RDP   (without Laplace charges): {psum['rdp_without_laplace']:.6f}")
    print(f"  Laplace Non-Training Charges (Ledger):            {psum['laplace_meta_charges']:.6f}")

    # Phase 2 C2ST and DCR Evaluator
    print("\n4. PHASE 2 HONEST EVALUATION:")
    c2st = ClassifierTwoSampleTest()
    c2st_res = c2st.evaluate(df_cohort.drop(columns=["patient_id"]), df_syn)
    print(f"  Classifier Two-Sample Test (C2ST) AUC: {c2st_res.get('c2st_auc', 'N/A')}")
    print(f"  C2ST Indistinguishability:            {c2st_res.get('indistinguishability', 'N/A')}")

    dcr_checker = DuplicateMemorizationChecker()
    dcr_res = dcr_checker.evaluate(df_cohort.drop(columns=["patient_id"]), df_syn)
    print(f"  DCR Training Distribution (quantiles): {dcr_res.get('dcr_train_distribution', {})}")
    if "dcr_holdout_distribution" in dcr_res:
        print(f"  DCR Holdout Distribution (quantiles):  {dcr_res.get('dcr_holdout_distribution', {})}")
        print(f"  Train-to-Holdout DCR Ratio:            {dcr_res.get('train_to_holdout_dcr_ratio', 'N/A')}")
        print(f"  Selective Memorization Detected:       {dcr_res.get('selective_memorization_detected', 'N/A')}")

    print(f"\n  Total Elapsed Time: {elapsed:.2f}s")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
