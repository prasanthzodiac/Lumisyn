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
    print(" GATE 1b REALISTIC BENCHMARK RUN: N >= 5,000, q <= 0.05")
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
    
    # Mortality outcome correlated with age, creatinine, and systolic BP
    logit = -3.5 + 0.03 * (age - 60) + 0.25 * (creatinine - 1.0) + 0.01 * (systolic - 120)
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

    epochs = 4  # Stage 8 flow epochs
    # Note: Stage 4 representation epochs = max(5, epochs // 2) = 5 epochs
    print(f"[Config] Stage 4 Representation Epochs: 5 | Stage 8 Flow Epochs: {epochs}")

    # 3. Fit & Generate
    print("\n--- BEGIN PIPELINE EXECUTION ---")
    df_syn, report = pipeline.fit_generate(
        df_raw=df_cohort,
        num_samples=500,
        target_col="mortality",
        epochs=epochs,
        protocol_id="IRB-GATE1B-REALISTIC",
    )
    print("--- END PIPELINE EXECUTION ---\n")

    # 4. Compile & Print Final Gate 1b Verification Report
    elapsed = time.time() - start_time
    pt = report["privacy_training"]
    pc = report["privacy_composition"]
    gov = report["governance_certificate"]

    print("================================================================================")
    print(" GATE 1b REALISTIC RUN RESULTS & PRIVACY ACCOUNTING SUMMARY")
    print("================================================================================")
    print(f" Cohort Size N:               {N}")
    print(f" Expected Batch Size:         {pt['expected_batch_size']}")
    print(f" Subsampling Rate q:          {pt['sample_rate']:.6f} (q <= 0.05: {pt['sample_rate'] <= 0.05})")
    print(f" Noise Multiplier sigma:      {pt['noise_multiplier']:.6f}")
    print(f" Max Gradient Norm C:         {pt['max_grad_norm']}")
    print(f" Stage 4 (HVAE) Steps:        {pt['hvae_steps']}")
    print(f" Stage 8 (Flow) Steps:        {pt['flow_steps']}")
    print(f" Total DP Steps:              {pt['steps']}")
    print(f" Training Delta:              {pt['target_delta']:.2e}")
    print("--------------------------------------------------------------------------------")
    print(f" Training DP-SGD Epsilon:     {pt['accounted_epsilon']:.6f}")
    print(f" Formal Guarantee (RDP eps):  {pt['formal_guarantee_rdp_epsilon']:.6f}")
    print(f" Opacus PRV Upper Bound eps:  {pt['prv_epsilon_upper']:.6f}")
    print(f" Laplace Meta Charges eps:    {pc['meta_charges_epsilon']:.6f}")
    print(f" Total Composed Epsilon:      {pc['epsilon']:.6f} <= Target {pc['target_epsilon']:.4f}")
    print(f" Verification Status:         {report['privacy_status']}")
    print(f" Total Elapsed Time:          {elapsed:.2f}s")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
