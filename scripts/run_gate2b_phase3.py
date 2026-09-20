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
from acfm_net.privacy.checkpoint_audit import audit_released_checkpoint
from acfm_net.evaluation.harness import run_phase3_evaluation


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
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
    print(" GATE 2b FIXES & PHASE 3 COMPREHENSIVE BENCHMARK RUN (STOP AT GATE 3)")
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
    
    # Missingness: ~5% in BMI and ~8% in glucose
    bmi[np.random.rand(N) < 0.05] = np.nan
    glucose[np.random.rand(N) < 0.08] = np.nan

    # Mortality outcome correlated with age, creatinine, systolic BP
    logit = -3.5 + 0.03 * (np.nan_to_num(age, nan=62.5) - 60) + 0.25 * (np.nan_to_num(creatinine, nan=1.0) - 1.0) + 0.01 * (np.nan_to_num(systolic, nan=130) - 120)
    prob_mort = 1.0 / (1.0 + np.exp(-logit))
    mortality = (np.random.rand(N) < prob_mort).astype(int)

    df_full = pd.DataFrame({
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

    # Strict patient deduplication check
    assert df_full["patient_id"].nunique() == N, "All patient IDs must be strictly unique (1 row per patient)"
    print(f"[Data Prep] Patient uniqueness verified: {df_full['patient_id'].nunique()} unique IDs out of {len(df_full)} rows.")

    # 2. Fixed Disjoint Splits for Phase 3 Evaluation
    # 50% Train (N=2,500), 50% Disjoint Holdout (N=2,500)
    perm = np.random.RandomState(seed).permutation(N)
    n_train = 2500
    train_idx = perm[:n_train]
    holdout_idx = perm[n_train:]

    df_train = df_full.iloc[train_idx].drop(columns=["patient_id"]).reset_index(drop=True)
    df_holdout = df_full.iloc[holdout_idx].drop(columns=["patient_id"]).reset_index(drop=True)

    # Holdout is further partitioned into negative membership set and test evaluation set
    df_eval_test = df_holdout.iloc[:1250].reset_index(drop=True)
    df_eval_holdout = df_holdout.iloc[1250:].reset_index(drop=True)

    print(f"[Splits] Train: {len(df_train)} rows | Disjoint Holdout (MIA Negatives): {len(df_eval_holdout)} rows | Test Set: {len(df_eval_test)} rows")
    print(f"[Data Prep] Train Missing cells: bmi={df_train['bmi'].isna().sum()}, glucose={df_train['glucose'].isna().sum()}")

    # 3. Configure Pipeline for DP-SGD (q <= 0.05 on N_train=2500)
    # batch_size = 125 => q = 125 / 2500 = 0.05
    batch_size = 125
    q_target = batch_size / n_train
    print(f"\n[Config] Setting N_train={n_train}, batch_size={batch_size} => expected subsampling rate q={q_target:.4f} <= 0.05")

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

    # 4. Fit & Generate
    print("\n--- BEGIN PIPELINE EXECUTION ---")
    df_syn, report = pipeline.fit_generate(
        df_raw=df_train,
        num_samples=n_train,
        target_col="mortality",
        epochs=epochs,
    )
    print("--- PIPELINE EXECUTION FINISHED ---\n")

    # 5. Checkpoint-Audit Test (Gate 2b Item 2)
    print("\n====================== CHECKPOINT-AUDIT VERIFICATION ======================")
    checkpoint = pipeline.released_checkpoint
    audit_res = audit_released_checkpoint(checkpoint)
    summary = audit_res["summary"]
    print(f" Total items audited:            {summary['total_items_audited']}")
    print(f" Total DP-SGD trained weights:   {summary['total_dp_trained_parameters']:,} parameters")
    print(f" Total Ledger-charged elements:  {summary['total_ledger_charged_elements']} elements")
    print(f" Total Public entries/constants: {summary['total_public_entries']}")
    print(f" Unbudgeted private violations:  {summary['unbudgeted_violations_count']}")
    print(f" Checkpoint Audit Status:        {'PASSED' if summary['checkpoint_audit_passed'] else 'FAILED'}")
    assert summary["checkpoint_audit_passed"] is True, "Checkpoint audit must pass for valid release"
    print("============================================================================\n")

    # 6. Verify Missing Values Restoration (Gate 2b Item 7)
    print("\n====================== MISSING VALUE RESTORATION AUDIT ======================")
    syn_missing_bmi = int(df_syn["bmi"].isna().sum()) if "bmi" in df_syn.columns else 0
    syn_missing_glucose = int(df_syn["glucose"].isna().sum()) if "glucose" in df_syn.columns else 0
    print(f" Synthetic Missing cells restored: bmi={syn_missing_bmi}/{len(df_syn)} ({syn_missing_bmi/len(df_syn):.2%}), glucose={syn_missing_glucose}/{len(df_syn)} ({syn_missing_glucose/len(df_syn):.2%})")
    
    # Check that no raw 0.0 values appear for positive medical features
    if "systolic" in df_syn.columns:
        zeros_sys = int((df_syn["systolic"] == 0.0).sum())
        min_sys = float(df_syn["systolic"].dropna().min())
        print(f" Systolic BP: zeros count={zeros_sys}, minimum observed={min_sys:.1f} mmHg (raw 0.0 absent: {zeros_sys == 0})")
        assert zeros_sys == 0, "No raw 0.0 fill values should reach the output"
    print("============================================================================\n")

    # 7. PHASE 3 EVALUATION HARNESS EXECUTION
    schema_bounds = {
        "age": (18.0, 95.0),
        "bmi": (14.0, 55.0),
        "systolic": (80.0, 210.0),
        "diastolic": (50.0, 130.0),
        "glucose": (40.0, 500.0),
        "creatinine": (0.4, 12.0),
    }

    eval_results = run_phase3_evaluation(
        df_train=df_train,
        df_holdout=df_eval_holdout,
        df_test=df_eval_test,
        df_syn=df_syn,
        target_col="mortality",
        schema_bounds=schema_bounds,
        seed=seed,
    )

    total_runtime = round(time.time() - start_time, 2)
    print("================================================================================")
    print(f" GATE 3 COMPLETED SUCCESSFULLY IN {total_runtime}s")
    print(" STOPPED AT GATE 3 AS REQUIRED.")
    print("================================================================================")


if __name__ == "__main__":
    main()
