import sys, os, time
from pathlib import Path

# Force unbuffered stdout so prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = REPO_ROOT / "lumisyn" / "framework"
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lumisyn"))
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import pandas as pd

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.privacy.dpsgd import DPSGDWrapper
from acfm_net.generative import ContinuousOTCFM
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    SlicedWassersteinEvaluator,
    ClassifierTwoSampleTestHarness,
    TSTRHarness,
    _encode_for_eval,
)

def load_data(seed: int = 42):
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

def run_detailed_profile():
    print("==========================================================================", flush=True)
    print("=== PROFILING ONE RUN (Adult 30k, Identity Mode, B=512, E=10, eps=1.0) ===", flush=True)
    print("==========================================================================", flush=True)
    df_train, df_holdout = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # Instrument DPSGDWrapper
    orig_step_microbatch = DPSGDWrapper.step_microbatch
    grad_chunk_time = 0.0
    grad_chunk_calls = 0

    def timed_step_microbatch(self, losses, retain_graph=False):
        nonlocal grad_chunk_time, grad_chunk_calls
        torch.cuda.synchronize()
        t0 = time.time()
        res = orig_step_microbatch(self, losses, retain_graph)
        torch.cuda.synchronize()
        grad_chunk_time += (time.time() - t0)
        grad_chunk_calls += 1
        return res

    DPSGDWrapper.step_microbatch = timed_step_microbatch

    # Instrument Flow Matching compute_loss
    orig_compute_loss = ContinuousOTCFM.compute_loss
    forward_loss_time = 0.0
    forward_loss_calls = 0

    def timed_compute_loss(self, *args, **kwargs):
        nonlocal forward_loss_time, forward_loss_calls
        torch.cuda.synchronize()
        t0 = time.time()
        res = orig_compute_loss(self, *args, **kwargs)
        torch.cuda.synchronize()
        forward_loss_time += (time.time() - t0)
        forward_loss_calls += 1
        return res

    ContinuousOTCFM.compute_loss = timed_compute_loss

    # Instrument Flow Matching sample (RK4 generation)
    orig_sample = ContinuousOTCFM.sample
    generation_time = 0.0

    def timed_sample(self, *args, **kwargs):
        nonlocal generation_time
        torch.cuda.synchronize()
        t0 = time.time()
        res = orig_sample(self, *args, **kwargs)
        torch.cuda.synchronize()
        generation_time += (time.time() - t0)
        return res

    ContinuousOTCFM.sample = timed_sample

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
        max_grad_norm_flow=3.0822,
    )

    t_start_pipeline = time.time()
    df_syn, rep = pipe.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
    t_pipeline = time.time() - t_start_pipeline

    stage_durations = {s.get("stage"): s.get("duration_seconds", 0.0) for s in rep.get("stage_reports", [])}
    stage8_dur = stage_durations.get("8", 0.0)
    poisson_and_data_transfer_time = stage8_dur - grad_chunk_time - forward_loss_time

    print("\n--- Measuring Metrics ---", flush=True)
    target_col = "class"
    common = [c for c in df_train.columns if c in df_syn.columns and not c.endswith("_isna")]
    Xr, feat_names = _encode_for_eval(df_train[common])
    Xs, _ = _encode_for_eval(df_syn[common])
    real_train_std = np.std(Xr, axis=0)
    real_train_std = np.where(real_train_std < 1e-6, 1.0, real_train_std)

    # 1. 1-way TVD
    t0 = time.time()
    marg = MarginalErrorEvaluator()
    eval_1w = marg.evaluate_1way(df_train[common], df_syn[common], df_real_holdout=df_holdout[common])
    t_1w = time.time() - t0
    print(f"  1-way TVD: {t_1w:.3f} s (1wTVD={eval_1w['mean_1way_marginal_error_tvd']:.4f})", flush=True)

    # 2. Correlation Error (Frobenius)
    t0 = time.time()
    corr_eval = CorrelationErrorEvaluator()
    eval_corr = corr_eval.evaluate(df_train[common], df_syn[common], df_real_holdout=df_holdout[common])
    t_corr = time.time() - t0
    print(f"  Correlation Frobenius: {t_corr:.3f} s (Frob={eval_corr.get('pearson_frobenius_error', 0):.4f})", flush=True)

    # 3. Standardized Sliced Wasserstein Distance & 1D W1
    t0 = time.time()
    swd_eval = SlicedWassersteinEvaluator(num_projections=100)
    swd_res = swd_eval.evaluate(Xr, Xs, seed=42, standardize=True, real_std=real_train_std, feature_names=feat_names)
    t_swd = time.time() - t0
    print(f"  Standardized SWD & 1D W1: {t_swd:.3f} s (SWD={swd_res['sliced_wasserstein_distance']:.4f})", flush=True)

    # 4. 2-way TVD (Expensive)
    t0 = time.time()
    eval_2w = marg.evaluate_2way(df_train[common], df_syn[common], max_pairs=30, seed=42)
    t_2w = time.time() - t0
    print(f"  2-way TVD: {t_2w:.3f} s (2wTVD={eval_2w['mean_2way_marginal_error_tvd']:.4f})", flush=True)

    # 5. C2ST AUC (RandomForest max_samples=2000)
    t0 = time.time()
    c2st_eval = ClassifierTwoSampleTestHarness()
    c2st_res = c2st_eval.evaluate(df_train[common], df_syn[common], max_samples=2000, seed=42)
    t_c2st = time.time() - t0
    print(f"  C2ST AUC: {t_c2st:.3f} s (AUC={c2st_res.get('c2st_auc_roc', 0.5):.4f})", flush=True)

    # 6. TSTR (default RandomForest on 30k rows)
    t0 = time.time()
    tstr_eval = TSTRHarness()
    tstr_res = tstr_eval.evaluate(df_real_train=df_train[common], df_real_test=df_holdout[common], df_syn=df_syn[common], target_col=target_col, seed=42)
    t_tstr = time.time() - t0
    tstr_auc_val = tstr_res.get("tstr", {}).get("auroc", 0.5)
    print(f"  TSTR (default RF, 30k rows, n_jobs=1): {t_tstr:.3f} s (AUC={tstr_auc_val:.4f})", flush=True)

    # 7. Fast TSTR (RandomForest 5k rows, n_jobs=-1)
    t0 = time.time()
    rng = np.random.default_rng(42)
    sub_syn = df_syn.iloc[rng.permutation(len(df_syn))[:5000]].reset_index(drop=True)
    fast_tstr_res = tstr_eval.evaluate(df_real_train=df_train[common], df_real_test=df_holdout[common], df_syn=sub_syn[common], target_col=target_col, seed=42)
    t_fast_tstr = time.time() - t0
    fast_auc = fast_tstr_res.get("tstr", {}).get("auroc", 0.5)
    print(f"  Fast TSTR (5k subsample, n_jobs=-1): {t_fast_tstr:.3f} s (AUC={fast_auc:.4f}, Speedup={t_tstr/max(0.01, t_fast_tstr):.1f}x)", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("=== FINAL TIME SPLIT REPORT (1 Run: Adult 30k, B=512, E=10) ===", flush=True)
    print("=" * 75, flush=True)
    print(f"1. Total Pipeline Execution:                  {t_pipeline:6.2f} s (100.0%)", flush=True)
    print(f"   - Stage 0-3 (Graph, Feature Encoding):     {stage_durations.get('0-3', 0.0):6.2f} s ({stage_durations.get('0-3', 0.0)/t_pipeline*100:4.1f}%)", flush=True)
    print(f"   - Stage 8 (Flow Training Total):           {stage8_dur:6.2f} s ({stage8_dur/t_pipeline*100:4.1f}%)", flush=True)
    print(f"       * Gradient Chunks (DPSGD autograd):    {grad_chunk_time:6.2f} s ({grad_chunk_time/t_pipeline*100:4.1f}%) [{grad_chunk_calls} steps, {grad_chunk_time/max(1, grad_chunk_calls)*1000:.1f} ms/step]", flush=True)
    print(f"       * Forward Loss computation:            {forward_loss_time:6.2f} s ({forward_loss_time/t_pipeline*100:4.1f}%)", flush=True)
    print(f"       * CPU DataLoader, Poisson & Transfers: {poisson_and_data_transfer_time:6.2f} s ({poisson_and_data_transfer_time/t_pipeline*100:4.1f}%)", flush=True)
    print(f"   - Generative Sampling (RK4 30k rows):      {generation_time:6.2f} s ({generation_time/t_pipeline*100:4.1f}%)", flush=True)
    print("\n2. Metric Evaluation Overhead per seed:", flush=True)
    print(f"   - Cheap Metrics (1-way TVD + Corr + SWD):  {t_1w + t_corr + t_swd:6.2f} s", flush=True)
    print(f"   - Expensive Metrics (2-way TVD+C2ST+TSTR): {t_2w + t_c2st + t_tstr:6.2f} s", flush=True)
    print(f"   - Fast TSTR alternative (5k rows, n_jobs=-1): {t_fast_tstr:6.2f} s", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_detailed_profile()
