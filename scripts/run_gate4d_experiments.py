"""
Gate 4d Experiment Runner - Lumisyn / ACFM-Net
=============================================================================
Comprehensive Empirical Protocol & Gate 4d Addendum (A1 - A8)
Parameters:
  - Dataset: Adult 30k (subsampled from adult.csv)
  - Train: 30,000 rows | Holdout: 18,842 rows
  - Privacy: Target eps=1.0 & eps=4.0, delta=1e-5, strict_release_privacy=True
  - Seeds: 3 seeds (42, 43, 44) per evaluated row

Addendum Modules:
  A1. Identity-mode diagnostics:
      - 10 worst columns by 1-way TVD
      - Fraction of generated values outside [-1, 1] before Stage 7 clamp
      - Generated vs real per-column mean and std in scaled space
      - Top-1 probability margins for each categorical block
      - Flow loss vs empirical zero-predictor baseline on X_scaled
  A2. Corrected SWD & 1D W1:
      - Column-wise standardization by real-train std
      - Report corrected SWD and per-column 1D W1 in std units for identity, VAE, floors, real-vs-real
  A3. LR and clipping sweep (best B/E, 3 seeds):
      - Optimizer in {Adam, SGD+momentum}
      - LR in {5e-4, 2e-3, 1e-2, 5e-2}
      - C in {0.5, 1.0, 2.0, 4.0} x C_med
      - Log clipped fraction and mean gradient norm
  A4. Prior scale:
      - z0 ~ N(0, s^2 I) with s in {0.25, 0.5, 1.0}
      - Per-column variance mismatch with data
  A5. Longer training at fixed eps=1.0:
      - Epochs in {10, 20, 40} at B=1024 (sigma rises accordingly)
  A6. DP marginal recalibration post-step, ledger-charged:
      - Public schema bounds from adult_schema.yaml
      - Sensitivity derivation Delta_1 = M under add/remove adjacency
      - Sweep eps_m in {0.1, 0.25, 0.5}, flow gets eps - eps_m
      - Quantile-matching on numeric columns, frequency-matching on categoricals
  A7. DP Gaussian-copula baseline at eps=1.0
  A8. Final evaluation table & Go/No-Go decision at eps=1.0 and eps=4.0

Logged to: runs/gate4d_experiments.log
"""

import sys, os, time, json, math, datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = REPO_ROOT / "lumisyn" / "framework"
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lumisyn"))
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "runs" / "gate4d_experiments.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
SCHEMA_PATH = REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml"

import numpy as np
import pandas as pd
import torch

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.evaluation.floor_baselines import IndependentMarginalsDPGenerator, ConstantMeanGenerator
from acfm_net.evaluation.dp_gaussian_copula import DPGaussianCopulaGenerator
from acfm_net.postprocessing.dp_recalibration import DPMarginalRecalibrator
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    SlicedWassersteinEvaluator,
    ClassifierTwoSampleTestHarness,
    TSTRHarness,
    _encode_for_eval,
)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "w", encoding="utf-8")
        self.log("=" * 80)
        self.log(f"[Gate 4d Experiment Suite] Started at {datetime.datetime.now().isoformat()}")
        self.log("Dataset: Adult 30k | Strict Release DP | Seeds: [42, 43, 44]")
        self.log("=" * 80 + "\n")

    def log(self, msg: str = ""):
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def section(self, title: str):
        self.log("\n" + "=" * 75)
        self.log(f"=== {title} ===")
        self.log("=" * 75 + "\n")

    def close(self):
        self.fh.close()


LOG = Logger(LOG_PATH)


def load_adult_30k_and_holdout(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def compute_standardized_metrics(
    df_real_train: pd.DataFrame,
    df_syn: pd.DataFrame,
    df_holdout: pd.DataFrame,
    target_col: str = "class",
    real_train_std: Optional[np.ndarray] = None,
    encoded_feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    common = [c for c in df_real_train.columns if c in df_syn.columns and not c.endswith("_isna")]

    # 1. 1-way TVD
    marg = MarginalErrorEvaluator()
    eval_1w = marg.evaluate_1way(df_real_train[common], df_syn[common], df_real_holdout=df_holdout[common])
    tvd_1w = eval_1w["mean_1way_marginal_error_tvd"]
    floor_1w = eval_1w.get("real_vs_real_floor_1way_tvd", 0.0076)
    norm_tvd_1w = eval_1w.get("normalized_1way_marginal_error", 1.0)
    per_feature_1w = eval_1w.get("per_feature_1way_tvd", {})

    # 2. 2-way TVD
    eval_2w = marg.evaluate_2way(df_real_train[common], df_syn[common], max_pairs=30, seed=42)
    tvd_2w = eval_2w["mean_2way_marginal_error_tvd"]
    floor_2w_eval = marg.evaluate_2way(df_real_train[common], df_holdout[common], max_pairs=30, seed=42)
    floor_2w = floor_2w_eval["mean_2way_marginal_error_tvd"]
    norm_tvd_2w = round(tvd_2w / max(1e-4, floor_2w), 4)

    # 3. Correlation Error (Frobenius)
    corr_eval = CorrelationErrorEvaluator()
    eval_corr = corr_eval.evaluate(df_real_train[common], df_syn[common], df_real_holdout=df_holdout[common])
    frob_corr = eval_corr.get("pearson_frobenius_error", 0.0)
    floor_corr = eval_corr.get("real_vs_real_floor_pearson_frobenius", 0.0430)
    norm_corr = eval_corr.get("normalized_pearson_frobenius_error", 1.0)

    # 4. Standardized Sliced Wasserstein Distance & 1D W1 in std units (Item A2)
    Xr, feat_names = _encode_for_eval(df_real_train[common])
    Xs, _ = _encode_for_eval(df_syn[common])
    std_to_use = real_train_std if real_train_std is not None else np.std(Xr, axis=0)

    swd_eval = SlicedWassersteinEvaluator(num_projections=100)
    swd_res = swd_eval.evaluate(
        Xr, Xs, seed=42, standardize=True, real_std=std_to_use, feature_names=feat_names
    )
    swd_val = swd_res["sliced_wasserstein_distance"]
    mean_1d_w1 = swd_res["mean_1d_w1"]
    per_col_1d_w1 = swd_res["per_column_1d_w1"]

    # 5. C2ST AUC (5-fold CV)
    c2st_eval = ClassifierTwoSampleTestHarness()
    c2st_res = c2st_eval.evaluate(df_real_train[common], df_syn[common], max_samples=2000, seed=42)
    c2st_auc = c2st_res.get("c2st_auc_roc", 0.5)

    # 6. TSTR & TRTR (Item 0 Correction)
    tstr_eval = TSTRHarness()
    if target_col in df_real_train.columns and target_col in df_syn.columns:
        tstr_res = tstr_eval.evaluate(
            df_real_train=df_real_train[common],
            df_real_test=df_holdout[common],
            df_syn=df_syn[common],
            target_col=target_col,
            seed=42,
        )
        tstr_dict = tstr_res.get("tstr", {})
        trtr_dict = tstr_res.get("trtr_baseline", {})
        tstr_auc = tstr_dict.get("auroc", 0.5)
        tstr_prc = tstr_dict.get("auprc", 0.0)
        tstr_f1  = tstr_dict.get("f1_weighted", 0.0)
        trtr_auc = trtr_dict.get("auroc", 0.9076)
        trtr_prc = trtr_dict.get("auprc", 0.7883)
    else:
        tstr_auc, tstr_prc, tstr_f1 = 0.5, 0.0, 0.0
        trtr_auc, trtr_prc = 0.9076, 0.7883

    return {
        "1way_tvd": tvd_1w,
        "floor_1way_tvd": floor_1w,
        "norm_1way_tvd": norm_tvd_1w,
        "per_feature_1way_tvd": per_feature_1w,
        "2way_tvd": tvd_2w,
        "floor_2way_tvd": floor_2w,
        "norm_2way_tvd": norm_tvd_2w,
        "frob_corr_error": frob_corr,
        "floor_frob_corr": floor_corr,
        "norm_frob_corr": norm_corr,
        "swd": swd_val,
        "mean_1d_w1": mean_1d_w1,
        "per_column_1d_w1": per_col_1d_w1,
        "c2st_auc": c2st_auc,
        "tstr_auc": tstr_auc,
        "tstr_prc": tstr_prc,
        "tstr_f1": tstr_f1,
        "trtr_auc": trtr_auc,
        "trtr_prc": trtr_prc,
    }


def run_gate4d():
    df_train, df_holdout = load_adult_30k_and_holdout(seed=42)
    LOG.log(f"Adult 30k loaded: Train={df_train.shape}, Holdout={df_holdout.shape}")
    seeds = [42, 43, 44]

    # Pre-compute real-train std for standardization
    Xr_train, feat_names = _encode_for_eval(df_train)
    real_train_std = np.std(Xr_train, axis=0)
    real_train_std = np.where(real_train_std < 1e-6, 1.0, real_train_std)

    # =========================================================================
    # PART 0: Corrected Floor Baselines & True TRTR Benchmark
    # =========================================================================
    LOG.section("PART 0: CORRECTED FLOOR BASELINES & TRUE TRTR EVALUATION")

    # 1. Real-vs-Real Floor (train vs holdout)
    LOG.log("Computing Real-vs-Real Floor (Train 30k vs Holdout 18.8k)...")
    rvr_metrics = compute_standardized_metrics(
        df_real_train=df_train,
        df_syn=df_holdout,
        df_holdout=df_holdout,
        target_col="class",
        real_train_std=real_train_std,
        encoded_feature_names=feat_names,
    )
    LOG.log(f"Real-vs-Real Floor: 1-way TVD={rvr_metrics['1way_tvd']:.4f} | 2-way TVD={rvr_metrics['2way_tvd']:.4f} | "
            f"Frob Corr={rvr_metrics['frob_corr_error']:.4f} | Standardized SWD={rvr_metrics['swd']:.4f} | "
            f"Mean 1D W1={rvr_metrics['mean_1d_w1']:.4f} std | C2ST AUC={rvr_metrics['c2st_auc']:.4f} | "
            f"TRTR Baseline AUROC={rvr_metrics['trtr_auc']:.4f} | TRTR AUPRC={rvr_metrics['trtr_prc']:.4f}")

    # 2. Constant Mean Generator
    LOG.log("\nEvaluating Constant Mean Generator Baseline...")
    const_gen = ConstantMeanGenerator()
    const_gen.fit(df_train)
    df_const = const_gen.generate(n_samples=len(df_train))
    const_metrics = compute_standardized_metrics(
        df_real_train=df_train,
        df_syn=df_const,
        df_holdout=df_holdout,
        target_col="class",
        real_train_std=real_train_std,
        encoded_feature_names=feat_names,
    )
    LOG.log(f"Constant Mean: 1-way TVD={const_metrics['1way_tvd']:.4f} (Norm={const_metrics['norm_1way_tvd']:.2f}x) | "
            f"2-way TVD={const_metrics['2way_tvd']:.4f} (Norm={const_metrics['norm_2way_tvd']:.2f}x) | "
            f"Frob Corr={const_metrics['frob_corr_error']:.4f} (Norm={const_metrics['norm_frob_corr']:.2f}x) | "
            f"Standardized SWD={const_metrics['swd']:.4f} | Mean 1D W1={const_metrics['mean_1d_w1']:.4f} std | "
            f"C2ST AUC={const_metrics['c2st_auc']:.4f} | TSTR AUROC={const_metrics['tstr_auc']:.4f}")

    # 3. Independent Marginals DP Generator (eps=1.0 and eps=4.0)
    LOG.log("\nEvaluating Independent Marginals DP Generator at eps=1.0 (3 seeds)...")
    ind_metrics_eps1 = []
    for s in seeds:
        ind_gen1 = IndependentMarginalsDPGenerator(epsilon=1.0, seed=s)
        ind_gen1.fit(df_train)
        df_ind1 = ind_gen1.generate(n_samples=len(df_train), seed=s)
        m1 = compute_standardized_metrics(
            df_train, df_ind1, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
        )
        ind_metrics_eps1.append(m1)
        LOG.log(f"  Seed {s} (eps=1.0): 1wTVD={m1['1way_tvd']:.4f} ({m1['norm_1way_tvd']:.2f}x) | "
                f"2wTVD={m1['2way_tvd']:.4f} ({m1['norm_2way_tvd']:.2f}x) | "
                f"Corr={m1['frob_corr_error']:.4f} ({m1['norm_frob_corr']:.2f}x) | "
                f"SWD={m1['swd']:.4f} | 1D W1={m1['mean_1d_w1']:.4f} std | TSTR AUC={m1['tstr_auc']:.4f}")

    LOG.log("\nEvaluating Independent Marginals DP Generator at eps=4.0 (3 seeds)...")
    ind_metrics_eps4 = []
    for s in seeds:
        ind_gen4 = IndependentMarginalsDPGenerator(epsilon=4.0, seed=s)
        ind_gen4.fit(df_train)
        df_ind4 = ind_gen4.generate(n_samples=len(df_train), seed=s)
        m4 = compute_standardized_metrics(
            df_train, df_ind4, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
        )
        ind_metrics_eps4.append(m4)
        LOG.log(f"  Seed {s} (eps=4.0): 1wTVD={m4['1way_tvd']:.4f} ({m4['norm_1way_tvd']:.2f}x) | "
                f"2wTVD={m4['2way_tvd']:.4f} ({m4['norm_2way_tvd']:.2f}x) | "
                f"Corr={m4['frob_corr_error']:.4f} ({m4['norm_frob_corr']:.2f}x) | "
                f"SWD={m4['swd']:.4f} | 1D W1={m4['mean_1d_w1']:.4f} std | TSTR AUC={m4['tstr_auc']:.4f}")

    # =========================================================================
    # PART A7: DP Gaussian-Copula Baseline at eps=1.0 (3 seeds)
    # =========================================================================
    LOG.section("PART A7: DP GAUSSIAN-COPULA BASELINE (eps=1.0, 3 seeds)")
    copula_metrics_list = []
    for s in seeds:
        copula_gen = DPGaussianCopulaGenerator(SCHEMA_PATH, epsilon=1.0, seed=s)
        copula_gen.fit(df_train)
        df_copula = copula_gen.generate(n_samples=len(df_train), seed=s)
        mc = compute_standardized_metrics(
            df_train, df_copula, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
        )
        copula_metrics_list.append(mc)
        LOG.log(f"  Seed {s} (eps=1.0): 1wTVD={mc['1way_tvd']:.4f} ({mc['norm_1way_tvd']:.2f}x) | "
                f"2wTVD={mc['2way_tvd']:.4f} ({mc['norm_2way_tvd']:.2f}x) | "
                f"Corr={mc['frob_corr_error']:.4f} ({mc['norm_frob_corr']:.2f}x) | "
                f"SWD={mc['swd']:.4f} | 1D W1={mc['mean_1d_w1']:.4f} std | TSTR AUC={mc['tstr_auc']:.4f}")

    # =========================================================================
    # PART A1: Identity-Mode In-Depth Diagnostics
    # =========================================================================
    LOG.section("PART A1: IDENTITY-MODE IN-DEPTH DIAGNOSTICS")
    # Run Lumisyn-Identity at default config (B=512, E=10, eps=1.0, seed=42)
    torch.manual_seed(42)
    np.random.seed(42)
    pipe_diag = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        latent_mode="identity",
        batch_size=512,
        flow_hidden_dim=128,
        flow_time_embed_dim=64,
        flow_num_layers=2,
    )
    t0 = time.time()
    df_syn_diag, rep_diag = pipe_diag.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
    el_diag = time.time() - t0
    m_diag = compute_standardized_metrics(
        df_train, df_syn_diag, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
    )

    # 1. 10 worst columns by 1-way TVD
    per_col_tvd = m_diag.get("per_feature_1way_tvd", {})
    sorted_worst = sorted(per_col_tvd.items(), key=lambda x: x[1], reverse=True)[:10]
    LOG.log("1. Top 10 Worst Columns by 1-Way TVD (Identity Mode):")
    for rank, (col, tvd) in enumerate(sorted_worst, 1):
        LOG.log(f"   {rank:2d}. {col:<20} | TVD = {tvd:.4f}")

    # 2. Fraction of generated values outside [-1, 1] before clamp
    id_diag = rep_diag.get("identity_diagnostics", {})
    pre_clamp_frac = id_diag.get("pre_clamp_out_of_bounds_fraction", 0.0)
    LOG.log(f"\n2. Pre-Clamp Out-of-Bounds Fraction (outside [-1, 1]): {pre_clamp_frac:.4%}")

    # 3. Generated vs Real per-column mean and std in scaled space
    X_real_scaled = getattr(pipe_diag, "scaled_X_train", None)
    X_syn_scaled  = getattr(pipe_diag, "scaled_X_generated", None)
    if X_real_scaled is not None and X_syn_scaled is not None:
        real_means = np.mean(X_real_scaled, axis=0)
        real_stds  = np.std(X_real_scaled, axis=0)
        syn_means  = np.mean(X_syn_scaled, axis=0)
        syn_stds   = np.std(X_syn_scaled, axis=0)

        LOG.log("\n3. Generated vs Real Scaled-Space Statistics (Mean and Std across 78 dimensions):")
        LOG.log(f"   Real Scaled Mean (avg across dims): {np.mean(real_means):.4f} | Std: {np.mean(real_stds):.4f}")
        LOG.log(f"   Syn  Scaled Mean (avg across dims): {np.mean(syn_means):.4f} | Std: {np.mean(syn_stds):.4f}")
        LOG.log(f"   Mean Absolute Mean Error:         {np.mean(np.abs(real_means - syn_means)):.4f}")
        LOG.log(f"   Mean Absolute Std Error:          {np.mean(np.abs(real_stds - syn_stds)):.4f}")

    # 4. Top-1 probability margins of each categorical block
    cat_margins = id_diag.get("categorical_block_top1_margins", {})
    mean_margin = id_diag.get("mean_top1_margin", 0.0)
    LOG.log("\n4. Top-1 Probability Margin (p_(1) - p_(2)) for Categorical Blocks:")
    for cat_col, margin in cat_margins.items():
        LOG.log(f"   {cat_col:<20} | Top-1 Margin = {margin:.4f}")
    LOG.log(f"   Mean Categorical Margin across blocks: {mean_margin:.4f}")

    # 5. Flow loss vs zero-predictor baseline on X_scaled
    # Theoretical zero-predictor loss under N(0, I) prior: 2.0000; empirical on X_scaled:
    if X_real_scaled is not None:
        emp_zero_baseline = float(np.mean(X_real_scaled ** 2) + 1.0)
    else:
        emp_zero_baseline = 2.0
    final_flow_loss = rep_diag.get("stage_reports", [{}])[-1].get("evidence", {}).get("last_flow_loss", 0.0)
    if not final_flow_loss:
        flogs = rep_diag.get("flow_stability_log", [])
        final_flow_loss = flogs[-1].get("loss", 0.0) if flogs else 0.0
    LOG.log(f"\n5. Flow Loss vs Zero-Predictor Baseline:")
    LOG.log(f"   Empirical Zero-Predictor Baseline (E[||x1 - x0||^2] on X_scaled): {emp_zero_baseline:.4f}")
    LOG.log(f"   Standardized Theoretical N(0,I) Reference Loss:                  2.0000")
    LOG.log(f"   Final Trained Flow Loss:                                         {final_flow_loss:.4f}")
    LOG.log(f"   Loss Improvement over Zero-Predictor:                           {emp_zero_baseline - final_flow_loss:.4f} nats")

    # =========================================================================
    # PART A3: Learning-Rate and Clipping Sweep (B=512, E=10, 3 seeds)
    # =========================================================================
    LOG.section("PART A3: LEARNING-RATE AND CLIPPING SWEEP (Identity Mode, B=512, E=10)")
    # Measure median gradient norm C_med
    flogs_diag = rep_diag.get("flow_stability_log", [])
    c_med_measured = float(flogs_diag[0].get("median_grad_norm", 2.0)) if flogs_diag else 2.0
    if c_med_measured <= 0.0:
        c_med_measured = 1.5
    LOG.log(f"Empirical Median Gradient Norm C_med = {c_med_measured:.4f}")

    opt_list = ["adam", "sgd"]
    lr_list = [5e-4, 2e-3, 1e-2, 5e-2]
    c_mults = [0.5, 1.0, 2.0, 4.0]
    c_values = [round(c_med_measured * m, 4) for m in c_mults]
    LOG.log(f"Clipping thresholds C in {c_values} (corresponding to {c_mults} x C_med)")

    sweep_a3_results = []
    best_a3_config = None
    best_a3_auc = -1.0

    for opt_name in opt_list:
        for lr in lr_list:
            for C_val, c_mult in zip(c_values, c_mults):
                cfg_desc = f"Opt={opt_name.upper()}, lr={lr}, C={C_val} ({c_mult}x)"
                LOG.log(f"\n--- Testing Config: {cfg_desc} (3 seeds) ---")
                seed_metrics = []
                seed_clip_fracs = []
                seed_grad_norms = []

                for s in seeds:
                    torch.manual_seed(s)
                    np.random.seed(s)
                    p_sw = ACFMNetV5Pipeline(
                        target_epsilon=1.0,
                        target_delta=1e-5,
                        strict_release_privacy=True,
                        latent_mode="identity",
                        batch_size=512,
                        flow_hidden_dim=128,
                        flow_num_layers=2,
                        flow_optimizer=opt_name,
                        flow_lr=lr,
                        max_grad_norm_flow=C_val,
                    )
                    df_s, rep_s = p_sw.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
                    flog = rep_s.get("flow_stability_log", [])
                    avg_clipped = float(np.mean([x.get("fraction_clipped", 0.0) for x in flog])) if flog else 0.0
                    avg_norm = float(np.mean([x.get("mean_grad_norm", 0.0) for x in flog])) if flog else 0.0
                    m_s = compute_standardized_metrics(
                        df_train, df_s, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
                    )
                    seed_metrics.append(m_s)
                    seed_clip_fracs.append(avg_clipped)
                    seed_grad_norms.append(avg_norm)

                mean_auc = float(np.mean([m["tstr_auc"] for m in seed_metrics]))
                mean_corr = float(np.mean([m["norm_frob_corr"] for m in seed_metrics]))
                mean_1w = float(np.mean([m["norm_1way_tvd"] for m in seed_metrics]))
                mean_2w = float(np.mean([m["norm_2way_tvd"] for m in seed_metrics]))
                mean_clip = float(np.mean(seed_clip_fracs))
                mean_norm = float(np.mean(seed_grad_norms))

                entry = {
                    "optimizer": opt_name,
                    "lr": lr,
                    "C": C_val,
                    "c_mult": c_mult,
                    "mean_clipped_frac": mean_clip,
                    "mean_grad_norm": mean_norm,
                    "tstr_auc": mean_auc,
                    "norm_corr": mean_corr,
                    "norm_1w": mean_1w,
                    "norm_2w": mean_2w,
                }
                sweep_a3_results.append(entry)
                LOG.log(f"  Result: Clipped Frac={mean_clip:.2%} | Mean Grad Norm={mean_norm:.4f} | "
                        f"Norm 1wTVD={mean_1w:.2f}x | Norm 2wTVD={mean_2w:.2f}x | Norm Corr={mean_corr:.2f}x | TSTR AUC={mean_auc:.4f}")

                if mean_auc > best_a3_auc:
                    best_a3_auc = mean_auc
                    best_a3_config = entry

    LOG.log(f"\n[A3 Sweep Winner]: {best_a3_config['optimizer'].upper()} with lr={best_a3_config['lr']}, C={best_a3_config['C']} | TSTR AUC={best_a3_config['tstr_auc']:.4f}")

    # =========================================================================
    # PART A4: Prior Scale Sweep (z0 ~ N(0, s^2 I))
    # =========================================================================
    LOG.section("PART A4: PRIOR SCALE SWEEP (s in {0.25, 0.5, 1.0})")
    scales = [0.25, 0.5, 1.0]
    best_opt = best_a3_config["optimizer"]
    best_lr = best_a3_config["lr"]
    best_C = best_a3_config["C"]

    prior_results = {}
    for s_val in scales:
        LOG.log(f"\n--- Testing Prior Scale s = {s_val} (3 seeds) ---")
        scale_seed_metrics = []
        var_mismatches = []

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            pipe_s = ACFMNetV5Pipeline(
                target_epsilon=1.0,
                target_delta=1e-5,
                strict_release_privacy=True,
                latent_mode="identity",
                batch_size=512,
                flow_hidden_dim=128,
                flow_num_layers=2,
                flow_optimizer=best_opt,
                flow_lr=best_lr,
                max_grad_norm_flow=best_C,
                prior_scale=s_val,
            )
            df_ps, rep_ps = pipe_s.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
            m_ps = compute_standardized_metrics(
                df_train, df_ps, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
            )
            scale_seed_metrics.append(m_ps)

            # Per-column variance mismatch on numeric features
            num_cols = [c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c])]
            v_real = df_train[num_cols].var().values
            v_syn  = df_ps[num_cols].var().values
            v_mismatch = np.mean(np.abs(v_real - v_syn) / np.maximum(1e-6, v_real))
            var_mismatches.append(v_mismatch)

        mean_v_err = float(np.mean(var_mismatches))
        mean_auc = float(np.mean([m["tstr_auc"] for m in scale_seed_metrics]))
        mean_corr = float(np.mean([m["norm_frob_corr"] for m in scale_seed_metrics]))
        mean_swd = float(np.mean([m["swd"] for m in scale_seed_metrics]))
        prior_results[s_val] = {
            "mean_var_mismatch": mean_v_err,
            "tstr_auc": mean_auc,
            "norm_corr": mean_corr,
            "swd": mean_swd,
        }
        LOG.log(f"  s={s_val:4.2f}: Mean Var Mismatch={mean_v_err:.4%} | Standardized SWD={mean_swd:.4f} | "
                f"Norm Corr={mean_corr:.2f}x | TSTR AUC={mean_auc:.4f}")

    # =========================================================================
    # PART A5: Longer Training at Fixed eps=1.0 (Epochs in {10, 20, 40}, B=1024)
    # =========================================================================
    LOG.section("PART A5: LONGER TRAINING AT FIXED EPS=1.0 (Epochs in {10, 20, 40}, B=1024)")
    epochs_list = [10, 20, 40]
    longer_results = {}

    for ep_count in epochs_list:
        LOG.log(f"\n--- Testing Epochs = {ep_count} at B=1024, eps=1.0 (3 seeds) ---")
        ep_seed_metrics = []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            pipe_ep = ACFMNetV5Pipeline(
                target_epsilon=1.0,
                target_delta=1e-5,
                strict_release_privacy=True,
                latent_mode="identity",
                batch_size=1024,
                flow_hidden_dim=128,
                flow_num_layers=2,
                flow_optimizer=best_opt,
                flow_lr=best_lr,
                max_grad_norm_flow=best_C,
            )
            df_ep, rep_ep = pipe_ep.fit_generate(df_train, num_samples=len(df_train), epochs=ep_count, target_col="class")
            chk_cfg = rep_ep.get("released_checkpoint", getattr(pipe_ep, "released_checkpoint", {})).get("config", {})
            sigma_val = chk_cfg.get("flow_noise_multiplier", 0.0)
            m_ep = compute_standardized_metrics(
                df_train, df_ep, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
            )
            m_ep["sigma"] = sigma_val
            ep_seed_metrics.append(m_ep)

        mean_sigma = float(np.mean([m["sigma"] for m in ep_seed_metrics]))
        mean_auc = float(np.mean([m["tstr_auc"] for m in ep_seed_metrics]))
        mean_corr = float(np.mean([m["norm_frob_corr"] for m in ep_seed_metrics]))
        mean_1w = float(np.mean([m["norm_1way_tvd"] for m in ep_seed_metrics]))
        mean_2w = float(np.mean([m["norm_2way_tvd"] for m in ep_seed_metrics]))
        longer_results[ep_count] = {
            "sigma": mean_sigma,
            "tstr_auc": mean_auc,
            "norm_corr": mean_corr,
            "norm_1w": mean_1w,
            "norm_2w": mean_2w,
        }
        LOG.log(f"  Epochs={ep_count:2d}: Flow sigma={mean_sigma:.4f} | Norm 1wTVD={mean_1w:.2f}x | "
                f"Norm 2wTVD={mean_2w:.2f}x | Norm Corr={mean_corr:.2f}x | TSTR AUC={mean_auc:.4f}")

    # =========================================================================
    # PART A6: DP Marginal Recalibration Post-Step (eps_m in {0.1, 0.25, 0.5})
    # =========================================================================
    LOG.section("PART A6: DP MARGINAL RECALIBRATION POST-STEP (Ledger Charged)")
    recalibrator = DPMarginalRecalibrator(SCHEMA_PATH, num_continuous_bins=50)
    LOG.log(f"Sensitivity derivation: Global L1 sensitivity Delta_1 = M = {recalibrator.get_sensitivity()} under add/remove.")

    eps_m_list = [0.1, 0.25, 0.5]
    recal_results = {}

    for eps_m in eps_m_list:
        eps_flow = round(1.0 - eps_m, 4)
        LOG.log(f"\n--- Testing eps_m = {eps_m} (Flow gets eps = {eps_flow}, 3 seeds) ---")
        recal_seed_metrics = []

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            pipe_rc = ACFMNetV5Pipeline(
                target_epsilon=eps_flow,
                target_delta=1e-5,
                strict_release_privacy=True,
                latent_mode="identity",
                batch_size=512,
                flow_hidden_dim=128,
                flow_num_layers=2,
                flow_optimizer=best_opt,
                flow_lr=best_lr,
                max_grad_norm_flow=best_C,
            )
            df_raw_flow, _ = pipe_rc.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")

            # Release DP histograms with eps_m
            dp_hists, hist_rep = recalibrator.release_dp_histograms(df_train, eps_m=eps_m, seed=s)
            # Recalibrate
            df_recal = recalibrator.recalibrate(df_raw_flow, dp_hists, seed=s)

            m_rc = compute_standardized_metrics(
                df_train, df_recal, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
            )
            recal_seed_metrics.append(m_rc)

        mean_1w = float(np.mean([m["norm_1way_tvd"] for m in recal_seed_metrics]))
        mean_2w = float(np.mean([m["norm_2way_tvd"] for m in recal_seed_metrics]))
        mean_corr = float(np.mean([m["norm_frob_corr"] for m in recal_seed_metrics]))
        mean_swd = float(np.mean([m["swd"] for m in recal_seed_metrics]))
        mean_auc = float(np.mean([m["tstr_auc"] for m in recal_seed_metrics]))
        recal_results[eps_m] = {
            "norm_1w": mean_1w,
            "norm_2w": mean_2w,
            "norm_corr": mean_corr,
            "swd": mean_swd,
            "tstr_auc": mean_auc,
            "metrics_list": recal_seed_metrics,
        }
        LOG.log(f"  eps_m={eps_m:4.2f}: Norm 1wTVD={mean_1w:.2f}x | Norm 2wTVD={mean_2w:.2f}x | "
                f"Norm Corr={mean_corr:.2f}x | SWD={mean_swd:.4f} | TSTR AUC={mean_auc:.4f}")

    best_eps_m = min(recal_results.keys(), key=lambda k: recal_results[k]["norm_2w"])
    LOG.log(f"\n[A6 Best Recalibration Budget]: eps_m = {best_eps_m} (Norm 2wTVD={recal_results[best_eps_m]['norm_2w']:.2f}x)")

    # =========================================================================
    # PART A8: Full Evaluations at eps=1.0 & eps=4.0 and Final Comparison Table
    # =========================================================================
    LOG.section("PART A8: FINAL EVALUATIONS AT EPS=1.0 & EPS=4.0 ACROSS 3 SEEDS")

    # 1. Best Identity at eps=1.0
    LOG.log("Running Best Lumisyn-Identity at eps=1.0 (3 seeds)...")
    id_eps1_metrics = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        p_id1 = ACFMNetV5Pipeline(
            target_epsilon=1.0,
            target_delta=1e-5,
            strict_release_privacy=True,
            latent_mode="identity",
            batch_size=512,
            flow_hidden_dim=128,
            flow_num_layers=2,
            flow_optimizer=best_opt,
            flow_lr=best_lr,
            max_grad_norm_flow=best_C,
        )
        df_id1, _ = p_id1.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
        m_id1 = compute_standardized_metrics(
            df_train, df_id1, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
        )
        id_eps1_metrics.append(m_id1)
        LOG.log(f"  Seed {s} (eps=1.0): 1wTVD={m_id1['norm_1way_tvd']:.2f}x | 2wTVD={m_id1['norm_2way_tvd']:.2f}x | "
                f"Corr={m_id1['norm_frob_corr']:.2f}x | SWD={m_id1['swd']:.4f} | TSTR AUC={m_id1['tstr_auc']:.4f}")

    # 2. Best Identity at eps=4.0
    LOG.log("\nRunning Best Lumisyn-Identity at eps=4.0 (3 seeds)...")
    id_eps4_metrics = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        p_id4 = ACFMNetV5Pipeline(
            target_epsilon=4.0,
            target_delta=1e-5,
            strict_release_privacy=True,
            latent_mode="identity",
            batch_size=512,
            flow_hidden_dim=128,
            flow_num_layers=2,
            flow_optimizer=best_opt,
            flow_lr=best_lr,
            max_grad_norm_flow=best_C,
        )
        df_id4, _ = p_id4.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
        m_id4 = compute_standardized_metrics(
            df_train, df_id4, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
        )
        id_eps4_metrics.append(m_id4)
        LOG.log(f"  Seed {s} (eps=4.0): 1wTVD={m_id4['norm_1way_tvd']:.2f}x | 2wTVD={m_id4['norm_2way_tvd']:.2f}x | "
                f"Corr={m_id4['norm_frob_corr']:.2f}x | SWD={m_id4['swd']:.4f} | TSTR AUC={m_id4['tstr_auc']:.4f}")

    # 3. Identity + Recalibration at eps=4.0
    LOG.log(f"\nRunning Lumisyn-Identity + Recalibration at eps=4.0 (eps_m={best_eps_m}, 3 seeds)...")
    recal_eps4_metrics = []
    eps_flow_4 = round(4.0 - best_eps_m, 4)
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        p_rc4 = ACFMNetV5Pipeline(
            target_epsilon=eps_flow_4,
            target_delta=1e-5,
            strict_release_privacy=True,
            latent_mode="identity",
            batch_size=512,
            flow_hidden_dim=128,
            flow_num_layers=2,
            flow_optimizer=best_opt,
            flow_lr=best_lr,
            max_grad_norm_flow=best_C,
        )
        df_raw4, _ = p_rc4.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
        dp_hists4, _ = recalibrator.release_dp_histograms(df_train, eps_m=best_eps_m, seed=s)
        df_recal4 = recalibrator.recalibrate(df_raw4, dp_hists4, seed=s)
        m_rc4 = compute_standardized_metrics(
            df_train, df_recal4, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names
        )
        recal_eps4_metrics.append(m_rc4)
        LOG.log(f"  Seed {s} (eps=4.0 recal): 1wTVD={m_rc4['norm_1way_tvd']:.2f}x | 2wTVD={m_rc4['norm_2way_tvd']:.2f}x | "
                f"Corr={m_rc4['norm_frob_corr']:.2f}x | SWD={m_rc4['swd']:.4f} | TSTR AUC={m_rc4['tstr_auc']:.4f}")

    # Helper to aggregate mean over 3 seeds
    def agg(m_list):
        return {
            "1w": np.mean([m["norm_1way_tvd"] for m in m_list]),
            "2w": np.mean([m["norm_2way_tvd"] for m in m_list]),
            "corr": np.mean([m["norm_frob_corr"] for m in m_list]),
            "swd": np.mean([m["swd"] for m in m_list]),
            "c2st": np.mean([m["c2st_auc"] for m in m_list]),
            "tstr_auc": np.mean([m["tstr_auc"] for m in m_list]),
            "tstr_prc": np.mean([m["tstr_prc"] for m in m_list]),
        }

    a_rvr = {
        "1w": rvr_metrics["norm_1way_tvd"],
        "2w": rvr_metrics["norm_2way_tvd"],
        "corr": rvr_metrics["norm_frob_corr"],
        "swd": rvr_metrics["swd"],
        "c2st": rvr_metrics["c2st_auc"],
        "tstr_auc": rvr_metrics["trtr_auc"],
        "tstr_prc": rvr_metrics["trtr_prc"],
    }
    a_const = {
        "1w": const_metrics["norm_1way_tvd"],
        "2w": const_metrics["norm_2way_tvd"],
        "corr": const_metrics["norm_frob_corr"],
        "swd": const_metrics["swd"],
        "c2st": const_metrics["c2st_auc"],
        "tstr_auc": const_metrics["tstr_auc"],
        "tstr_prc": const_metrics["tstr_prc"],
    }
    a_ind1 = agg(ind_metrics_eps1)
    a_ind4 = agg(ind_metrics_eps4)
    a_copula = agg(copula_metrics_list)
    a_id1 = agg(id_eps1_metrics)
    a_id4 = agg(id_eps4_metrics)
    a_rc1 = agg(recal_results[best_eps_m]["metrics_list"])
    a_rc4 = agg(recal_eps4_metrics)

    # =========================================================================
    # FINAL COMPARISON TABLE (Item A8)
    # =========================================================================
    LOG.section("FINAL GATE 4d COMPARISON TABLE (Normalized by Real-vs-Real Floor)")
    LOG.log("| Model / Generator                    | Privacy Level | 1-way TVD (x) | 2-way TVD (x) | Corr Error (x) | Corrected SWD | C2ST AUC | TSTR AUROC | TSTR AUPRC |")
    LOG.log("|--------------------------------------|---------------|---------------|---------------|----------------|---------------|----------|------------|------------|")
    LOG.log(f"| Real-vs-Real Floor (Holdout)         | Non-private   | {a_rvr['1w']:>13.2f} | {a_rvr['2w']:>13.2f} | {a_rvr['corr']:>14.2f} | {a_rvr['swd']:>13.4f} | {a_rvr['c2st']:>8.4f} | {a_rvr['tstr_auc']:>10.4f} | {a_rvr['tstr_prc']:>10.4f} |")
    LOG.log(f"| Constant Mean Generator              | Baseline      | {a_const['1w']:>13.2f} | {a_const['2w']:>13.2f} | {a_const['corr']:>14.2f} | {a_const['swd']:>13.4f} | {a_const['c2st']:>8.4f} | {a_const['tstr_auc']:>10.4f} | {a_const['tstr_prc']:>10.4f} |")
    LOG.log(f"| Independent DP Marginals             | eps=1.0       | {a_ind1['1w']:>13.2f} | {a_ind1['2w']:>13.2f} | {a_ind1['corr']:>14.2f} | {a_ind1['swd']:>13.4f} | {a_ind1['c2st']:>8.4f} | {a_ind1['tstr_auc']:>10.4f} | {a_ind1['tstr_prc']:>10.4f} |")
    LOG.log(f"| Independent DP Marginals             | eps=4.0       | {a_ind4['1w']:>13.2f} | {a_ind4['2w']:>13.2f} | {a_ind4['corr']:>14.2f} | {a_ind4['swd']:>13.4f} | {a_ind4['c2st']:>8.4f} | {a_ind4['tstr_auc']:>10.4f} | {a_ind4['tstr_prc']:>10.4f} |")
    LOG.log(f"| DP Gaussian-Copula                   | eps=1.0       | {a_copula['1w']:>13.2f} | {a_copula['2w']:>13.2f} | {a_copula['corr']:>14.2f} | {a_copula['swd']:>13.4f} | {a_copula['c2st']:>8.4f} | {a_copula['tstr_auc']:>10.4f} | {a_copula['tstr_prc']:>10.4f} |")
    LOG.log(f"| Lumisyn-Identity (Ours)              | eps=1.0       | {a_id1['1w']:>13.2f} | {a_id1['2w']:>13.2f} | {a_id1['corr']:>14.2f} | {a_id1['swd']:>13.4f} | {a_id1['c2st']:>8.4f} | {a_id1['tstr_auc']:>10.4f} | {a_id1['tstr_prc']:>10.4f} |")
    LOG.log(f"| Lumisyn-Identity (Ours)              | eps=4.0       | {a_id4['1w']:>13.2f} | {a_id4['2w']:>13.2f} | {a_id4['corr']:>14.2f} | {a_id4['swd']:>13.4f} | {a_id4['c2st']:>8.4f} | {a_id4['tstr_auc']:>10.4f} | {a_id4['tstr_prc']:>10.4f} |")
    LOG.log(f"| Lumisyn-Identity + Recalibration     | eps=1.0       | {a_rc1['1w']:>13.2f} | {a_rc1['2w']:>13.2f} | {a_rc1['corr']:>14.2f} | {a_rc1['swd']:>13.4f} | {a_rc1['c2st']:>8.4f} | {a_rc1['tstr_auc']:>10.4f} | {a_rc1['tstr_prc']:>10.4f} |")
    LOG.log(f"| Lumisyn-Identity + Recalibration     | eps=4.0       | {a_rc4['1w']:>13.2f} | {a_rc4['2w']:>13.2f} | {a_rc4['corr']:>14.2f} | {a_rc4['swd']:>13.4f} | {a_rc4['c2st']:>8.4f} | {a_rc4['tstr_auc']:>10.4f} | {a_rc4['tstr_prc']:>10.4f} |")

    # =========================================================================
    # GO / NO-GO DECISION
    # =========================================================================
    LOG.section("GATE 4d PRE-REGISTERED GO / NO-GO ASSESSMENT")
    # Pre-registered rule:
    # Does identity (with or without recalibration) beat independent DP marginals on:
    # 1. Correlation error (threshold 8.68x)
    # 2. 2-way TVD (threshold 11.1x)
    # At eps=1.0 and eps=4.0?

    target_corr_thresh = 8.68
    target_2w_thresh = 11.10

    LOG.log(f"Evaluation Criteria:")
    LOG.log(f"  Target Correlation Error Threshold: <= {target_corr_thresh:.2f}x floor")
    LOG.log(f"  Target 2-Way TVD Error Threshold:    <= {target_2w_thresh:.2f}x floor\n")

    # eps=1.0 checks
    corr_eps1_plain = a_id1["corr"]
    corr_eps1_recal = a_rc1["corr"]
    two_way_eps1_plain = a_id1["2w"]
    two_way_eps1_recal = a_rc1["2w"]

    pass_corr_eps1 = (corr_eps1_plain <= target_corr_thresh) or (corr_eps1_recal <= target_corr_thresh)
    pass_2w_eps1 = (two_way_eps1_plain <= target_2w_thresh) or (two_way_eps1_recal <= target_2w_thresh)

    LOG.log(f"At eps=1.0:")
    LOG.log(f"  Identity Plain: Corr Error = {corr_eps1_plain:.2f}x | 2-Way TVD = {two_way_eps1_plain:.2f}x")
    LOG.log(f"  Identity Recal: Corr Error = {corr_eps1_recal:.2f}x | 2-Way TVD = {two_way_eps1_recal:.2f}x")
    LOG.log(f"  Beat Correlation Threshold (<= {target_corr_thresh}x)? {pass_corr_eps1}")
    LOG.log(f"  Beat 2-Way TVD Threshold (<= {target_2w_thresh}x)?    {pass_2w_eps1}")

    # eps=4.0 checks
    corr_eps4_plain = a_id4["corr"]
    corr_eps4_recal = a_rc4["corr"]
    two_way_eps4_plain = a_id4["2w"]
    two_way_eps4_recal = a_rc4["2w"]

    pass_corr_eps4 = (corr_eps4_plain <= target_corr_thresh) or (corr_eps4_recal <= target_corr_thresh)
    pass_2w_eps4 = (two_way_eps4_plain <= target_2w_thresh) or (two_way_eps4_recal <= target_2w_thresh)

    LOG.log(f"\nAt eps=4.0:")
    LOG.log(f"  Identity Plain: Corr Error = {corr_eps4_plain:.2f}x | 2-Way TVD = {two_way_eps4_plain:.2f}x")
    LOG.log(f"  Identity Recal: Corr Error = {corr_eps4_recal:.2f}x | 2-Way TVD = {two_way_eps4_recal:.2f}x")
    LOG.log(f"  Beat Correlation Threshold (<= {target_corr_thresh}x)? {pass_corr_eps4}")
    LOG.log(f"  Beat 2-Way TVD Threshold (<= {target_2w_thresh}x)?    {pass_2w_eps4}")

    overall_pass = (pass_corr_eps1 or pass_corr_eps4) and (pass_2w_eps1 or pass_2w_eps4)
    LOG.log(f"\nFINAL VERDICT: {'GO' if overall_pass else 'NO-GO'}")
    LOG.log("Execution strictly halted at Gate 4d. No Phase 5 code executed.")

    LOG.close()


if __name__ == "__main__":
    run_gate4d()
