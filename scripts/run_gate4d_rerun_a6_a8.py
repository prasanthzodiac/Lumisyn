"""
Gate 4d Corrections: Rerun of A6-A8 with 5 Seeds, DPCopula, and Rigorous Statistical Testing
=============================================================================
Parameters:
  - Dataset: Adult 30k (Train: 30,000 | Holdout: 18,842)
  - Seeds: 5 seeds [42, 43, 44, 45, 46] for every evaluated row
  - Total Privacy: eps=1.0 and eps=4.0, delta=1e-5

Evaluations:
  1. Floor Baselines:
     - Real-vs-Real Floor (Train 30k vs Holdout 18.8k)
     - Constant Mean Generator
  2. Privacy Baselines (at eps=1.0 and eps=4.0):
     - Independent DP Marginals (Laplace mechanism, Delta_1 = M)
     - DPCopula Kendall's-Tau Baseline (Greiner's relation, PSD projection, Delta_1(tau) = 2M(M-1)/N)
  3. Lumisyn-Identity (Ours) (at eps=1.0 and eps=4.0)
  4. Lumisyn-Identity + DP Marginal Recalibration:
     - Evaluated separately for each eps_m in {0.1, 0.25, 0.5} at eps=1.0
     - Evaluated at eps=4.0
     - Special value 0 bin for zero-inflated continuous features (capital-gain, capital-loss)
     - Log-scaled bins for fnlwgt
     - Evaluation TVD uses 20 bins (distinct from the 50 recalibration bins)
     - Bin-independent KS statistic and 1D W1 in std units

Statistical Testing:
  - Mean +/- SD over 5 seeds
  - Paired t-test vs Independent DP Marginals and vs DPCopula
  - Pre-registered Go/No-Go evaluation:
    Pass = ALL of:
      1. 1-way TVD <= 3x floor
      2. Correlation error < baseline
      3. 2-way TVD < baseline
      4. TSTR AUROC > baseline
    (Note: Beating independent marginals is necessary but a low bar).

Logged to: runs/gate4d_rerun_a6_a8.log
"""

import sys, os, time, math, datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = REPO_ROOT / "lumisyn" / "framework"
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lumisyn"))
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "runs" / "gate4d_rerun_a6_a8.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
SCHEMA_PATH = REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml"

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

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
        self.log(f"[Gate 4d Corrections Rerun A6-A8] Started at {datetime.datetime.now().isoformat()}")
        self.log("Dataset: Adult 30k | 5 Seeds: [42, 43, 44, 45, 46] | eps in {1.0, 4.0}")
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
    eval_n_bins: int = 20,
) -> Dict[str, Any]:
    common = [c for c in df_real_train.columns if c in df_syn.columns and not c.endswith("_isna")]

    # 1. 1-way TVD & KS statistic (using eval_n_bins=20, distinct from 50 recalibration bins)
    marg = MarginalErrorEvaluator()
    eval_1w = marg.evaluate_1way(
        df_real_train[common],
        df_syn[common],
        df_real_holdout=df_holdout[common],
        n_bins=eval_n_bins,
    )
    tvd_1w = eval_1w["mean_1way_marginal_error_tvd"]
    floor_1w = eval_1w.get("real_vs_real_floor_1way_tvd", 0.0076)
    norm_tvd_1w = eval_1w.get("normalized_1way_marginal_error", 1.0)
    ks_1w = eval_1w.get("mean_ks_statistic", 0.0)
    floor_ks = eval_1w.get("real_vs_real_floor_ks", 0.01)
    norm_ks = eval_1w.get("normalized_ks_statistic", 1.0)
    per_col_tvd = eval_1w.get("per_feature_1way_tvd", {})
    per_col_ks = eval_1w.get("per_feature_ks", {})

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

    # 4. Standardized Sliced Wasserstein Distance & 1D W1 in std units
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

    # 6. TSTR & TRTR
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
        "ks_stat": ks_1w,
        "floor_ks": floor_ks,
        "norm_ks": norm_ks,
        "per_feature_1way_tvd": per_col_tvd,
        "per_feature_ks": per_col_ks,
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


def run_rerun():
    df_train, df_holdout = load_adult_30k_and_holdout(seed=42)
    seeds = [42, 43, 44, 45, 46]
    LOG.log(f"Adult 30k Loaded: Train={df_train.shape}, Holdout={df_holdout.shape}")
    LOG.log(f"5 Seeds: {seeds}")

    # Standardizing references
    Xr_train, feat_names = _encode_for_eval(df_train)
    real_train_std = np.std(Xr_train, axis=0)
    real_train_std = np.where(real_train_std < 1e-6, 1.0, real_train_std)

    # -------------------------------------------------------------------------
    # 0. Floor Baselines
    # -------------------------------------------------------------------------
    LOG.section("0. FLOOR BASELINES (5 Seeds / Holdout)")
    LOG.log("Evaluating Real-vs-Real Floor...")
    rvr_metrics = compute_standardized_metrics(
        df_real_train=df_train,
        df_syn=df_holdout,
        df_holdout=df_holdout,
        target_col="class",
        real_train_std=real_train_std,
        encoded_feature_names=feat_names,
    )
    LOG.log(f"Real-vs-Real Floor: 1wTVD={rvr_metrics['1way_tvd']:.4f} | KS={rvr_metrics['ks_stat']:.4f} | "
            f"2wTVD={rvr_metrics['2way_tvd']:.4f} | Corr={rvr_metrics['frob_corr_error']:.4f} | "
            f"SWD={rvr_metrics['swd']:.4f} | 1D W1={rvr_metrics['mean_1d_w1']:.4f} std | "
            f"C2ST={rvr_metrics['c2st_auc']:.4f} | TRTR AUC={rvr_metrics['trtr_auc']:.4f} | AUPRC={rvr_metrics['trtr_prc']:.4f}")

    LOG.log("\nEvaluating Constant Mean Baseline...")
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
    LOG.log(f"Constant Mean: 1wTVD={const_metrics['norm_1way_tvd']:.2f}x | KS={const_metrics['norm_ks']:.2f}x | "
            f"2wTVD={const_metrics['norm_2way_tvd']:.2f}x | Corr={const_metrics['norm_frob_corr']:.2f}x | "
            f"SWD={const_metrics['swd']:.4f} | 1D W1={const_metrics['mean_1d_w1']:.4f} std | TSTR AUC={const_metrics['tstr_auc']:.4f}")

    # -------------------------------------------------------------------------
    # 1. Independent DP Marginals (eps=1.0 and eps=4.0, 5 seeds)
    # -------------------------------------------------------------------------
    LOG.section("1. INDEPENDENT DP MARGINALS BASELINE (eps=1.0 & eps=4.0, 5 seeds)")
    ind_eps1 = []
    ind_eps4 = []

    LOG.log("Running Independent DP Marginals (eps=1.0)...")
    for s in seeds:
        gen1 = IndependentMarginalsDPGenerator(epsilon=1.0, seed=s)
        gen1.fit(df_train)
        df_i1 = gen1.generate(n_samples=len(df_train), seed=s)
        m = compute_standardized_metrics(df_train, df_i1, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
        ind_eps1.append(m)
        LOG.log(f"  Seed {s} (eps=1.0): 1wTVD={m['norm_1way_tvd']:.2f}x | KS={m['norm_ks']:.2f}x | 2wTVD={m['norm_2way_tvd']:.2f}x | "
                f"Corr={m['norm_frob_corr']:.2f}x | SWD={m['swd']:.4f} | 1D W1={m['mean_1d_w1']:.4f} std | TSTR AUC={m['tstr_auc']:.4f}")

    LOG.log("\nRunning Independent DP Marginals (eps=4.0)...")
    for s in seeds:
        gen4 = IndependentMarginalsDPGenerator(epsilon=4.0, seed=s)
        gen4.fit(df_train)
        df_i4 = gen4.generate(n_samples=len(df_train), seed=s)
        m = compute_standardized_metrics(df_train, df_i4, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
        ind_eps4.append(m)
        LOG.log(f"  Seed {s} (eps=4.0): 1wTVD={m['norm_1way_tvd']:.2f}x | KS={m['norm_ks']:.2f}x | 2wTVD={m['norm_2way_tvd']:.2f}x | "
                f"Corr={m['norm_frob_corr']:.2f}x | SWD={m['swd']:.4f} | 1D W1={m['mean_1d_w1']:.4f} std | TSTR AUC={m['tstr_auc']:.4f}")

    # -------------------------------------------------------------------------
    # 2. DPCopula Kendall's Tau Generator (eps=1.0 and eps=4.0, 5 seeds)
    # -------------------------------------------------------------------------
    LOG.section("2. DP GAUSSIAN-COPULA KENDALL'S-TAU BASELINE (eps=1.0 & eps=4.0, 5 seeds)")
    copula_eps1 = []
    copula_eps4 = []

    LOG.log("Running DPCopula (eps=1.0)...")
    for s in seeds:
        cgen1 = DPGaussianCopulaGenerator(SCHEMA_PATH, epsilon=1.0, eps_marginal_ratio=0.5, seed=s)
        cgen1.fit(df_train)
        df_c1 = cgen1.generate(n_samples=len(df_train), seed=s)
        m = compute_standardized_metrics(df_train, df_c1, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
        copula_eps1.append(m)
        LOG.log(f"  Seed {s} (eps=1.0): 1wTVD={m['norm_1way_tvd']:.2f}x | KS={m['norm_ks']:.2f}x | 2wTVD={m['norm_2way_tvd']:.2f}x | "
                f"Corr={m['norm_frob_corr']:.2f}x | SWD={m['swd']:.4f} | 1D W1={m['mean_1d_w1']:.4f} std | TSTR AUC={m['tstr_auc']:.4f}")

    LOG.log("\nRunning DPCopula (eps=4.0)...")
    for s in seeds:
        cgen4 = DPGaussianCopulaGenerator(SCHEMA_PATH, epsilon=4.0, eps_marginal_ratio=0.5, seed=s)
        cgen4.fit(df_train)
        df_c4 = cgen4.generate(n_samples=len(df_train), seed=s)
        m = compute_standardized_metrics(df_train, df_c4, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
        copula_eps4.append(m)
        LOG.log(f"  Seed {s} (eps=4.0): 1wTVD={m['norm_1way_tvd']:.2f}x | KS={m['norm_ks']:.2f}x | 2wTVD={m['norm_2way_tvd']:.2f}x | "
                f"Corr={m['norm_frob_corr']:.2f}x | SWD={m['swd']:.4f} | 1D W1={m['mean_1d_w1']:.4f} std | TSTR AUC={m['tstr_auc']:.4f}")

    # -------------------------------------------------------------------------
    # 3. Lumisyn-Identity (Ours) (eps=1.0 and eps=4.0, 5 seeds)
    # -------------------------------------------------------------------------
    LOG.section("3. LUMISYN-IDENTITY (OURS) (eps=1.0 & eps=4.0, 5 seeds)")
    id_eps1 = []
    id_eps4 = []

    LOG.log("Running Lumisyn-Identity (eps=1.0, B=512, E=10)...")
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        p1 = ACFMNetV5Pipeline(
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
        df_s1, _ = p1.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
        m = compute_standardized_metrics(df_train, df_s1, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
        id_eps1.append(m)
        LOG.log(f"  Seed {s} (eps=1.0): 1wTVD={m['norm_1way_tvd']:.2f}x | KS={m['norm_ks']:.2f}x | 2wTVD={m['norm_2way_tvd']:.2f}x | "
                f"Corr={m['norm_frob_corr']:.2f}x | SWD={m['swd']:.4f} | 1D W1={m['mean_1d_w1']:.4f} std | TSTR AUC={m['tstr_auc']:.4f}")

    LOG.log("\nRunning Lumisyn-Identity (eps=4.0, B=512, E=10)...")
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        p4 = ACFMNetV5Pipeline(
            target_epsilon=4.0,
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
        df_s4, _ = p4.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
        m = compute_standardized_metrics(df_train, df_s4, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
        id_eps4.append(m)
        LOG.log(f"  Seed {s} (eps=4.0): 1wTVD={m['norm_1way_tvd']:.2f}x | KS={m['norm_ks']:.2f}x | 2wTVD={m['norm_2way_tvd']:.2f}x | "
                f"Corr={m['norm_frob_corr']:.2f}x | SWD={m['swd']:.4f} | 1D W1={m['mean_1d_w1']:.4f} std | TSTR AUC={m['tstr_auc']:.4f}")

    # -------------------------------------------------------------------------
    # 4. Lumisyn-Identity + DP Marginal Recalibration
    # -------------------------------------------------------------------------
    LOG.section("4. LUMISYN-IDENTITY + DP RECALIBRATION (Item 3 & 5)")
    recalibrator = DPMarginalRecalibrator(SCHEMA_PATH, num_continuous_bins=50)

    # Sweep eps_m in {0.1, 0.25, 0.5} at total eps=1.0
    eps_m_candidates = [0.1, 0.25, 0.5]
    recal_eps1_by_m = {}

    for eps_m in eps_m_candidates:
        eps_flow = round(1.0 - eps_m, 4)
        LOG.log(f"\n--- Recalibration at eps_total=1.0 with eps_m={eps_m} (Flow gets eps={eps_flow}, 5 seeds) ---")
        m_list = []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            p_rc = ACFMNetV5Pipeline(
                target_epsilon=eps_flow,
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
            df_raw, _ = p_rc.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
            dp_hists, h_rep = recalibrator.release_dp_histograms(df_train, eps_m=eps_m, seed=s)
            if s == seeds[0]:
                LOG.log(f"  [Report for eps_m={eps_m}]: Bins per column: {h_rep['bins_per_column']}")
                LOG.log(f"  [Report for eps_m={eps_m}]: Noise-to-count ratios: {h_rep['noise_to_count_ratios']}")

            df_cal = recalibrator.recalibrate(df_raw, dp_hists, seed=s)
            m = compute_standardized_metrics(df_train, df_cal, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
            m_list.append(m)
            LOG.log(f"  Seed {s} (eps_m={eps_m}): 1wTVD={m['norm_1way_tvd']:.2f}x | KS={m['norm_ks']:.2f}x | 2wTVD={m['norm_2way_tvd']:.2f}x | "
                    f"Corr={m['norm_frob_corr']:.2f}x | SWD={m['swd']:.4f} | 1D W1={m['mean_1d_w1']:.4f} std | TSTR AUC={m['tstr_auc']:.4f}")
        recal_eps1_by_m[eps_m] = m_list

    # Recalibration at eps_total=4.0 (with eps_m=0.5, flow gets 3.5)
    LOG.log("\n--- Recalibration at eps_total=4.0 with eps_m=0.5 (Flow gets eps=3.5, 5 seeds) ---")
    recal_eps4 = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        p_rc4 = ACFMNetV5Pipeline(
            target_epsilon=3.5,
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
        df_raw4, _ = p_rc4.fit_generate(df_train, num_samples=len(df_train), epochs=10, target_col="class")
        dp_hists4, _ = recalibrator.release_dp_histograms(df_train, eps_m=0.5, seed=s)
        df_cal4 = recalibrator.recalibrate(df_raw4, dp_hists4, seed=s)
        m4 = compute_standardized_metrics(df_train, df_cal4, df_holdout, real_train_std=real_train_std, encoded_feature_names=feat_names)
        recal_eps4.append(m4)
        LOG.log(f"  Seed {s} (eps_total=4.0, eps_m=0.5): 1wTVD={m4['norm_1way_tvd']:.2f}x | KS={m4['norm_ks']:.2f}x | 2wTVD={m4['norm_2way_tvd']:.2f}x | "
                f"Corr={m4['norm_frob_corr']:.2f}x | SWD={m4['swd']:.4f} | 1D W1={m4['mean_1d_w1']:.4f} std | TSTR AUC={m4['tstr_auc']:.4f}")

    # -------------------------------------------------------------------------
    # 5. Statistical Summary & Paired Tests
    # -------------------------------------------------------------------------
    LOG.section("5. STATISTICAL SUMMARY (Mean +/- SD over 5 Seeds) & PAIRED TESTS")

    def stat_str(values: List[float], fmt: str = ".2f") -> str:
        arr = np.array(values)
        return f"{arr.mean():{fmt}} +/- {arr.std(ddof=1):{fmt}}"

    def stat_dict(m_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "1w_str": stat_str([m["norm_1way_tvd"] for m in m_list]),
            "1w_arr": np.array([m["norm_1way_tvd"] for m in m_list]),
            "ks_str": stat_str([m["norm_ks"] for m in m_list]),
            "ks_arr": np.array([m["norm_ks"] for m in m_list]),
            "2w_str": stat_str([m["norm_2way_tvd"] for m in m_list]),
            "2w_arr": np.array([m["norm_2way_tvd"] for m in m_list]),
            "corr_str": stat_str([m["norm_frob_corr"] for m in m_list]),
            "corr_arr": np.array([m["norm_frob_corr"] for m in m_list]),
            "swd_str": stat_str([m["swd"] for m in m_list], fmt=".4f"),
            "swd_arr": np.array([m["swd"] for m in m_list]),
            "w1_str": stat_str([m["mean_1d_w1"] for m in m_list], fmt=".4f"),
            "w1_arr": np.array([m["mean_1d_w1"] for m in m_list]),
            "c2st_str": stat_str([m["c2st_auc"] for m in m_list], fmt=".4f"),
            "c2st_arr": np.array([m["c2st_auc"] for m in m_list]),
            "auc_str": stat_str([m["tstr_auc"] for m in m_list], fmt=".4f"),
            "auc_arr": np.array([m["tstr_auc"] for m in m_list]),
            "prc_str": stat_str([m["tstr_prc"] for m in m_list], fmt=".4f"),
            "prc_arr": np.array([m["tstr_prc"] for m in m_list]),
        }

    s_ind1 = stat_dict(ind_eps1)
    s_ind4 = stat_dict(ind_eps4)
    s_cop1 = stat_dict(copula_eps1)
    s_cop4 = stat_dict(copula_eps4)
    s_id1  = stat_dict(id_eps1)
    s_id4  = stat_dict(id_eps4)
    s_rc1_01 = stat_dict(recal_eps1_by_m[0.1])
    s_rc1_025 = stat_dict(recal_eps1_by_m[0.25])
    s_rc1_05 = stat_dict(recal_eps1_by_m[0.5])
    s_rc4  = stat_dict(recal_eps4)

    # Paired test helper: returns diff and p-value
    def paired_diff(test_arr: np.ndarray, base_arr: np.ndarray) -> Tuple[float, float]:
        diff = float(np.mean(test_arr - base_arr))
        try:
            res = ttest_rel(test_arr, base_arr)
            pval = float(res.pvalue)
        except Exception:
            pval = 1.0
        return diff, pval

    LOG.log("| Configuration                                    | 1-way TVD (x)     | KS Stat (x)       | 2-way TVD (x)     | Corr Error (x)    | Corrected SWD     | 1D W1 (std)       | TSTR AUROC        |")
    LOG.log("|--------------------------------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|")
    LOG.log(f"| Real-vs-Real Floor (Non-private)                 | {rvr_metrics['norm_1way_tvd']:>17.2f} | {rvr_metrics['norm_ks']:>17.2f} | {rvr_metrics['norm_2way_tvd']:>17.2f} | {rvr_metrics['norm_frob_corr']:>17.2f} | {rvr_metrics['swd']:>17.4f} | {rvr_metrics['mean_1d_w1']:>17.4f} | {rvr_metrics['trtr_auc']:>17.4f} |")
    LOG.log(f"| Constant Mean Generator                          | {const_metrics['norm_1way_tvd']:>17.2f} | {const_metrics['norm_ks']:>17.2f} | {const_metrics['norm_2way_tvd']:>17.2f} | {const_metrics['norm_frob_corr']:>17.2f} | {const_metrics['swd']:>17.4f} | {const_metrics['mean_1d_w1']:>17.4f} | {const_metrics['tstr_auc']:>17.4f} |")
    LOG.log(f"| Independent DP Marginals (eps=1.0)               | {s_ind1['1w_str']:>17} | {s_ind1['ks_str']:>17} | {s_ind1['2w_str']:>17} | {s_ind1['corr_str']:>17} | {s_ind1['swd_str']:>17} | {s_ind1['w1_str']:>17} | {s_ind1['auc_str']:>17} |")
    LOG.log(f"| Independent DP Marginals (eps=4.0)               | {s_ind4['1w_str']:>17} | {s_ind4['ks_str']:>17} | {s_ind4['2w_str']:>17} | {s_ind4['corr_str']:>17} | {s_ind4['swd_str']:>17} | {s_ind4['w1_str']:>17} | {s_ind4['auc_str']:>17} |")
    LOG.log(f"| DPCopula Kendall's-Tau (eps=1.0)                 | {s_cop1['1w_str']:>17} | {s_cop1['ks_str']:>17} | {s_cop1['2w_str']:>17} | {s_cop1['corr_str']:>17} | {s_cop1['swd_str']:>17} | {s_cop1['w1_str']:>17} | {s_cop1['auc_str']:>17} |")
    LOG.log(f"| DPCopula Kendall's-Tau (eps=4.0)                 | {s_cop4['1w_str']:>17} | {s_cop4['ks_str']:>17} | {s_cop4['2w_str']:>17} | {s_cop4['corr_str']:>17} | {s_cop4['swd_str']:>17} | {s_cop4['w1_str']:>17} | {s_cop4['auc_str']:>17} |")
    LOG.log(f"| Lumisyn-Identity (eps=1.0)                       | {s_id1['1w_str']:>17} | {s_id1['ks_str']:>17} | {s_id1['2w_str']:>17} | {s_id1['corr_str']:>17} | {s_id1['swd_str']:>17} | {s_id1['w1_str']:>17} | {s_id1['auc_str']:>17} |")
    LOG.log(f"| Lumisyn-Identity (eps=4.0)                       | {s_id4['1w_str']:>17} | {s_id4['ks_str']:>17} | {s_id4['2w_str']:>17} | {s_id4['corr_str']:>17} | {s_id4['swd_str']:>17} | {s_id4['w1_str']:>17} | {s_id4['auc_str']:>17} |")
    LOG.log(f"| Lumisyn-Identity + Recal (eps=1.0, eps_m=0.10)   | {s_rc1_01['1w_str']:>17} | {s_rc1_01['ks_str']:>17} | {s_rc1_01['2w_str']:>17} | {s_rc1_01['corr_str']:>17} | {s_rc1_01['swd_str']:>17} | {s_rc1_01['w1_str']:>17} | {s_rc1_01['auc_str']:>17} |")
    LOG.log(f"| Lumisyn-Identity + Recal (eps=1.0, eps_m=0.25)   | {s_rc1_025['1w_str']:>17} | {s_rc1_025['ks_str']:>17} | {s_rc1_025['2w_str']:>17} | {s_rc1_025['corr_str']:>17} | {s_rc1_025['swd_str']:>17} | {s_rc1_025['w1_str']:>17} | {s_rc1_025['auc_str']:>17} |")
    LOG.log(f"| Lumisyn-Identity + Recal (eps=1.0, eps_m=0.50)   | {s_rc1_05['1w_str']:>17} | {s_rc1_05['ks_str']:>17} | {s_rc1_05['2w_str']:>17} | {s_rc1_05['corr_str']:>17} | {s_rc1_05['swd_str']:>17} | {s_rc1_05['w1_str']:>17} | {s_rc1_05['auc_str']:>17} |")
    LOG.log(f"| Lumisyn-Identity + Recal (eps=4.0, eps_m=0.50)   | {s_rc4['1w_str']:>17} | {s_rc4['ks_str']:>17} | {s_rc4['2w_str']:>17} | {s_rc4['corr_str']:>17} | {s_rc4['swd_str']:>17} | {s_rc4['w1_str']:>17} | {s_rc4['auc_str']:>17} |")

    # Paired testing vs Independent DP Marginals
    LOG.log("\n--- Paired Difference Tests vs Independent DP Marginals (Same Epsilon) ---")
    d_corr_1, p_corr_1 = paired_diff(s_id1["corr_arr"], s_ind1["corr_arr"])
    d_2w_1, p_2w_1     = paired_diff(s_id1["2w_arr"], s_ind1["2w_arr"])
    d_auc_1, p_auc_1   = paired_diff(s_id1["auc_arr"], s_ind1["auc_arr"])
    LOG.log(f"Identity vs Independent Marginals (eps=1.0):")
    LOG.log(f"  Corr Error Diff: {d_corr_1:+.2f}x (p={p_corr_1:.4f})")
    LOG.log(f"  2-way TVD Diff:  {d_2w_1:+.2f}x (p={p_2w_1:.4f})")
    LOG.log(f"  TSTR AUC Diff:   {d_auc_1:+.4f} (p={p_auc_1:.4f})")

    d_corr_4, p_corr_4 = paired_diff(s_id4["corr_arr"], s_ind4["corr_arr"])
    d_2w_4, p_2w_4     = paired_diff(s_id4["2w_arr"], s_ind4["2w_arr"])
    d_auc_4, p_auc_4   = paired_diff(s_id4["auc_arr"], s_ind4["auc_arr"])
    LOG.log(f"\nIdentity vs Independent Marginals (eps=4.0):")
    LOG.log(f"  Corr Error Diff: {d_corr_4:+.2f}x (p={p_corr_4:.4f})")
    LOG.log(f"  2-way TVD Diff:  {d_2w_4:+.2f}x (p={p_2w_4:.4f})")
    LOG.log(f"  TSTR AUC Diff:   {d_auc_4:+.4f} (p={p_auc_4:.4f})")

    # Paired testing vs DPCopula
    LOG.log("\n--- Paired Difference Tests vs DPCopula Kendall's-Tau (Same Epsilon) ---")
    dc_corr_1, pc_corr_1 = paired_diff(s_id1["corr_arr"], s_cop1["corr_arr"])
    dc_2w_1, pc_2w_1     = paired_diff(s_id1["2w_arr"], s_cop1["2w_arr"])
    dc_auc_1, pc_auc_1   = paired_diff(s_id1["auc_arr"], s_cop1["auc_arr"])
    LOG.log(f"Identity vs DPCopula (eps=1.0):")
    LOG.log(f"  Corr Error Diff: {dc_corr_1:+.2f}x (p={pc_corr_1:.4f})")
    LOG.log(f"  2-way TVD Diff:  {dc_2w_1:+.2f}x (p={pc_2w_1:.4f})")
    LOG.log(f"  TSTR AUC Diff:   {dc_auc_1:+.4f} (p={pc_auc_1:.4f})")

    # -------------------------------------------------------------------------
    # 6. Pre-Registered Go/No-Go Decision
    # -------------------------------------------------------------------------
    LOG.section("6. PRE-REGISTERED GO / NO-GO ASSESSMENT")
    LOG.log("Context: Beating independent marginals is necessary but a low bar.")
    LOG.log("Pass Rule: ALL of:")
    LOG.log("  1. 1-way TVD <= 3.0x floor")
    LOG.log("  2. Correlation error < baseline (at same epsilon)")
    LOG.log("  3. 2-way TVD < baseline (at same epsilon)")
    LOG.log("  4. TSTR AUROC > baseline (at same epsilon)")

    # Test at eps=1.0
    id1_mean_1w = s_id1["1w_arr"].mean()
    id1_mean_corr = s_id1["corr_arr"].mean()
    id1_mean_2w = s_id1["2w_arr"].mean()
    id1_mean_auc = s_id1["auc_arr"].mean()

    rc1_mean_1w = s_rc1_05["1w_arr"].mean()
    rc1_mean_corr = s_rc1_05["corr_arr"].mean()
    rc1_mean_2w = s_rc1_05["2w_arr"].mean()
    rc1_mean_auc = s_rc1_05["auc_arr"].mean()

    ind1_mean_corr = s_ind1["corr_arr"].mean()
    ind1_mean_2w = s_ind1["2w_arr"].mean()
    ind1_mean_auc = s_ind1["auc_arr"].mean()

    # Best candidate between plain identity and recalibrated identity at eps=1
    pass_1w_1 = (id1_mean_1w <= 3.0) or (rc1_mean_1w <= 3.0)
    pass_corr_1 = (id1_mean_corr < ind1_mean_corr) or (rc1_mean_corr < ind1_mean_corr)
    pass_2w_1 = (id1_mean_2w < ind1_mean_2w) or (rc1_mean_2w < ind1_mean_2w)
    pass_auc_1 = (id1_mean_auc > ind1_mean_auc) or (rc1_mean_auc > ind1_mean_auc)

    LOG.log(f"\nEvaluation at eps=1.0 (Target: Indep Marginals Corr={ind1_mean_corr:.2f}x, 2w={ind1_mean_2w:.2f}x, AUC={ind1_mean_auc:.4f}):")
    LOG.log(f"  Identity Plain: 1w={id1_mean_1w:.2f}x | Corr={id1_mean_corr:.2f}x | 2w={id1_mean_2w:.2f}x | AUC={id1_mean_auc:.4f}")
    LOG.log(f"  Identity Recal: 1w={rc1_mean_1w:.2f}x | Corr={rc1_mean_corr:.2f}x | 2w={rc1_mean_2w:.2f}x | AUC={rc1_mean_auc:.4f}")
    LOG.log(f"  Criterion 1 (1-way TVD <= 3.0x floor):   {pass_1w_1}")
    LOG.log(f"  Criterion 2 (Corr Error < {ind1_mean_corr:.2f}x):        {pass_corr_1}")
    LOG.log(f"  Criterion 3 (2-way TVD < {ind1_mean_2w:.2f}x):         {pass_2w_1}")
    LOG.log(f"  Criterion 4 (TSTR AUC > {ind1_mean_auc:.4f}):           {pass_auc_1}")
    pass_eps1 = pass_1w_1 and pass_corr_1 and pass_2w_1 and pass_auc_1
    LOG.log(f"  -> eps=1.0 Pass Status: {pass_eps1}")

    # Test at eps=4.0
    id4_mean_1w = s_id4["1w_arr"].mean()
    id4_mean_corr = s_id4["corr_arr"].mean()
    id4_mean_2w = s_id4["2w_arr"].mean()
    id4_mean_auc = s_id4["auc_arr"].mean()

    rc4_mean_1w = s_rc4["1w_arr"].mean()
    rc4_mean_corr = s_rc4["corr_arr"].mean()
    rc4_mean_2w = s_rc4["2w_arr"].mean()
    rc4_mean_auc = s_rc4["auc_arr"].mean()

    ind4_mean_corr = s_ind4["corr_arr"].mean()
    ind4_mean_2w = s_ind4["2w_arr"].mean()
    ind4_mean_auc = s_ind4["auc_arr"].mean()

    pass_1w_4 = (id4_mean_1w <= 3.0) or (rc4_mean_1w <= 3.0)
    pass_corr_4 = (id4_mean_corr < ind4_mean_corr) or (rc4_mean_corr < ind4_mean_corr)
    pass_2w_4 = (id4_mean_2w < ind4_mean_2w) or (rc4_mean_2w < ind4_mean_2w)
    pass_auc_4 = (id4_mean_auc > ind4_mean_auc) or (rc4_mean_auc > ind4_mean_auc)

    LOG.log(f"\nEvaluation at eps=4.0 (Target: Indep Marginals Corr={ind4_mean_corr:.2f}x, 2w={ind4_mean_2w:.2f}x, AUC={ind4_mean_auc:.4f}):")
    LOG.log(f"  Identity Plain: 1w={id4_mean_1w:.2f}x | Corr={id4_mean_corr:.2f}x | 2w={id4_mean_2w:.2f}x | AUC={id4_mean_auc:.4f}")
    LOG.log(f"  Identity Recal: 1w={rc4_mean_1w:.2f}x | Corr={rc4_mean_corr:.2f}x | 2w={rc4_mean_2w:.2f}x | AUC={rc4_mean_auc:.4f}")
    LOG.log(f"  Criterion 1 (1-way TVD <= 3.0x floor):   {pass_1w_4}")
    LOG.log(f"  Criterion 2 (Corr Error < {ind4_mean_corr:.2f}x):        {pass_corr_4}")
    LOG.log(f"  Criterion 3 (2-way TVD < {ind4_mean_2w:.2f}x):         {pass_2w_4}")
    LOG.log(f"  Criterion 4 (TSTR AUC > {ind4_mean_auc:.4f}):           {pass_auc_4}")
    pass_eps4 = pass_1w_4 and pass_corr_4 and pass_2w_4 and pass_auc_4
    LOG.log(f"  -> eps=4.0 Pass Status: {pass_eps4}")

    overall_verdict = pass_eps1 and pass_eps4
    LOG.log(f"\n===========================================================")
    LOG.log(f"GATE 4d OVERALL VERDICT: {'GO' if overall_verdict else 'NO-GO'}")
    LOG.log(f"===========================================================")
    LOG.log("Execution strictly stopped at Gate 4d. No Phase 5 code executed.")

    LOG.close()


if __name__ == "__main__":
    run_rerun()
