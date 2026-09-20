import os
import sys
import time
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    SlicedWassersteinEvaluator,
    _encode_for_eval,
)
from acfm_net.evaluation.dp_gaussian_copula import DPGaussianCopulaGenerator
from scripts.run_tier_a import load_adult_validation_half, compute_all_metrics

def main():
    print("Running Steps b and c for full Tier A diagnostics...")
    df_train, df_val = load_adult_validation_half(seed=42)
    schema_path = REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml"
    with open(schema_path, "r", encoding="utf-8") as f:
        adult_schema = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_cols = [c for c in df_train.columns if not c.endswith("_isna")]
    Xr_train, feat_names = _encode_for_eval(df_train[common_cols])
    real_train_std = np.std(Xr_train, axis=0)

    # Step b: DP eps=1 identity at prior scale s in {0.25, 0.5, 1.0}
    scales = [0.25, 0.5, 1.0]
    dp_scale_results = {}
    num_cols = [c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c])]

    for s_val in scales:
        t0 = time.time()
        torch.manual_seed(42)
        np.random.seed(42)
        pipe_b = ACFMNetV5Pipeline(
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
            prior_scale=s_val,
        )
        df_syn_b, rep_b = pipe_b.fit_generate(df_train, num_samples=len(df_val), epochs=10, target_col="class")
        m_b = compute_all_metrics(df_train, df_syn_b, df_val, real_train_std=real_train_std, feat_names=feat_names)
        std_real = df_train[num_cols].std().values
        std_syn = df_syn_b[num_cols].std().values
        col_std_mismatch = float(np.mean(np.abs(std_real - std_syn) / np.maximum(1e-6, std_real)))

        dp_scale_results[s_val] = {
            "norm_1way_tvd": m_b["norm_1way_tvd"],
            "raw_1way_tvd": m_b["raw_1way_tvd"],
            "norm_2way_tvd": m_b["norm_2way_tvd"],
            "norm_corr": m_b["norm_corr"],
            "swd": m_b["swd"],
            "col_std_mismatch": col_std_mismatch,
            "per_feature_1way": m_b["per_feature_1way"],
            "elapsed": time.time() - t0,
        }
        print(f"s={s_val}: 1w={m_b['norm_1way_tvd']:.2f}x, 2w={m_b['norm_2way_tvd']:.2f}x, corr={m_b['norm_corr']:.2f}x, swd={m_b['swd']:.4f}, std_mismatch={col_std_mismatch:.2%}")

    # Best s
    best_s = min(scales, key=lambda s: dp_scale_results[s]["norm_1way_tvd"])
    worst_10 = sorted(dp_scale_results[best_s]["per_feature_1way"].items(), key=lambda x: x[1], reverse=True)[:10]

    # Step c: DP Kendall-tau copula with NON-PRIVATE tau as positive control
    t0 = time.time()
    col_specs = {}
    for col in df_train.columns:
        if col.endswith("_isna"):
            continue
        if pd.api.types.is_numeric_dtype(df_train[col]):
            col_specs[col] = {
                "type": "numeric",
                "bounds": (float(df_train[col].min()), float(df_train[col].max())),
            }
        else:
            cats = list(df_train[col].dropna().unique())
            col_specs[col] = {
                "type": "categorical",
                "categories": cats,
            }
    copula = DPGaussianCopulaGenerator(
        target_epsilon=1.0,
        eps_marginal_ratio=0.5,
        eps_corr_ratio=0.5,
        columns_spec=col_specs,
        seed=42,
    )
    copula.eps_corr = 1e6
    copula.eps_marginal = 1.0
    copula.fit(df_train)
    df_syn_c = copula.generate(len(df_val), seed=42)
    m_c = compute_all_metrics(df_train, df_syn_c, df_val, real_train_std=real_train_std, feat_names=feat_names)
    print(f"Copula (non-priv tau): 1w={m_c['norm_1way_tvd']:.2f}x, 2w={m_c['norm_2way_tvd']:.2f}x, corr={m_c['norm_corr']:.2f}x, swd={m_c['swd']:.4f}")

    out = {
        "dp_scale_results": dp_scale_results,
        "best_s": best_s,
        "worst_10_columns": worst_10,
        "copula_pos_control": {
            "norm_1way_tvd": m_c["norm_1way_tvd"],
            "raw_1way_tvd": m_c["raw_1way_tvd"],
            "norm_2way_tvd": m_c["norm_2way_tvd"],
            "norm_corr": m_c["norm_corr"],
            "swd": m_c["swd"],
            "elapsed": time.time() - t0,
        }
    }
    with open(REPO_ROOT / "runs" / "tier_a_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Diagnostics saved to runs/tier_a_diagnostics.json")

if __name__ == "__main__":
    main()
