import yaml
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))

from acfm_net.evaluation.dp_gaussian_copula import DPGaussianCopulaGenerator
from scripts.run_tier_a import load_adult_validation_half, compute_all_metrics

def main():
    df_train, df_val = load_adult_validation_half(seed=42)
    schema_path = REPO_ROOT / "lumisyn" / "data" / "schemas" / "adult_schema.yaml"
    copula = DPGaussianCopulaGenerator(schema_path, epsilon=1.0, eps_marginal_ratio=0.5, seed=42)
    copula.eps_corr = 1e6 # Effectively non-private tau as positive control
    t0 = time.time()
    copula.fit(df_train)
    df_syn_c = copula.generate(len(df_val), seed=42)
    t_gen = time.time() - t0
    m_c = compute_all_metrics(df_train, df_syn_c, df_val)

    print("================================================================================")
    print(" DP KENDALL-TAU COPULA WITH NON-PRIVATE TAU (POSITIVE CONTROL)")
    print("================================================================================")
    print(f"Time: {t_gen:.2f}s")
    print(f"Normalized 1-way TVD:   {m_c['norm_1way_tvd']:.4f}x (Raw: {m_c['raw_1way_tvd']:.4f}, Floor: {m_c['floor_1way']:.4f})")
    print(f"Normalized 2-way TVD:   {m_c['norm_2way_tvd']:.4f}x")
    print(f"Normalized Correlation: {m_c['norm_corr']:.4f}x")
    print(f"Standardized SWD:       {m_c['swd']:.4f}")

if __name__ == "__main__":
    main()
