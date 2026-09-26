import sys
import os
import pandas as pd
import numpy as np

# Add Lumisyn framework & backend to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "framework"))
sys.path.append(os.path.join(PROJECT_ROOT, "backend"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from app_backend import get_preset_nhanes

def generate_synthetic_csv():
    print("=" * 60)
    print("ACFM-Net v5 Synthetic Data Generator")
    print("=" * 60)

    # Load high-quality clinical benchmark dataset (1,000 patient records)
    print("Generating seed clinical dataset (NHANES Clinical Schema)...")
    df_real, target_col = get_preset_nhanes(num_samples=1000)

    print(f"Seed Dataset Shape: {df_real.shape}")
    print(f"Target Column: {target_col}")
    print(f"Features: {list(df_real.columns)}")

    # Instantiate ACFM-Net v5 Pipeline with Differential Privacy target (epsilon=1.0)
    pipeline = ACFMNetV5Pipeline(target_epsilon=1.0, target_delta=1e-5)

    print("\nExecuting 10-Stage ACFM-Net v5 Generation Pipeline...")
    df_synthetic, report = pipeline.fit_generate(
        df_raw=df_real,
        num_samples=1000,
        target_col=target_col,
        epochs=5,
        protocol_id="IRB-ACFMV5-GEN-2026"
    )

    # Save Synthetic Data to CSV
    output_csv = os.path.join(PROJECT_ROOT, "data", "synthetic_clinical_dataset.csv")
    df_synthetic.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Synthetic Dataset generated and saved to:")
    print(f"   {os.path.abspath(output_csv)}")
    print("=" * 60)
    print(f"Synthetic Rows: {len(df_synthetic)}")
    print(f"Synthetic Columns: {len(df_synthetic.columns)}")
    print(f"Differential Privacy Epsilon Used: {pipeline.prv_accountant.accumulated_privacy_cost:.4f}")
    print(f"Pipeline Status: {report['pipeline_status']}")
    print("\nFirst 5 rows of generated synthetic data:")
    print(df_synthetic.head())

if __name__ == "__main__":
    generate_synthetic_csv()
