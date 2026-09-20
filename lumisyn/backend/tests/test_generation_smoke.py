import pytest
import os
import tempfile
import numpy as np
import pandas as pd
import yaml
import torch
from acfm_net.pipeline import ACFMNetV5Pipeline

def test_generation_path_end_to_end_smoke():
    """
    Generation path smoke test:
    Asserts that CSV output from a 2-epoch tiny run has valid schema,
    categories in the allowed set, values in public bounds, missing indicators respected,
    and no NaN. Under 30 seconds.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Construct tiny dataset (N=120 >= 100 for DAPC safety gating)
    n = 120
    age = np.random.randint(18, 80, size=n).astype(float)
    chol = np.random.uniform(120.0, 300.0, size=n)
    category = np.random.choice(["A", "B", "C"], size=n)

    # Add a few NaNs to chol to test missing indicator handling
    chol[10] = np.nan
    chol[25] = np.nan

    df_raw = pd.DataFrame({
        "age": age,
        "cholesterol": chol,
        "category": category
    })

    # Public schema
    public_schema = {
        "columns": [
            {"name": "age", "type": "numeric", "bounds": [18.0, 100.0]},
            {"name": "cholesterol", "type": "numeric", "bounds": [100.0, 350.0], "nullable": True},
            {"name": "category", "type": "categorical", "allowed_values": ["A", "B", "C"]}
        ]
    }

    pipe = ACFMNetV5Pipeline(
        target_epsilon=2.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        allow_nonprivate_debug=True,
        public_schema=public_schema,
        batch_size=120,
        disable_gat=True,
    )

    # 2. Fit and generate (2 epochs)
    df_synthetic, report = pipe.fit_generate(df_raw, num_samples=40, epochs=2)

    # 3. Export to CSV in temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
        tmp_csv_path = tmp_file.name

    try:
        df_synthetic.to_csv(tmp_csv_path, index=False)

        # Load back from CSV to test full export roundtrip
        df_csv = pd.read_csv(tmp_csv_path)

        # Assertion 1: Valid schema (matching raw columns)
        expected_cols = ["age", "cholesterol", "category"]
        assert list(df_csv.columns) == expected_cols, f"Schema mismatch: {list(df_csv.columns)} vs {expected_cols}"
        assert len(df_csv) == 40

        # Assertion 2: Categories strictly in the allowed set
        allowed_cats = {"A", "B", "C", "MISSING"}
        assert set(df_csv["category"].unique()).issubset(allowed_cats), (
            f"Generated categories {set(df_csv['category'].unique())} contain unallowed levels outside {allowed_cats}"
        )

        # Assertion 3: Numeric values strictly in public bounds
        assert (df_csv["age"] >= 18.0).all() and (df_csv["age"] <= 100.0).all(), (
            f"Age out of bounds: min={df_csv['age'].min()}, max={df_csv['age'].max()}"
        )
        assert (df_csv["cholesterol"] >= 100.0).all() and (df_csv["cholesterol"] <= 350.0).all(), (
            f"Cholesterol out of bounds: min={df_csv['cholesterol'].min()}, max={df_csv['cholesterol'].max()}"
        )

        # Assertion 4: No NaN anywhere in the output
        assert not df_csv.isna().any().any(), "Generated CSV contains unexpected NaNs!"

        # Assertion 5: Checkpoint and metadata verification
        assert pipe.released_checkpoint is not None
        assert "config" in pipe.released_checkpoint

    finally:
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
