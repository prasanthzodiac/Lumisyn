import os
import yaml
import pytest
import pandas as pd
import numpy as np

from acfm_net.preparation.encoding import DistributionPreservingEncoder
from acfm_net.preparation.imputation import AdaptiveImputer

def test_schema_dimension_invariance_on_category_removal():
    """
    Gate 4c Item 5: Removing a rare category from the raw data
    must NOT change the encoded dimension d_enc when public schema is used.
    """
    schema_path = "lumisyn/data/schemas/adult_schema.yaml"
    assert os.path.exists(schema_path), "adult_schema.yaml must exist"

    with open(schema_path, "r") as f:
        schema_dict = yaml.safe_load(f)["columns"]

    data_path = "lumisyn/data/kaggle_datasets/adult.csv"
    assert os.path.exists(data_path), "adult.csv must exist"

    df_full = pd.read_csv(data_path)
    df_base = df_full.sample(n=5000, random_state=42).reset_index(drop=True)

    # Dataset A: Normal subset
    imp_a = AdaptiveImputer()
    df_imp_a, _ = imp_a.fit_transform(df_base.copy(), None, privacy_mode=True, public_schema=schema_dict)
    enc_a = DistributionPreservingEncoder()
    enc_X_a, info_a = enc_a.fit_transform(df_imp_a, privacy_schema=schema_dict)
    d_enc_a = enc_X_a.shape[1]

    # Dataset B: Remove rare category completely (e.g. 'Armed-Forces' from occupation and 'Without-pay' from workclass)
    df_removed = df_base[
        (df_base["occupation"] != "Armed-Forces") &
        (df_base["workclass"] != "Without-pay") &
        (df_base["education"] != "Preschool")
    ].reset_index(drop=True)

    # Verify that the rare categories are completely absent in df_removed
    assert "Armed-Forces" not in df_removed["occupation"].values
    assert "Without-pay" not in df_removed["workclass"].values
    assert "Preschool" not in df_removed["education"].values

    imp_b = AdaptiveImputer()
    df_imp_b, _ = imp_b.fit_transform(df_removed.copy(), None, privacy_mode=True, public_schema=schema_dict)
    enc_b = DistributionPreservingEncoder()
    enc_X_b, info_b = enc_b.fit_transform(df_imp_b, privacy_schema=schema_dict)
    d_enc_b = enc_X_b.shape[1]

    # Both encoded representations MUST have the exact same column dimension and column names
    assert d_enc_a == d_enc_b, f"Encoded dimension changed from {d_enc_a} to {d_enc_b} after category removal!"
    assert info_a["encoded_column_names"] == info_b["encoded_column_names"], "Encoded column names diverged!"
    assert d_enc_a == 78, f"Expected 78 encoded dimensions for adult schema, got {d_enc_a}"
