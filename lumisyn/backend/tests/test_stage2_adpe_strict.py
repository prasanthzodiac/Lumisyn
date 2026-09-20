"""
Stage 2 (ADPE) Strict Invariant and Dimension Consistency Tests.
Verifies:
1. Encoded dimension d_enc = 78 is invariant to removing rare categories from raw data.
2. Missing indicator columns (*_isna) are derived strictly from public schema (not empirical missingness).
3. Removing all NaNs from data preserves the exact d_enc = 78 encoded dimension.
4. Encode-decode round trip recovers real values within tolerance for continuous, integer, and categorical features.
5. Missing-value indicators restore NaN upon decoding.
6. Unseen categories cleanly map to an unknown level ('MISSING') without throwing errors or changing dimension.
7. Optional public schema-level log1p transform operates strictly behind enable_schema_log1p flag (class c).
"""

import os
import yaml
import pytest
import numpy as np
import pandas as pd

from lumisyn.framework.acfm_net.preparation.encoding import DistributionPreservingEncoder
from lumisyn.framework.acfm_net.preparation.imputation import AdaptiveImputer


@pytest.fixture
def adult_schema():
    schema_path = "lumisyn/data/schemas/adult_schema.yaml"
    assert os.path.exists(schema_path), "adult_schema.yaml not found"
    with open(schema_path, "r") as f:
        data = yaml.safe_load(f)
    return data["columns"]


@pytest.fixture
def sample_adult_df():
    data_path = "lumisyn/data/kaggle_datasets/adult.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path).head(1000).copy()
    else:
        # Synthetic fallback with exact adult schema columns
        df = pd.DataFrame({
            "age": [39, 50, 38, 53],
            "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
            "fnlwgt": [77516, 83311, 215646, 234721],
            "education": ["Bachelors", "Bachelors", "HS-grad", "11th"],
            "education-num": [13, 13, 9, 7],
            "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
            "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Handlers-cleaners"],
            "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband"],
            "race": ["White", "White", "White", "Black"],
            "sex": ["Male", "Male", "Male", "Male"],
            "capital-gain": [2174, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40],
            "native-country": ["United-States", "United-States", "United-States", "United-States"],
            "class": ["<=50K", "<=50K", "<=50K", "<=50K"],
        })
    return df


def test_d_enc_invariance_on_category_removal(sample_adult_df, adult_schema):
    """Removing rare categories from raw data must not alter d_enc (fixed at 78)."""
    df_a = sample_adult_df.copy()
    # Remove rare categories in df_b
    df_b = sample_adult_df[
        (sample_adult_df["occupation"] != "Armed-Forces") &
        (sample_adult_df["workclass"] != "Without-pay")
    ].reset_index(drop=True)

    imp = AdaptiveImputer()
    df_imp_a, _ = imp.fit_transform(df_a, privacy_mode=True, public_schema=adult_schema)
    df_imp_b, _ = imp.fit_transform(df_b, privacy_mode=True, public_schema=adult_schema)

    enc_a = DistributionPreservingEncoder()
    mat_a, info_a = enc_a.fit_transform(df_imp_a, privacy_schema=adult_schema)

    enc_b = DistributionPreservingEncoder()
    mat_b, info_b = enc_b.fit_transform(df_imp_b, privacy_schema=adult_schema)

    assert mat_a.shape[1] == 78, f"Expected 78 encoded dimensions for adult schema, got {mat_a.shape[1]}"
    assert mat_b.shape[1] == 78, f"Encoded dimension changed after removing rare categories: {mat_b.shape[1]}"
    assert info_a["encoded_column_names"] == info_b["encoded_column_names"]


def test_d_enc_invariance_when_nan_completely_absent(sample_adult_df, adult_schema):
    """Even if a dataset has zero NaNs, schema-derived missingness indicators keep d_enc at 78."""
    # Create complete dataset without any NaNs
    df_complete = sample_adult_df.dropna().copy()

    imp = AdaptiveImputer()
    df_imp, _ = imp.fit_transform(df_complete, privacy_mode=True, public_schema=adult_schema)

    # Indicator columns for nullable schema features must still be generated as zeros
    assert "workclass_isna" in df_imp.columns
    assert "occupation_isna" in df_imp.columns
    assert "native-country_isna" in df_imp.columns

    enc = DistributionPreservingEncoder()
    mat, info = enc.fit_transform(df_imp, privacy_schema=adult_schema)
    assert mat.shape[1] == 78, f"Expected 78 dimensions on NaN-free dataset, got {mat.shape[1]}"


def test_unseen_categories_map_to_unknown_level(sample_adult_df, adult_schema):
    """Unseen categories (e.g. 'Martian' or novel category) must map to 'MISSING' without error."""
    df_unseen = sample_adult_df.copy()
    # Inject unseen categories
    df_unseen.loc[0, "workclass"] = "Intergalactic-Corp"
    df_unseen.loc[0, "native-country"] = "Atlantis"

    imp = AdaptiveImputer()
    df_imp, _ = imp.fit_transform(df_unseen, privacy_mode=True, public_schema=adult_schema)

    enc = DistributionPreservingEncoder()
    mat, info = enc.fit_transform(df_imp, privacy_schema=adult_schema)

    # Must retain exact dimension 78
    assert mat.shape[1] == 78

    # Decode and check that unseen categories mapped to 'MISSING'
    df_dec = enc.decode(mat)
    assert df_dec.loc[0, "workclass"] == "MISSING"
    assert df_dec.loc[0, "native-country"] == "MISSING"


def test_encode_decode_roundtrip_tolerance(sample_adult_df, adult_schema):
    """Encode-decode round trip must restore non-missing values within tolerance."""
    df_clean = sample_adult_df.dropna().copy().reset_index(drop=True)
    imp = AdaptiveImputer()
    df_imp, _ = imp.fit_transform(df_clean, privacy_mode=True, public_schema=adult_schema)

    enc = DistributionPreservingEncoder()
    mat, info = enc.fit_transform(df_imp, privacy_schema=adult_schema)
    df_dec = enc.decode(mat)

    # Check continuous / integer recovery
    for col in ["age", "education-num", "hours-per-week"]:
        diff = np.abs(df_dec[col].values - df_clean[col].values)
        assert np.max(diff) < 1.0, f"Roundtrip error on {col} exceeded tolerance: max diff = {np.max(diff)}"

    # Check categorical recovery
    for col in ["workclass", "education", "marital-status", "relationship", "race", "sex"]:
        accuracy = np.mean(df_dec[col].values == df_clean[col].values)
        assert accuracy >= 0.99, f"Categorical recovery on {col} was below 99%: {accuracy:.2%}"


def test_missing_value_indicators_restore_nan(sample_adult_df, adult_schema):
    """When *_isna is 1, decoder must restore NaN in the primary feature."""
    imp = AdaptiveImputer()
    df_imp, _ = imp.fit_transform(sample_adult_df.copy(), privacy_mode=True, public_schema=adult_schema)
    enc = DistributionPreservingEncoder()
    mat, info = enc.fit_transform(df_imp, privacy_schema=adult_schema)

    # Manually toggle an indicator column to 1.0
    isna_idx = info["encoded_column_names"].index("workclass_isna")
    mat_test = mat.copy()
    mat_test[0, isna_idx] = 1.0
    mat_test[1, isna_idx] = 0.0

    df_dec = enc.decode(mat_test)
    assert pd.isna(df_dec.loc[0, "workclass"]), "Row with workclass_isna=1 must decode to NaN"
    assert not pd.isna(df_dec.loc[1, "workclass"]), "Row with workclass_isna=0 must not decode to NaN"


def test_log1p_flag_controls_transformation(sample_adult_df, adult_schema):
    """Optional schema-level log1p transform must be inactive by default, active when flag is True."""
    enc_default = DistributionPreservingEncoder(enable_schema_log1p=False)
    _, info_def = enc_default.fit_transform(sample_adult_df.copy(), privacy_schema=adult_schema)
    assert info_def["feature_specs"]["capital-gain"]["transform"] is None

    enc_active = DistributionPreservingEncoder(enable_schema_log1p=True)
    _, info_act = enc_active.fit_transform(sample_adult_df.copy(), privacy_schema=adult_schema)
    assert info_act["feature_specs"]["capital-gain"]["transform"] == "log1p"


def test_enable_schema_log1p_default_false_and_recorded_in_config():
    """Verify enable_schema_log1p defaults to False and is recorded in every run's config and checkpoint."""
    from lumisyn.framework.acfm_net.pipeline import ACFMNetV5Pipeline

    # Default encoder instance must have enable_schema_log1p == False
    enc = DistributionPreservingEncoder()
    assert enc.enable_schema_log1p is False

    # Pipeline run must record enable_schema_log1p in released_checkpoint config and privacy contract
    n = 105
    df_tiny = pd.DataFrame({
        "age": np.random.randint(20, 70, size=n),
        "income": np.random.uniform(20000, 100000, size=n),
    })
    pipe = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        allow_nonprivate_debug=True,
        batch_size=105,
    )
    df_syn, report = pipe.fit_generate(df_tiny, num_samples=10, epochs=1)

    assert "config" in pipe.released_checkpoint
    assert pipe.released_checkpoint["config"]["enable_schema_log1p"] is False
    assert report["config"]["enable_schema_log1p"] is False
    assert report["privacy_contract"]["enable_schema_log1p"] is False

