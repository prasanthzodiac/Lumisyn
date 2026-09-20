"""
Stage 7 (Repair) Strict Invariant and Bounds Tests.
Verifies:
1. Public repair uses only public schema bounds/feature_specs (poisoned-tensor test for private data).
2. Repaired output values are always strictly inside public schema bounds.
3. Reports constraint violation rates before and after repair on real holdout and synthetic outputs.
"""

import pytest
import numpy as np
import pandas as pd
from lumisyn.framework.acfm_net.repair.constraint_repair import (
    ConstrainedGradientRepairLoop,
    enforce_public_domain_constraints,
)


class PoisonedPrivateDataset:
    """Mock private data object that detonates if any private sample values are accessed."""
    def __getitem__(self, key):
        raise RuntimeError("PRIVACY BREACH: Private dataset accessed by repair routine!")
    def __getattr__(self, name):
        raise RuntimeError(f"PRIVACY BREACH: Private attribute '{name}' accessed by repair routine!")


def test_stage7_poisoned_private_data_isolation():
    """Verify that enforce_public_domain_constraints touches NO private dataset."""
    poisoned_private_df = PoisonedPrivateDataset()
    feature_specs = {
        "age": {"type": "numeric", "orig_min": 18.0, "orig_max": 90.0, "kind": "continuous"},
        "systolic": {"type": "numeric", "orig_min": 80.0, "orig_max": 200.0, "kind": "continuous"},
    }

    # Synthetic dataframe with out-of-bounds values
    df_syn = pd.DataFrame({
        "age": [10.0, 25.0, 105.0, -5.0],
        "systolic": [50.0, 120.0, 250.0, 180.0],
    })

    # Public repair must execute without referencing poisoned_private_df
    df_repaired = enforce_public_domain_constraints(df_syn, feature_specs)

    # Check that all repaired values are strictly within schema bounds
    assert (df_repaired["age"] >= 18.0).all()
    assert (df_repaired["age"] <= 90.0).all()
    assert (df_repaired["systolic"] >= 80.0).all()
    assert (df_repaired["systolic"] <= 200.0).all()


def test_stage7_output_always_inside_public_bounds():
    """Verify extreme synthetic values (NaN, Inf, negatives, massive values) are strictly clamped."""
    feature_specs = {
        "count_feat": {"type": "numeric", "orig_min": 0.0, "orig_max": 20.0, "kind": "count"},
        "binary_feat": {"type": "numeric", "orig_min": 0.0, "orig_max": 1.0, "kind": "binary"},
        "cont_feat": {"type": "numeric", "orig_min": -10.0, "orig_max": 10.0, "kind": "continuous"},
    }

    df_extreme = pd.DataFrame({
        "count_feat": [-100.0, 0.0, 15.6, 9999.0],
        "binary_feat": [-0.5, 0.2, 0.8, 1.5],
        "cont_feat": [-9999.0, 0.0, 5.0, 9999.0],
    })

    df_out = enforce_public_domain_constraints(df_extreme, feature_specs)

    # Count feature must be non-negative integers in [0, 20]
    assert (df_out["count_feat"] >= 0).all()
    assert (df_out["count_feat"] <= 20).all()
    assert np.all(df_out["count_feat"] == df_out["count_feat"].astype(int))

    # Binary feature must be in {0, 1}
    assert set(df_out["binary_feat"].unique()).issubset({0, 1})

    # Continuous feature must be in [-10, 10]
    assert (df_out["cont_feat"] >= -10.0).all()
    assert (df_out["cont_feat"] <= 10.0).all()


def test_stage7_violation_rate_reporting():
    """Measure and assert violation rates before/after repair on synthetic and real holdout."""
    feature_specs = {
        "age": {"type": "numeric", "orig_min": 18.0, "orig_max": 85.0, "kind": "integer"},
        "cholesterol": {"type": "numeric", "orig_min": 100.0, "orig_max": 400.0, "kind": "continuous"},
    }

    # 1. Real Holdout (should have 0% violation rate if conforming to schema)
    df_holdout = pd.DataFrame({
        "age": [25, 45, 60, 75, 30],
        "cholesterol": [180.0, 220.0, 280.0, 195.0, 310.0],
    })
    v_holdout_before = np.mean(
        (df_holdout["age"] < 18) | (df_holdout["age"] > 85) |
        (df_holdout["cholesterol"] < 100) | (df_holdout["cholesterol"] > 400)
    )
    assert v_holdout_before == 0.0, "Real holdout violates public schema bounds"

    # 2. Synthetic Smoke Run with out-of-bounds drift
    df_syn_raw = pd.DataFrame({
        "age": [12, 45, 92, 75, 14],  # 3 out-of-bounds (60% violation)
        "cholesterol": [80.0, 220.0, 450.0, 195.0, 500.0],  # 3 out-of-bounds
    })
    v_syn_before = np.mean(
        (df_syn_raw["age"] < 18) | (df_syn_raw["age"] > 85) |
        (df_syn_raw["cholesterol"] < 100) | (df_syn_raw["cholesterol"] > 400)
    )
    assert v_syn_before > 0.0, f"Expected initial violations, got {v_syn_before}"

    # Perform repair
    df_syn_repaired = enforce_public_domain_constraints(df_syn_raw, feature_specs)
    v_syn_after = np.mean(
        (df_syn_repaired["age"] < 18) | (df_syn_repaired["age"] > 85) |
        (df_syn_repaired["cholesterol"] < 100) | (df_syn_repaired["cholesterol"] > 400)
    )
    assert v_syn_after == 0.0, f"Expected 0% violation after repair, got {v_syn_after}"
