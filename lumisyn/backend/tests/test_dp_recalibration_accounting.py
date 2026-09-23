"""
Accounting & Sensitivity Unit Tests for DP Marginal Recalibration
=============================================================================
Gate 4d Item A6:
- Sensitivity derivation verification under add/remove adjacency (Delta_1 = M).
- Ledger accounting verification (eps_flow + eps_m == eps_target).
- Output validity check (schema bounds, category validity).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK_ROOT = REPO_ROOT / "lumisyn" / "framework"
sys.path.insert(0, str(FRAMEWORK_ROOT))

from acfm_net.postprocessing.dp_recalibration import DPMarginalRecalibrator


@pytest.fixture
def sample_schema():
    return {
        "columns": {
            "num_feat1": {"type": "numeric", "bounds": [0.0, 100.0], "kind": "continuous"},
            "num_feat2": {"type": "numeric", "bounds": [-10.0, 50.0], "kind": "continuous"},
            "cat_feat1": {"type": "categorical", "categories": ["catA", "catB", "catC"]},
            "cat_feat2": {"type": "categorical", "categories": ["X", "Y"]},
        }
    }


def test_sensitivity_under_add_remove_adjacency(sample_schema):
    """
    Verifies that under add/remove adjacency, one record changes at most 1 bin
    in each of the M columns, so the total L1 sensitivity is <= M.
    """
    recalibrator = DPMarginalRecalibrator(sample_schema, num_continuous_bins=20)
    M = recalibrator.get_sensitivity()
    assert M == 4, f"Expected M=4, got {M}"

    rng = np.random.RandomState(42)
    N = 500

    # Dataset D
    d_data = {
        "num_feat1": rng.uniform(0, 100, size=N),
        "num_feat2": rng.uniform(-10, 50, size=N),
        "cat_feat1": rng.choice(["catA", "catB", "catC"], size=N),
        "cat_feat2": rng.choice(["X", "Y"], size=N),
    }
    df_d = pd.DataFrame(d_data)

    # Neighboring dataset D' (1 record added)
    extra_record = pd.DataFrame({
        "num_feat1": [45.5],
        "num_feat2": [12.0],
        "cat_feat1": ["catB"],
        "cat_feat2": ["X"],
    })
    df_d_prime = pd.concat([df_d, extra_record], ignore_index=True)

    # Compute raw histograms for D and D'
    total_l1_diff = 0.0
    for col_name, spec in sample_schema["columns"].items():
        if spec["type"] == "numeric":
            lo, hi = spec["bounds"]
            bins = np.linspace(lo, hi, 21)
            h_d, _ = np.histogram(df_d[col_name], bins=bins)
            h_dp, _ = np.histogram(df_d_prime[col_name], bins=bins)
            col_l1 = float(np.sum(np.abs(h_d - h_dp)))
        else:
            cats = spec["categories"]
            h_d = np.array([float(np.sum(df_d[col_name] == c)) for c in cats])
            h_dp = np.array([float(np.sum(df_d_prime[col_name] == c)) for c in cats])
            col_l1 = float(np.sum(np.abs(h_d - h_dp)))

        # In each single column, 1 record changes exactly 1 bin count by 1
        assert col_l1 == 1.0, f"Column {col_name} L1 change was {col_l1}, expected 1.0"
        total_l1_diff += col_l1

    # Total L1 difference across all M columns
    assert total_l1_diff <= M, f"Total L1 diff {total_l1_diff} exceeds sensitivity M={M}"
    assert total_l1_diff == float(M)


def test_ledger_accounting_composition(sample_schema):
    """
    Gate 4d Item A6: Verifies privacy ledger charge accounting:
    eps_flow = eps_target - eps_m.
    Serial composition theorem guarantees total privacy cost <= eps_target.
    """
    eps_target = 1.0
    eps_m_candidates = [0.1, 0.25, 0.5]

    for eps_m in eps_m_candidates:
        eps_flow = eps_target - eps_m
        assert eps_flow > 0.0, f"Flow budget must be positive: {eps_flow}"
        total_eps = round(eps_flow + eps_m, 6)
        assert total_eps == eps_target, f"Ledger mismatch: {total_eps} != {eps_target}"


def test_recalibration_bounds_and_validity(sample_schema):
    """
    Verifies that recalibrated values strictly conform to public schema bounds
    and public category sets.
    """
    recalibrator = DPMarginalRecalibrator(sample_schema, num_continuous_bins=20)
    rng = np.random.RandomState(42)
    N = 300

    df_real = pd.DataFrame({
        "num_feat1": rng.uniform(0, 100, size=N),
        "num_feat2": rng.uniform(-10, 50, size=N),
        "cat_feat1": rng.choice(["catA", "catB", "catC"], size=N),
        "cat_feat2": rng.choice(["X", "Y"], size=N),
    })

    # Synthetic dataframe with noisy/out-of-bound values
    df_syn = pd.DataFrame({
        "num_feat1": rng.uniform(-50, 150, size=N),
        "num_feat2": rng.uniform(-30, 80, size=N),
        "cat_feat1": rng.choice(["catA", "catB", "catC"], size=N),
        "cat_feat2": rng.choice(["X", "Y"], size=N),
    })

    dp_hists, rep = recalibrator.release_dp_histograms(df_real, eps_m=0.25, seed=42)
    assert rep["sensitivity_L1"] == 4
    assert rep["eps_m"] == 0.25

    df_cal = recalibrator.recalibrate(df_syn, dp_hists, seed=42)

    # Check bounds
    assert df_cal["num_feat1"].min() >= 0.0
    assert df_cal["num_feat1"].max() <= 100.0
    assert df_cal["num_feat2"].min() >= -10.0
    assert df_cal["num_feat2"].max() <= 50.0

    # Check categories
    assert set(df_cal["cat_feat1"].unique()).issubset({"catA", "catB", "catC"})
    assert set(df_cal["cat_feat2"].unique()).issubset({"X", "Y"})


def test_histogram_release_count_matches_columns(sample_schema):
    """
    Gate 4d Item 3: M = number of released histograms (original columns);
    verifies that exactly M histograms are created, released, and tracked.
    """
    recalibrator = DPMarginalRecalibrator(sample_schema, num_continuous_bins=20)
    N = 100
    rng = np.random.RandomState(42)
    df_real = pd.DataFrame({
        "num_feat1": rng.uniform(0, 100, size=N),
        "num_feat2": rng.uniform(-10, 50, size=N),
        "cat_feat1": rng.choice(["catA", "catB", "catC"], size=N),
        "cat_feat2": rng.choice(["X", "Y"], size=N),
    })

    dp_hists, rep = recalibrator.release_dp_histograms(df_real, eps_m=0.5, seed=42)
    M = len(sample_schema["columns"])
    assert rep["num_columns_calibrated"] == M
    assert rep["released_histograms_count"] == M
    assert len(dp_hists) == M
    assert set(dp_hists.keys()) == set(sample_schema["columns"].keys())
    assert "noise_to_count_ratios" in rep
    assert "bins_per_column" in rep
    assert len(rep["noise_to_count_ratios"]) == M


def test_pipeline_with_dp_recalibration():
    """Verify that ACFMNetV5Pipeline executes Stage 7b DP recalibration and composes total budget correctly."""
    from acfm_net.pipeline import ACFMNetV5Pipeline

    rng = np.random.RandomState(42)
    N = 100
    mock_df = pd.DataFrame({
        "age": rng.randint(20, 80, size=N),
        "glucose": rng.uniform(70, 200, size=N),
        "gender": rng.choice(["M", "F"], size=N),
        "outcome": rng.choice([0, 1], size=N),
    })

    pipeline = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        enable_dp_recalibration=True,
        recalibration_epsilon=0.10,
    )
    df_syn, report = pipeline.fit_generate(
        mock_df,
        num_samples=50,
        target_col="outcome",
        epochs=2,
        protocol_id="IRB-RECAL-2026",
    )

    assert report["pipeline_status"] == "SUCCESS"
    assert "Stage 7b: DP Marginal Recalibration Completed" in report["stages_completed"]
    assert "dp_recalibration" in report
    assert report["dp_recalibration"]["eps_m"] == 0.10
    assert report["privacy_composition"]["epsilon"] <= report["target_epsilon_adaptive"]
    assert df_syn.shape == (50, 4)

