import pytest
import numpy as np
import pandas as pd
from acfm_net.evaluation.privacy_eval import (
    _encode_frames,
    evaluate_positive_controls,
    _compute_auc_with_ci,
    TAPASClosestDistanceMIA,
    TAPASShadowModelMIA,
    DOMIASMIA,
    DCRDistributionEvaluator,
)
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    ClassifierTwoSampleTestHarness,
    TSTRHarness,
    ClinicalConstraintEvaluator,
    SlicedWassersteinEvaluator,
)

@pytest.fixture
def sample_frames():
    rng = np.random.default_rng(42)
    n = 60
    df_train = pd.DataFrame({
        "age": rng.normal(40, 10, size=n),
        "glucose": rng.uniform(70, 150, size=n),
        "gender": rng.choice(["M", "F"], size=n),
        "outcome": rng.choice([0, 1], size=n),
    })
    df_holdout = pd.DataFrame({
        "age": rng.normal(41, 10, size=n),
        "glucose": rng.uniform(72, 148, size=n),
        "gender": rng.choice(["M", "F"], size=n),
        "outcome": rng.choice([0, 1], size=n),
    })
    df_syn = pd.DataFrame({
        "age": rng.normal(40.5, 9.5, size=n),
        "glucose": rng.uniform(71, 149, size=n),
        "gender": rng.choice(["M", "F"], size=n),
        "outcome": rng.choice([0, 1], size=n),
    })
    return df_train, df_holdout, df_syn


def test_tapas_closest_distance_mia(sample_frames):
    df_train, df_holdout, df_syn = sample_frames
    X_train, X_holdout, X_syn = _encode_frames(df_train, df_holdout, df_syn)
    evaluator = TAPASClosestDistanceMIA()
    res = evaluator.evaluate(X_train, X_holdout, X_syn, max_eval=40, seed=42)
    assert "mia_auc" in res
    assert 0.0 <= res["mia_auc"] <= 1.0
    assert "mia_95_ci" in res


def test_tapas_shadow_model_mia(sample_frames):
    df_train, df_holdout, df_syn = sample_frames
    X_train, X_holdout, X_syn = _encode_frames(df_train, df_holdout, df_syn)
    evaluator = TAPASShadowModelMIA()
    res = evaluator.evaluate(X_train, X_holdout, X_syn, max_eval=40, seed=42)
    assert "mia_auc" in res
    assert 0.0 <= res["mia_auc"] <= 1.0


def test_domias_density_ratio_mia(sample_frames):
    df_train, df_holdout, df_syn = sample_frames
    X_train, X_holdout, X_syn = _encode_frames(df_train, df_holdout, df_syn)
    evaluator = DOMIASMIA()
    res = evaluator.evaluate(X_train, X_holdout, X_syn, max_eval=40, k_neighbors=5, seed=42)
    assert "domias_auc" in res
    assert 0.0 <= res["domias_auc"] <= 1.0


def test_dcr_distribution_evaluator(sample_frames):
    df_train, df_holdout, df_syn = sample_frames
    X_train, X_holdout, X_syn = _encode_frames(df_train, df_holdout, df_syn)
    evaluator = DCRDistributionEvaluator()
    res = evaluator.evaluate(X_train, X_holdout, X_syn, num_repeats=3, seed=42)
    assert "train_to_holdout_dcr_ratio" in res
    assert res["train_to_holdout_dcr_ratio"] > 0.0


def test_positive_controls_and_noisy_copy_curve(sample_frames):
    df_train, df_holdout, _ = sample_frames
    X_train, X_holdout, _ = _encode_frames(df_train, df_holdout, df_train)
    res = evaluate_positive_controls(X_train, X_holdout, max_eval=40, seed=42)
    assert "noisy_copy_detection_curve" in res
    assert len(res["noisy_copy_detection_curve"]) > 0
    assert "controls_pass" in res


def test_marginal_evaluator_1way_and_ks(sample_frames):
    df_train, df_holdout, df_syn = sample_frames
    evaluator = MarginalErrorEvaluator()
    res = evaluator.evaluate_1way(df_train, df_syn, df_real_holdout=df_holdout, n_bins=10)
    assert "mean_1way_marginal_error_tvd" in res
    assert "mean_ks_statistic" in res
    assert res["mean_1way_marginal_error_tvd"] >= 0.0


def test_marginal_evaluator_2way(sample_frames):
    df_train, _, df_syn = sample_frames
    evaluator = MarginalErrorEvaluator()
    res = evaluator.evaluate_2way(df_train, df_syn, n_bins=5, max_pairs=6, seed=42)
    assert "mean_2way_marginal_error_tvd" in res
    assert res["mean_2way_marginal_error_tvd"] >= 0.0


def test_correlation_error_evaluator(sample_frames):
    df_train, df_holdout, df_syn = sample_frames
    evaluator = CorrelationErrorEvaluator()
    res = evaluator.evaluate(df_train, df_syn, df_real_holdout=df_holdout)
    assert "pearson_frobenius_error" in res
    assert "spearman_frobenius_error" in res
    assert res["pearson_frobenius_error"] >= 0.0


def test_c2st_evaluator(sample_frames):
    df_train, _, df_syn = sample_frames
    evaluator = ClassifierTwoSampleTestHarness()
    res = evaluator.evaluate(df_train, df_syn, n_splits=2, max_samples=40, seed=42)
    assert "c2st_auc" in res
    assert 0.0 <= res["c2st_auc"] <= 1.0


def test_swd_evaluator(sample_frames):
    df_train, _, df_syn = sample_frames
    num_cols = ["age", "glucose"]
    evaluator = SlicedWassersteinEvaluator(num_projections=20)
    res = evaluator.evaluate(df_train[num_cols].values, df_syn[num_cols].values, seed=42)
    assert "sliced_wasserstein_distance" in res
    assert res["sliced_wasserstein_distance"] >= 0.0


def test_clinical_constraint_evaluator():
    evaluator = ClinicalConstraintEvaluator()
    df = pd.DataFrame({
        "age": [25.0, 150.0],  # 1 out-of-bounds
        "systolic": [120.0, 100.0],
        "diastolic": [80.0, 110.0],  # second violates systolic > diastolic + 10
    })
    res = evaluator.evaluate(df)
    assert res["violating_records_count"] > 0
    assert "overall_record_violation_rate" in res


def test_tstr_harness(sample_frames):
    df_train, df_holdout, df_syn = sample_frames
    evaluator = TSTRHarness()
    res = evaluator.evaluate(df_train, df_holdout, df_syn, target_col="outcome", seed=42)
    assert "tstr" in res
    assert "auroc" in res["tstr"]


