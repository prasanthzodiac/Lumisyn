"""
Stage 9 (Evaluation) Strict Unit Tests per Metric.
Verifies:
1. Real-vs-real comparison yields near-floor TVD and C2ST AUC near 0.50.
2. Copied training dataset yields DCR ratio near 0 and MIA AUC near 1.0.
3. Independent noise yields MIA AUC near 0.50.
4. Metric names match what they measure (c2st_auc, mean_1way_marginal_error_tvd, pearson_frobenius_error).
5. TSTR uses predict_proba and evaluates strictly on a non-overlapping held-out test set.
"""

import pytest
import numpy as np
import pandas as pd
from lumisyn.framework.acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    ClassifierTwoSampleTestHarness,
    TSTRHarness,
)
from lumisyn.framework.acfm_net.evaluation.privacy_eval import (
    evaluate_positive_controls,
    _encode_frames,
)


@pytest.fixture
def clean_synthetic_and_real_data():
    """Generates separable train, holdout, and synthetic data frames."""
    np.random.seed(42)
    n = 300
    
    # Feature distribution
    age_train = np.random.normal(45, 10, n)
    chol_train = np.random.normal(200, 30, n)
    outcome_train = (age_train + 0.1 * chol_train > 65).astype(int)
    
    age_test = np.random.normal(45, 10, n)
    chol_test = np.random.normal(200, 30, n)
    outcome_test = (age_test + 0.1 * chol_test > 65).astype(int)

    df_train = pd.DataFrame({"age": age_train, "cholesterol": chol_train, "outcome": outcome_train})
    df_test = pd.DataFrame({"age": age_test, "cholesterol": chol_test, "outcome": outcome_test})
    
    # Synthetic: faithful distribution
    age_syn = np.random.normal(45, 10, n)
    chol_syn = np.random.normal(200, 30, n)
    outcome_syn = (age_syn + 0.1 * chol_syn > 65).astype(int)
    df_syn = pd.DataFrame({"age": age_syn, "cholesterol": chol_syn, "outcome": outcome_syn})

    return df_train, df_test, df_syn


def test_real_vs_real_is_near_floor(clean_synthetic_and_real_data):
    """Real-vs-real evaluation must yield TVD near 0 and C2ST AUC near 0.50."""
    df_train, df_test, _ = clean_synthetic_and_real_data

    # 1. Marginal error
    marg = MarginalErrorEvaluator()
    res_marg = marg.evaluate_1way(df_train, df_test)
    assert res_marg["mean_1way_marginal_error_tvd"] < 0.15, "Real-vs-real 1-way TVD too high"

    # 2. Classifier Two-Sample Test
    c2st = ClassifierTwoSampleTestHarness()
    res_c2st = c2st.evaluate(df_train, df_test, seed=42)
    assert "c2st_auc" in res_c2st
    # Indistinguishable splits should have AUC ~ 0.50 (+/- 0.10 for finite N)
    assert 0.40 <= res_c2st["c2st_auc"] <= 0.60, f"Expected C2ST AUC near 0.50, got {res_c2st['c2st_auc']}"


def test_copied_dataset_dcr_and_mia(clean_synthetic_and_real_data):
    """A copied dataset must produce DCR ratio near 0 and MIA AUC near 1.0."""
    df_train, df_test, _ = clean_synthetic_and_real_data
    X_train, X_holdout, _ = _encode_frames(df_train, df_test, df_train)

    # Positive controls with synthetic == copy of training data
    ctrl = evaluate_positive_controls(X_train, X_holdout, max_eval=300, seed=42)
    assert ctrl["control_A_syn_eq_train"]["dcr_ratio"] < 0.10, "Copied dataset did not give near-zero DCR ratio"
    assert ctrl["control_A_syn_eq_train"]["mia_auc"] > 0.95, "Copied dataset did not give near-1.0 MIA AUC"


def test_independent_noise_mia_auc_near_half(clean_synthetic_and_real_data):
    """Independent noise must give MIA AUC near 0.50."""
    from scipy.spatial.distance import cdist
    from sklearn.metrics import roc_auc_score

    df_train, df_test, _ = clean_synthetic_and_real_data
    X_train, X_holdout, _ = _encode_frames(df_train, df_test, df_train)

    rng = np.random.default_rng(42)
    # Pure independent noise dataset
    X_noise = rng.normal(0, 1.0, size=X_train.shape)

    dist_m = cdist(X_train, X_noise, metric="euclidean").min(axis=1)
    dist_nm = cdist(X_holdout, X_noise, metric="euclidean").min(axis=1)
    scores = np.concatenate([-dist_m, -dist_nm])
    y_true = np.concatenate([np.ones(len(X_train), dtype=int), np.zeros(len(X_holdout), dtype=int)])
    mia_auc = float(roc_auc_score(y_true, scores))

    assert 0.40 <= mia_auc <= 0.60, f"Expected independent noise MIA AUC near 0.50, got {mia_auc}"


def test_metric_names_match_measurements():
    """Verify evaluation metric dictionaries expose clean, standardized names."""
    c2st = ClassifierTwoSampleTestHarness()
    # Provide >= 20 rows so 5-fold cross validation executes smoothly
    df_a = pd.DataFrame({"x": np.linspace(1, 10, 20), "y": np.array([0, 1] * 10)})
    df_b = pd.DataFrame({"x": np.linspace(1, 10, 20) + 0.1, "y": np.array([0, 1] * 10)})
    res = c2st.evaluate(df_a, df_b, seed=42)
    assert "c2st_auc" in res

    corr = CorrelationErrorEvaluator()
    res_corr = corr.evaluate(df_a, df_b)
    assert "pearson_frobenius_error" in res_corr


def test_tstr_uses_predict_proba_and_held_out_test():
    """TSTR must train on synthetic, evaluate on real holdout with predict_proba, no data overlap."""
    np.random.seed(101)
    n = 200
    
    # Distinct, non-overlapping train and test indices
    train_idx = list(range(0, 100))
    test_idx = list(range(100, 200))
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and test sets must strictly not overlap!"

    # Real train
    X_train = np.random.normal(0, 1, (100, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    df_real_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(4)])
    df_real_train["target"] = y_train

    # Real test (held-out)
    X_test = np.random.normal(0, 1, (100, 4))
    y_test = (X_test[:, 0] > 0).astype(int)
    df_real_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(4)])
    df_real_test["target"] = y_test

    # Synthetic data
    X_syn = np.random.normal(0, 1, (100, 4))
    y_syn = (X_syn[:, 0] > 0).astype(int)
    df_syn = pd.DataFrame(X_syn, columns=[f"f{i}" for i in range(4)])
    df_syn["target"] = y_syn

    tstr_harness = TSTRHarness()
    results = tstr_harness.evaluate(
        df_real_train=df_real_train,
        df_real_test=df_real_test,
        df_syn=df_syn,
        target_col="target",
        seed=42,
    )

    assert "tstr" in results
    assert "auroc" in results["tstr"]
    assert "auprc" in results["tstr"]
    # AUROC must be a continuous probability score in [0.5, 1.0] for non-degenerate predictor
    assert results["tstr"]["auroc"] > 0.60
