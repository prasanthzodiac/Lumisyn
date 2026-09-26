import numpy as np
import pandas as pd
import pytest
from acfm_net.evaluation.fidelity import (
    MarginalErrorEvaluator,
    CorrelationErrorEvaluator,
    SlicedWassersteinEvaluator,
    TSTRHarness,
)
from acfm_net.evaluation.utility import _evaluate_model_clf
from sklearn.ensemble import RandomForestClassifier

def test_sliced_wasserstein():
    rng = np.random.RandomState(42)
    X1 = rng.randn(200, 10)
    X2 = rng.randn(200, 10)
    evaluator = SlicedWassersteinEvaluator(num_projections=50)
    res = evaluator.evaluate(X1, X2)
    assert "sliced_wasserstein_distance" in res
    assert res["sliced_wasserstein_distance"] > 0.0

def test_holdout_normalized_marginal_and_correlation():
    rng = np.random.RandomState(42)
    # Create synthetic and real datasets
    df_real = pd.DataFrame(rng.randn(200, 4), columns=["c1", "c2", "c3", "c4"])
    df_syn = pd.DataFrame(rng.randn(200, 4), columns=["c1", "c2", "c3", "c4"])
    df_holdout = pd.DataFrame(rng.randn(200, 4), columns=["c1", "c2", "c3", "c4"])

    # Marginal
    m_eval = MarginalErrorEvaluator()
    m_res = m_eval.evaluate_1way(df_real, df_syn, df_real_holdout=df_holdout)
    assert "normalized_1way_marginal_error" in m_res
    assert "real_vs_real_floor_1way_tvd" in m_res
    assert m_res["normalized_1way_marginal_error"] > 0.0

    # Correlation
    c_eval = CorrelationErrorEvaluator()
    c_res = c_eval.evaluate(df_real, df_syn, df_real_holdout=df_holdout)
    assert "normalized_pearson_frobenius_error" in c_res
    assert "real_vs_real_floor_pearson_frobenius" in c_res
    assert c_res["normalized_pearson_frobenius_error"] > 0.0

def test_no_auroc_auprc_duplication():
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5)
    y_train = rng.randint(0, 3, size=100) # multi-class
    X_test = rng.randn(50, 5)
    y_test = rng.randint(0, 3, size=50)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    res = _evaluate_model_clf(model, X_train, y_train, X_test, y_test)
    assert "roc_auc" in res
    assert "auprc" in res
    # AUROC and AUPRC should be valid numerical scores and not forcibly duplicated to f1
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["auprc"] <= 1.0
