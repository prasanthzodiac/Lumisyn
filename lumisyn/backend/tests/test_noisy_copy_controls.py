import pytest
import numpy as np
from acfm_net.evaluation.privacy_eval import evaluate_positive_controls

def test_noisy_copy_positive_controls_detection_curve():
    np.random.seed(42)
    N = 200
    D = 10
    X_train = np.random.uniform(-1.0, 1.0, size=(N, D)).astype(np.float32)
    X_holdout = np.random.uniform(-1.0, 1.0, size=(N, D)).astype(np.float32)

    res = evaluate_positive_controls(X_train, X_holdout, max_eval=200, seed=42)

    assert res["controls_pass"] is True
    assert "noisy_copy_detection_curve" in res

    curve = res["noisy_copy_detection_curve"]
    assert len(curve) == 7

    # Check monotonic behavior / detection curve characteristics:
    # At low noise (0.001), DCR ratio is near 0 and MIA AUC is near 1.0
    first = curve[0]
    assert first["noise_sigma"] == 0.001
    assert first["dcr_ratio"] < 0.2
    assert first["mia_auc"] > 0.85

    # As noise increases to 1.0, DCR ratio approaches ~1.0 and MIA AUC drops towards ~0.5
    last = curve[-1]
    assert last["noise_sigma"] == 1.0
    assert last["dcr_ratio"] > first["dcr_ratio"]
    assert last["mia_auc"] < first["mia_auc"]
