"""Stage 7c: DP class-conditional marginals — public domain, valid output, joint accounting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))

from acfm_net.postprocessing.conditional_marginals import ConditionalMarginals  # noqa: E402
from acfm_net.privacy.budget_allocator import AdaptiveBudgetAllocator  # noqa: E402

SCHEMA = {
    "age": {"type": "numeric", "bounds": [18, 90], "kind": "continuous"},
    "job": {"type": "categorical", "categories": ["a", "b", "c"]},
    "y": {"type": "categorical", "categories": ["no", "yes"]},
}


def _data(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    return pd.DataFrame({"age": np.where(y, rng.normal(60, 5, n), rng.normal(30, 5, n)).clip(18, 90),
                         "job": np.where(y, "c", rng.choice(["a", "b"], n)), "y": np.where(y, "yes", "no")})


def test_recalibrate_restores_label_feature_link_with_valid_output():
    real = _data(4000, 0)
    syn = _data(4000, 1).sample(frac=1.0, random_state=0)
    syn["y"] = np.random.default_rng(2).permutation(syn["y"].to_numpy())  # destroy the link
    cm = ConditionalMarginals(SCHEMA, "y", list(real.columns))
    report = cm.release(real, noise_multiplier=2.0)
    assert report["l2_sensitivity"] == np.sqrt(2)
    out = cm.recalibrate(syn)
    assert set(out["job"]) <= {"a", "b", "c"} and set(out["y"]) <= {"no", "yes"}
    assert out["age"].between(18, 90).all()
    assert out.loc[out.y == "yes", "age"].mean() > out.loc[out.y == "no", "age"].mean() + 15


def test_release_noise_is_unseeded():
    real = _data(500, 0)
    cm = ConditionalMarginals(SCHEMA, "y", list(real.columns))
    cm.release(real, 1.0)
    a = cm.tables["age"].copy()
    cm.release(real, 1.0)
    assert not np.allclose(a, cm.tables["age"])


def test_joint_calibration_fills_but_never_exceeds_budget():
    alloc = AdaptiveBudgetAllocator(target_epsilon=1.0)
    z = alloc.marginal_noise_multiplier(12000, 96, 14, 0.74, 8e-6, share=0.6)
    sigma = alloc.calibrate_noise_multiplier_accountant(0.74, 512 / 12000, 120, 8e-6, extra_history=((z, 1.0, 1),))
    from acfm_net.privacy.prv_accountant import PRVAccountant
    eps = PRVAccountant(target_delta=8e-6)._epsilon_from_history([(sigma, 512 / 12000, 120), (z, 1.0, 1)])
    assert 0.70 < eps <= 0.74 + 1e-6
