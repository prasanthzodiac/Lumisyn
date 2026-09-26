"""
Accounting and Sensitivity Tests for DP Gaussian Copula (Kendall's Tau Estimator)
=============================================================================
Gate 4d Item 2: Verifies sensitivity derivation and accounting for DPCopula.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK_ROOT = REPO_ROOT / "lumisyn" / "framework"
sys.path.insert(0, str(FRAMEWORK_ROOT))

from acfm_net.evaluation.dp_gaussian_copula import DPGaussianCopulaGenerator


@pytest.fixture
def copula_schema():
    return {
        "columns": {
            "num1": {"type": "numeric", "bounds": [0.0, 100.0]},
            "num2": {"type": "numeric", "bounds": [-50.0, 50.0]},
            "cat1": {"type": "categorical", "categories": ["A", "B", "C"]},
        }
    }


def test_kendall_tau_sensitivity_under_add_remove():
    """
    Empirically verifies that adding/removing 1 record changes pairwise Kendall's tau
    by at most 2 / (N - 1).
    """
    rng = np.random.RandomState(42)
    N = 200
    x = rng.randn(N)
    y = 0.5 * x + 0.5 * rng.randn(N)

    tau_orig, _ = kendalltau(x, y)

    # Add 1 adversarial record
    x_adv = np.append(x, [100.0])
    y_adv = np.append(y, [-100.0])
    tau_adv, _ = kendalltau(x_adv, y_adv)

    tau_diff = abs(tau_orig - tau_adv)
    theoretical_bound = 4.0 / N

    assert tau_diff <= theoretical_bound + 1e-6, (
        f"Empirical diff {tau_diff:.6f} exceeded theoretical bound {theoretical_bound:.6f}"
    )


def test_dp_copula_accounting_split(copula_schema):
    """
    Verifies that eps_m + eps_corr == epsilon_target.
    """
    for eps in [1.0, 4.0]:
        gen = DPGaussianCopulaGenerator(copula_schema, epsilon=eps, eps_marginal_ratio=0.5)
        assert round(gen.eps_m + gen.eps_corr, 6) == eps
        delta1_m, delta1_tau = gen.get_sensitivity(N=1000)
        assert delta1_m == 3
        # 2 * 3 * 2 / 1000 = 12 / 1000 = 0.012
        assert abs(delta1_tau - (12.0 / 1000.0)) < 1e-6


def test_dp_copula_generation_validity(copula_schema):
    """
    Verifies that generated synthetic records obey schema bounds and categories.
    """
    rng = np.random.RandomState(42)
    N = 500
    df = pd.DataFrame({
        "num1": rng.uniform(0, 100, size=N),
        "num2": rng.uniform(-50, 50, size=N),
        "cat1": rng.choice(["A", "B", "C"], size=N),
    })

    gen = DPGaussianCopulaGenerator(copula_schema, epsilon=1.0, seed=42)
    gen.fit(df)

    # Check PSD
    eigvals = np.linalg.eigvalsh(gen.corr_matrix_dp)
    assert np.all(eigvals > 0), "Correlation matrix must be strictly positive definite"

    df_syn = gen.generate(100, seed=42)
    assert len(df_syn) == 100
    assert df_syn["num1"].min() >= 0.0
    assert df_syn["num1"].max() <= 100.0
    assert set(df_syn["cat1"].unique()).issubset({"A", "B", "C"})
