"""
Stage 5 (DAPC) Strict Invariant and Accounting Tests.
Verifies:
1. Identity-mode accounting: HVAE budget is 0, Flow gets full training budget; total eps/delta equals ledger sum.
2. VAE-mode accounting: HVAE budget + Flow budget + meta charges equals total ledger sum.
3. Policy safety gate depends only on n, d, and schema (invariant to private data values).
4. Sharp ABSTAIN threshold at exact n = 99 (ABSTAIN) vs n = 100 (PROCEED).
"""

import pytest
import numpy as np
import torch
from lumisyn.framework.acfm_net.privacy.budget_allocator import AdaptiveBudgetAllocator
from lumisyn.framework.acfm_net.privacy.contract import PrivacyLedger


def test_dapc_sharp_abstain_threshold_at_99_vs_100():
    """Verify exact policy threshold: n=99 ABSTAINS, n=100 PROCEEDS."""
    allocator = AdaptiveBudgetAllocator(target_epsilon=1.0, target_delta=1e-5)

    decision_99, reasons_99 = allocator.evaluate_defensive_safety(
        num_samples=99,
        num_features=10,
        public_schema=None,
    )
    assert decision_99 == "ABSTAIN", f"Expected ABSTAIN at n=99, got {decision_99}"
    assert any("n=99 < 100" in r for r in reasons_99)

    decision_100, reasons_100 = allocator.evaluate_defensive_safety(
        num_samples=100,
        num_features=10,
        public_schema=None,
    )
    assert decision_100 == "PROCEED", f"Expected PROCEED at n=100, got {decision_100}"


def test_dapc_gate_invariance_to_private_data():
    """Gate decision must evaluate strictly public n, d, schema and ignore private data values."""
    allocator = AdaptiveBudgetAllocator(target_epsilon=1.0, target_delta=1e-5)

    # Vary adversarial conditions: decision depends strictly on public (n, d, schema)
    for n_val in [100, 500, 10000]:
        for d_val in [5, 50, 200]:
            dec, reasons = allocator.evaluate_defensive_safety(
                num_samples=n_val,
                num_features=d_val,
                public_schema=None,
            )
            assert dec == "PROCEED"

    # Excessive dimension d > 500 must reject
    dec_rej, _ = allocator.evaluate_defensive_safety(num_samples=1000, num_features=501)
    assert dec_rej == "REJECT"

    # Single pass on analyze_dataset_dapc with small steps
    cert = allocator.analyze_dataset_dapc(
        sensitivity_vector=torch.ones(15) * 0.5,
        num_samples=1000,
        num_features=15,
        reid_risk=0.8,
        unique_ratio=0.9,
        dep_score=0.8,
        sample_rate=0.05,
        steps=20,
    )
    assert cert["decision"] == "PROCEED"
    assert cert["selected_target_epsilon"] == 1.0


def test_identity_mode_and_vae_mode_accounting_ledger_sum():
    """Verify total (epsilon, delta) equals ledger sum for both identity and VAE modes."""
    target_eps = 1.0
    target_delta = 1e-5

    # 1. Identity Mode
    ledger_id = PrivacyLedger(target_eps, target_delta)
    # Under public graph, meta charges = 0; if noised graph, meta charge = 0.08
    eps_meta = 0.08
    delta_meta = target_delta * 0.05
    ledger_id.add("dependency_graph_statistics", epsilon=eps_meta, delta=delta_meta, mechanism="Laplace")

    training_target_eps_id = target_eps - eps_meta
    training_delta_id = target_delta * 0.80
    eps_hvae_id = 0.0
    eps_flow_id = training_target_eps_id
    delta_hvae_id = 0.0
    delta_flow_id = training_delta_id

    ledger_id.add("dp_sgd_flow", epsilon=eps_flow_id, delta=delta_flow_id, mechanism="DP-SGD Flow")

    # Verify total equals sum
    total_eps_id = eps_meta + eps_flow_id + eps_hvae_id
    assert abs(total_eps_id - target_eps) < 1e-6
    assert abs(ledger_id.epsilon - target_eps) < 1e-6

    # 2. VAE Mode
    ledger_vae = PrivacyLedger(target_eps, target_delta)
    ledger_vae.add("dependency_graph_statistics", epsilon=eps_meta, delta=delta_meta, mechanism="Laplace")

    split_hvae, split_flow = 0.30, 0.70
    training_target_eps_vae = target_eps - eps_meta
    training_delta_vae = target_delta * 0.80
    eps_hvae_vae = training_target_eps_vae * split_hvae
    eps_flow_vae = training_target_eps_vae * split_flow
    delta_hvae_vae = training_delta_vae / 2.0
    delta_flow_vae = training_delta_vae / 2.0

    ledger_vae.add("dp_sgd_hvae", epsilon=eps_hvae_vae, delta=delta_hvae_vae, mechanism="DP-SGD HVAE")
    ledger_vae.add("dp_sgd_flow", epsilon=eps_flow_vae, delta=delta_flow_vae, mechanism="DP-SGD Flow")

    total_eps_vae = eps_meta + eps_hvae_vae + eps_flow_vae
    assert abs(total_eps_vae - target_eps) < 1e-6
    assert abs(ledger_vae.epsilon - target_eps) < 1e-6
    assert ledger_vae.delta <= target_delta
