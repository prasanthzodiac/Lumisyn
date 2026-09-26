import torch
import numpy as np
from acfm_net.privacy import PRVAccountant, AdaptiveBudgetAllocator

def test_dp_budget_allocation_bound():
    target_eps = 1.0
    target_delta = 1e-5
    
    allocator = AdaptiveBudgetAllocator(target_epsilon=target_eps, target_delta=target_delta)
    sens_vec = torch.tensor([0.9, 0.5, 0.2, 0.8], dtype=torch.float32)
    fingerprint = torch.randn(256)

    eps_per_feat, noise_scales, info = allocator(sens_vec, fingerprint, epoch=20)
    
    # Assert formal sum constraint Σ ε_i == ε*
    assert abs(info["allocated_sum"] - target_eps) < 1e-4

def test_prv_accountant():
    accountant = PRVAccountant(target_epsilon=1.0, target_delta=1e-5)
    eps_cost, delta = accountant.step(noise_multiplier=1.2, sample_rate=0.01)
    assert eps_cost > 0.0
    assert not accountant.is_budget_exceeded()
