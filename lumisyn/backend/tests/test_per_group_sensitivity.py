import pytest
import torch
import numpy as np

def test_per_group_clipping_sensitivity():
    """
    Empirical test verifying that per-group gradient clipping satisfies
    global L2 sensitivity Delta_2 <= sqrt(sum_g C_g^2).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Define two groups with different dimensions and clip thresholds
    C_1 = 0.5
    C_2 = 1.2
    C_eff = np.sqrt(C_1**2 + C_2**2)

    dim_1 = 50
    dim_2 = 100
    batch_size = 32

    # Simulate random unclipped per-sample gradients
    g1 = torch.randn(batch_size, dim_1) * 10.0
    g2 = torch.randn(batch_size, dim_2) * 10.0

    # Clip group 1
    norms1 = torch.norm(g1, p=2, dim=1, keepdim=True)
    clip_factors1 = torch.clamp(C_1 / (norms1 + 1e-9), max=1.0)
    g1_clipped = g1 * clip_factors1

    # Clip group 2
    norms2 = torch.norm(g2, p=2, dim=1, keepdim=True)
    clip_factors2 = torch.clamp(C_2 / (norms2 + 1e-9), max=1.0)
    g2_clipped = g2 * clip_factors2

    # Verify per-sample bounds
    assert torch.all(torch.norm(g1_clipped, p=2, dim=1) <= C_1 + 1e-5)
    assert torch.all(torch.norm(g2_clipped, p=2, dim=1) <= C_2 + 1e-5)

    # Neighboring batch B' differs by exactly 1 sample at index k
    k = 7
    g1_prime = g1.clone()
    g2_prime = g2.clone()
    g1_prime[k] = torch.randn(dim_1) * 50.0  # arbitrary different sample
    g2_prime[k] = torch.randn(dim_2) * 50.0

    norms1_p = torch.norm(g1_prime, p=2, dim=1, keepdim=True)
    g1_clipped_p = g1_prime * torch.clamp(C_1 / (norms1_p + 1e-9), max=1.0)

    norms2_p = torch.norm(g2_prime, p=2, dim=1, keepdim=True)
    g2_clipped_p = g2_prime * torch.clamp(C_2 / (norms2_p + 1e-9), max=1.0)

    # Compute concatenated difference in sum
    sum_diff_1 = torch.sum(g1_clipped, dim=0) - torch.sum(g1_clipped_p, dim=0)
    sum_diff_2 = torch.sum(g2_clipped, dim=0) - torch.sum(g2_clipped_p, dim=0)
    total_diff = torch.cat([sum_diff_1, sum_diff_2])
    l2_sensitivity = torch.norm(total_diff, p=2).item()

    # The maximum possible change by replacing 1 sample is 2 * C_eff (if replace) or C_eff (if add/remove)
    # Here replacing 1 sample: diff is (g_k - g'_k), so norm is <= ||g_k|| + ||g'_k|| <= 2 * C_eff
    assert l2_sensitivity <= 2.0 * C_eff + 1e-5, f"L2 sensitivity {l2_sensitivity} exceeded 2*C_eff {2*C_eff}"

    # For add/remove 1 sample (pure DP neighbor):
    diff_remove_1 = g1_clipped[k]
    diff_remove_2 = g2_clipped[k]
    remove_diff = torch.cat([diff_remove_1, diff_remove_2])
    remove_l2 = torch.norm(remove_diff, p=2).item()
    assert remove_l2 <= C_eff + 1e-5, f"Add/remove sensitivity {remove_l2} exceeded C_eff {C_eff}"

    print(f"Per-group clipping empirical test passed: Add/remove sensitivity={remove_l2:.4f} <= C_eff={C_eff:.4f}")
