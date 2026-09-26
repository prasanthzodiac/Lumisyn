import pytest
import torch
import torch.nn as nn
from acfm_net.privacy.dpsgd import DPSGDWrapper

def test_dpsgd_chunked_equality_against_loop():
    torch.manual_seed(123)
    model = nn.Sequential(
        nn.Linear(15, 30),
        nn.SiLU(),
        nn.Linear(30, 15)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    wrapper = DPSGDWrapper(
        model=model,
        optimizer=opt,
        max_grad_norm=0.5,
        noise_multiplier=0.0, # Zero noise to verify exact deterministic clipped gradient equality
        target_epsilon=1.0,
        target_delta=1e-5,
        expected_batch_size=300
    )

    x = torch.randn(300, 15)
    y = torch.randn(300, 15)
    out = model(x)
    losses = ((out - y)**2).mean(dim=1)

    # 1. Compute via exact manual sequential loop first
    params = list(model.parameters())
    loop_grads = [torch.zeros_like(p) for p in params]
    loop_norms = []
    for i in range(300):
        s_grads = torch.autograd.grad(losses[i], params, retain_graph=True, allow_unused=True)
        norm = (sum(g.pow(2).sum() for g in s_grads)).sqrt().item()
        norm = max(1e-8, norm)
        loop_norms.append(norm)
        coef = min(1.0, 0.5 / norm)
        for acc, g in zip(loop_grads, s_grads):
            acc.add_(g * coef)
    loop_grads = [g / 300.0 for g in loop_grads]

    # 2. Compute via DPSGDWrapper (which uses chunked autograd)
    wrapper.step_microbatch(losses, retain_graph=False)
    chunked_grads = [p.grad.clone() for p in model.parameters()]

    for cg, lg in zip(chunked_grads, loop_grads):
        max_err = torch.max(torch.abs(cg - lg)).item()
        assert max_err < 1e-5, f"Grad error too high: {max_err}"

    assert len(wrapper.last_sample_norms) == 300
    assert len(wrapper.last_clipped_norms) == 300
    for cn in wrapper.last_clipped_norms:
        assert cn <= 0.5 + 1e-4

    # Check that sample norms match within float precision
    for sn_wrap, sn_loop in zip(wrapper.last_sample_norms, loop_norms):
        assert abs(sn_wrap - sn_loop) < 1e-5
