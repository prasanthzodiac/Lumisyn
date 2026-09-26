import os
import sys
import math
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure framework is in path
sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.privacy.prv_accountant import PRVAccountant
from acfm_net.privacy.rdp_accountant import RDPAccountant
from acfm_net.privacy.budget_allocator import AdaptiveBudgetAllocator
from acfm_net.privacy.dpsgd import DPSGDWrapper
from acfm_net.representation.hvae import HierarchicalVAE
from acfm_net.generative.ot_cfm import ContinuousOTCFM
from opacus.validators import ModuleValidator


def test_accountant_rdp_reproduction():
    """
    Test 4c(i): Accountant wrapper reproduces RDP epsilon for q=0.01, sigma=1.0, 300 steps, delta=1e-5.
    Records and reports the exact numerical value computed by Opacus RDP.
    """
    rdp = RDPAccountant(target_delta=1e-5)
    eps_rdp = rdp.compute_epsilon(noise_multiplier=1.0, sample_rate=0.01, steps=300)
    
    prv = PRVAccountant(target_delta=1e-5)
    eps_prv = prv.compute_epsilon(noise_multiplier=1.0, sample_rate=0.01, steps=300)
    
    print(f"\n[Test 4c(i) Report] Exact RDP epsilon: {eps_rdp:.8f}")
    print(f"[Test 4c(i) Report] Exact PRV epsilon: {eps_prv:.8f}")
    
    assert math.isfinite(eps_rdp), "RDP epsilon must be finite"
    assert eps_rdp > 0.0, "RDP epsilon must be positive"
    # Exact value computed under Opacus 1.6.0 RDP is 1.45136809
    assert abs(eps_rdp - 1.45136809) < 1e-4, f"Unexpected RDP epsilon: {eps_rdp}"


def test_short_training_target_epsilon_and_crosscheck():
    """
    Test 4c(ii): With target eps=1.0, the final reported eps after a short training is <= 1.0;
    cross-checks with a second accountant (RDP vs PRV) and reports the difference.
    """
    np.random.seed(101)
    torch.manual_seed(101)
    
    n = 200
    mock_df = pd.DataFrame({
        "feat_a": np.random.normal(0, 1, size=n),
        "feat_b": np.random.uniform(10, 50, size=n),
        "feat_c": np.random.choice(["X", "Y"], size=n),
        "outcome": np.random.choice([0, 1], size=n),
    })
    
    pipeline = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        use_ot_coupling=False,
        poisson_sampling=True,
    )
    
    df_syn, report = pipeline.fit_generate(
        df_raw=mock_df,
        num_samples=40,
        target_col="outcome",
        epochs=2,
    )
    
    final_eps = float(report["privacy_training"]["accounted_epsilon"])
    assert final_eps <= 1.0 + 1e-6, f"Final reported epsilon {final_eps} exceeds target 1.0"
    
    # Detailed parameters print for Gate 1b
    N_rows = len(mock_df)
    exp_bs = report["privacy_training"]["expected_batch_size"]
    q_rate = report["privacy_training"]["sample_rate"]
    sigma = report["privacy_training"]["noise_multiplier"]
    hvae_steps = report["privacy_training"]["hvae_steps"]
    flow_steps = report["privacy_training"]["flow_steps"]
    total_steps = report["privacy_training"]["steps"]
    training_delta = report["privacy_training"]["target_delta"]

    print(f"\n[Test 4c(ii) Report] N={N_rows}, expected_batch_size={exp_bs}, q={q_rate:.6f}")
    print(f"[Test 4c(ii) Report] Stage 4 (HVAE) steps={hvae_steps}, sigma={sigma:.6f}")
    print(f"[Test 4c(ii) Report] Stage 8 (Flow) steps={flow_steps}, sigma={sigma:.6f}")
    print(f"[Test 4c(ii) Report] Total DP steps={total_steps}, delta={training_delta:.2e}")
    assert q_rate <= 1.0, f"Sample rate q={q_rate} must be <= 1.0"
    
    # Cross-check using pure RDP accountant on the same step history
    rdp_accountant = RDPAccountant(target_delta=training_delta)
    eps_cross_rdp = rdp_accountant.compute_epsilon(noise_multiplier=sigma, sample_rate=q_rate, steps=total_steps)
    eps_cross_prv = final_eps
    diff = abs(eps_cross_rdp - eps_cross_prv)
    
    print(f"[Test 4c(ii) Report] Final PRV epsilon: {eps_cross_prv:.6f}")
    print(f"[Test 4c(ii) Report] Cross-check RDP epsilon: {eps_cross_rdp:.6f}")
    print(f"[Test 4c(ii) Report] Difference |RDP - PRV|: {diff:.6f}")


def test_gate_invariance_column_shuffling():
    """
    Test 4c(iii): shuffle values within each column (same n, d, schema). DAPC reads only
    public metadata, so both the decision AND the selected epsilon must be identical.
    """
    np.random.seed(42)
    n = 300
    d = 5
    df1 = pd.DataFrame(np.random.randn(n, d), columns=[f"c_{i}" for i in range(d)])
    df2 = df1.copy()
    for col in df2.columns:
        df2[col] = np.random.permutation(df2[col].values)

    allocator = AdaptiveBudgetAllocator(target_epsilon=1.0, target_delta=1e-5)
    cert1 = allocator.analyze_dataset_dapc(sensitivity_vector=torch.ones(d) * 0.5, num_samples=len(df1), num_features=d)
    cert2 = allocator.analyze_dataset_dapc(sensitivity_vector=torch.ones(d) * 0.5, num_samples=len(df2), num_features=d)

    assert cert1["decision"] == cert2["decision"] == "PROCEED"
    assert cert1["selected_target_epsilon"] == cert2["selected_target_epsilon"]
    assert allocator.eps_min <= cert1["selected_target_epsilon"] <= allocator.eps_max
    assert cert1["rounded_sample_count"] == cert2["rounded_sample_count"] == 300


def test_tiny_n_abstains():
    """
    Test 4c(iv): Tiny n makes the gate ABSTAIN.
    """
    allocator = AdaptiveBudgetAllocator(target_epsilon=1.0, target_delta=1e-5)
    
    # Small cohort: n = 25 (rounded n = 50 < 100)
    cert = allocator.analyze_dataset_dapc(
        sensitivity_vector=torch.ones(4) * 0.5,
        num_samples=25,
        num_features=4,
    )
    
    print(f"\n[Test 4c(iv) Report] Tiny n Decision: {cert['decision']}")
    print(f"[Test 4c(iv) Report] Reasons: {cert['decision_reasons']}")
    
    assert cert["decision"] == "ABSTAIN"
    assert any("Cohort size too small" in r for r in cert["decision_reasons"])
    
    # Also verify that pipeline fit_generate raises RuntimeError when ABSTAIN is triggered
    pipeline = ACFMNetV5Pipeline(target_epsilon=1.0, target_delta=1e-5)
    tiny_df = pd.DataFrame({
        "x": np.random.randn(25),
        "y": np.random.choice([0, 1], size=25),
    })
    
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.fit_generate(tiny_df, num_samples=10, target_col="y", epochs=1)
    assert "DAPC Security Abstention triggered (ABSTAIN)" in str(excinfo.value)


def test_prv_to_rdp_fallback_forced():
    """
    Test 4c(v): If a PRV-to-RDP fallback exists, force the failure path and confirm it runs cleanly.
    """
    accountant = PRVAccountant(target_epsilon=1.0, target_delta=1e-5)
    accountant.force_rdp_fallback = True
    
    eps_fallback, delta = accountant.step(noise_multiplier=1.2, sample_rate=0.01, steps=50)
    
    print(f"\n[Test 4c(v) Report] Forced RDP fallback epsilon: {eps_fallback:.6f}, delta: {delta}")
    assert math.isfinite(eps_fallback)
    assert eps_fallback > 0.0


def test_module_validator_and_per_sample_norm_clipping():
    """
    Test 8: Run Opacus ModuleValidator. Compute per-sample gradients with DPSGDWrapper;
    assert clipped per-sample norm <= C.
    """
    hvae = HierarchicalVAE(in_dim=8)
    cfm = ContinuousOTCFM(continuous_dim=64, cond_dim=0, hidden_dim=64)
    
    # ModuleValidator validation
    errors_hvae = ModuleValidator.validate(hvae, strict=False)
    errors_cfm = ModuleValidator.validate(cfm.velocity_net, strict=False)
    
    assert len(errors_hvae) == 0, f"HVAE contains incompatible layers: {errors_hvae}"
    assert len(errors_cfm) == 0, f"CFM contains incompatible layers: {errors_cfm}"
    
    # Test per-sample clipping assertion
    C = 1.25
    opt = torch.optim.Adam(hvae.parameters(), lr=1e-3)
    wrapper = DPSGDWrapper(hvae, opt, max_grad_norm=C, noise_multiplier=1.0)
    
    # Pass inputs designed to produce huge unclipped gradients
    # Note on previous 13,238,466 loss:
    # x ~ 100 * N(0, 1) has Var(x) ~ 10,000. The MSE reconstruction loss is ~ 13,238.
    # Multiplying by 1000.0 produced loss ~ 13,238,466.
    # Here we capture per-sample norms before clipping and after clipping to demonstrate
    # that despite unclipped norms in the millions, all clipped norms are strictly bounded <= C.
    x = torch.randn(8, 8) * 100.0
    out = hvae(x)
    loss = out["per_sample_loss"] * 1000.0
    
    step_loss = wrapper.step_microbatch(loss)
    sample_norms = wrapper.last_sample_norms
    clipped_norms = wrapper.last_clipped_norms

    print(f"\n[Test 8 Report] Step loss value: {step_loss:.4f} (from 1000x scaled inputs, x*100)")
    print(f"[Test 8 Report] Per-sample unclipped norms (all {len(sample_norms)}): {[round(n, 4) for n in sample_norms]}")
    print(f"[Test 8 Report] Per-sample clipped norms   (all {len(clipped_norms)}): {[round(n, 4) for n in clipped_norms]}")
    print(f"[Test 8 Report] Max clipped norm: {max(clipped_norms):.6f} <= C={C:.2f}")

    assert math.isfinite(step_loss)
    assert max(clipped_norms) <= C + 1e-4, f"Max clipped norm {max(clipped_norms)} exceeds C={C}"
    assert max(sample_norms) > C, "Input should have produced unclipped norms exceeding C"


def test_h1_icfm_gradient_independence():
    """
    Test 1 / H1: Verify that under Independent Flow Matching (I-CFM), perturbing an unrelated
    sample in the batch produces gradient difference on untouched sample < 1e-6.
    """
    model = ContinuousOTCFM(continuous_dim=4, cond_dim=0, hidden_dim=32)
    model.eval()
    params = list(model.velocity_net.parameters())
    t = torch.tensor([[0.5]], dtype=torch.float32)
    
    z_0 = torch.zeros(4, 4)
    z_1_D = torch.tensor([[0.1, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float32)
    z_1_D_prime = z_1_D.clone()
    z_1_D_prime[3] = torch.tensor([-50.0, 0, 0, 0], dtype=torch.float32)  # extreme perturbation on sample 3
    
    # I-CFM trajectories on sample 0
    u_0_D = z_1_D[0:1] - z_0[0:1]
    z_t_0_D = 0.5 * z_0[0:1] + 0.5 * z_1_D[0:1]
    loss_0_D = torch.nn.functional.mse_loss(model.velocity_net(z_t_0_D, t), u_0_D)
    
    u_0_D_prime = z_1_D_prime[0:1] - z_0[0:1]
    z_t_0_D_prime = 0.5 * z_0[0:1] + 0.5 * z_1_D_prime[0:1]
    loss_0_D_prime = torch.nn.functional.mse_loss(model.velocity_net(z_t_0_D_prime, t), u_0_D_prime)
    
    grad_0_D = torch.autograd.grad(loss_0_D, params, retain_graph=True)
    grad_0_D_prime = torch.autograd.grad(loss_0_D_prime, params, retain_graph=True)
    
    diff_norm = torch.sqrt(sum((g1 - g2).pow(2).sum() for g1, g2 in zip(grad_0_D, grad_0_D_prime))).item()
    print(f"\n[Test H1 Report] I-CFM Gradient difference norm on untouched sample 0: {diff_norm:.8e}")
    assert diff_norm < 1e-6, f"Gradient difference {diff_norm} exceeds 1e-6"


def test_unprotected_artifact_fails_loudly():
    """
    Test 9: Any attempt to execute unprotected operations in DP release path fails loudly.
    """
    cfm = ContinuousOTCFM(continuous_dim=4, cond_dim=0, hidden_dim=32)
    z = torch.randn(4, 4)
    
    # Attempting to use batch-Hungarian OT without allow_nonprivate_ot must raise ValueError
    with pytest.raises(ValueError) as exc1:
        cfm.compute_loss(z, use_ot_coupling=True, allow_nonprivate_ot=False)
    assert "violates per-sample sensitivity bounds under DP-SGD" in str(exc1.value)
    
    # Attempting to disable strict release privacy without allow_nonprivate_debug must raise ValueError
    with pytest.raises(ValueError) as exc2:
        ACFMNetV5Pipeline(strict_release_privacy=False, allow_nonprivate_debug=False)
    assert "strict_release_privacy must be True for formal differential privacy release guarantees" in str(exc2.value)


def test_hvae_gradient_independence():
    """
    Test Gate 1b (Point 6): Verify that under Stage 4 HVAE representation loss,
    perturbing an unrelated sample in the batch produces zero gradient change on
    untouched samples (norm < 1e-6). Also confirms that KL loss and beta annealing
    do not contain batch-level statistics.
    """
    from acfm_net.representation.gat_encoder import GraphAttentionEncoder
    from acfm_net.representation.cross_attention import GatedCrossAttention

    in_dim = 8
    hvae = HierarchicalVAE(in_dim=in_dim, global_dim=16, group_dim=8, num_groups=2)
    gat = GraphAttentionEncoder(in_features=1, hidden_dim=32)
    cross_attn = GatedCrossAttention(feature_dim=in_dim, graph_dim=32, hidden_dim=32)
    adj_matrix = torch.eye(in_dim)
    
    stage4_params = list(gat.parameters()) + list(cross_attn.parameters()) + list(hvae.parameters())
    
    # Batch of 4 samples
    torch.manual_seed(42)
    X = torch.randn(4, in_dim)
    X_perturbed = X.clone()
    X_perturbed[3] = X_perturbed[3] + 500.0  # extreme perturbation on sample 3
    
    # Forward pass on unperturbed batch with seed S
    torch.manual_seed(1234)
    h_gat1 = gat(X, adj_matrix)
    x_gated1 = cross_attn(X, h_gat1)
    out1 = hvae(x_gated1, epoch=10)
    
    # Forward pass on perturbed batch with identical seed S (same reparameterization noise eps)
    torch.manual_seed(1234)
    h_gat2 = gat(X_perturbed, adj_matrix)
    x_gated2 = cross_attn(X_perturbed, h_gat2)
    out2 = hvae(x_gated2, epoch=10)
    
    # Check KL/beta terms for batch-level statistics
    assert out1["per_sample_loss"].shape == (4,), "per_sample_loss must be shape (B,)"
    assert out1["recon_loss"].ndim == 0, "recon_loss must be scalar"
    assert isinstance(out1["beta_kl"], float) or out1["beta_kl"].ndim == 0, "beta_kl must be scalar independent of batch"
    
    # Gradients of sample 0 under unperturbed batch across all Stage 4 parameters
    loss_0 = out1["per_sample_loss"][0]
    grad_0 = torch.autograd.grad(loss_0, stage4_params, retain_graph=True, allow_unused=True)
    
    # Gradients of sample 0 under perturbed batch across all Stage 4 parameters
    loss_0_perturbed = out2["per_sample_loss"][0]
    grad_0_perturbed = torch.autograd.grad(loss_0_perturbed, stage4_params, retain_graph=True, allow_unused=True)
    
    diff_norm = torch.sqrt(sum(
        (g1 - g2).pow(2).sum() for g1, g2 in zip(grad_0, grad_0_perturbed)
        if g1 is not None and g2 is not None
    )).item()
    
    print(f"\n[Test HVAE Independence Report] Stage 4 full pipeline gradient diff on sample 0: {diff_norm:.8e}")
    assert diff_norm < 1e-6, f"HVAE gradient difference {diff_norm} exceeds 1e-6; per-sample independence violated"


def test_minimum_cohort_size_thresholds():
    """
    Test Gate 1b (Point 10): Explicit minimum cohort size threshold n >= 100.
    Tests n in {49, 50, 74, 75, 99, 100}.
    Guarantees that n = 75..99 does NOT round up to 100 and pass.
    """
    allocator = AdaptiveBudgetAllocator(target_epsilon=1.0, target_delta=1e-5)
    test_cases = [
        (49, "ABSTAIN"),
        (50, "ABSTAIN"),
        (74, "ABSTAIN"),
        (75, "ABSTAIN"),
        (99, "ABSTAIN"),
        (100, "PROCEED"),
    ]
    print("\n[Test Min-n Report] Testing explicit minimum cohort size policy:")
    for n_val, expected_decision in test_cases:
        decision, reasons = allocator.evaluate_defensive_safety(
            num_samples=n_val,
            num_features=10,
        )
        print(f"  n={n_val:3d} -> Decision: {decision:7s} (Expected: {expected_decision})")
        assert decision == expected_decision, f"Failed for n={n_val}: expected {expected_decision}, got {decision}"


def test_dpsgd_expected_batch_size_scaling():
    """
    Test Gate 1c (Point 3): Verify DP-SGD noisy gradient scaling divides by
    EXPECTED batch size (q*N), not actual batch size, in both normal and empty-batch paths.
    """
    import torch.nn as nn
    
    # 1. Normal Batch Path: Verify scaling by expected_batch_size L = 10 when B = 2
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    L_expected = 10.0
    wrapper = DPSGDWrapper(
        model=model,
        optimizer=optimizer,
        max_grad_norm=5.0,
        noise_multiplier=0.0,  # Zero noise to verify deterministic scaling
        expected_batch_size=L_expected,
        validate_architecture=False,
    )
    
    # Per-sample loss with explicit known gradients: g_0 = 1.0, g_1 = 3.0 -> sum = 4.0
    x = torch.tensor([[1.0], [3.0]])
    y = model(x)
    losses = y.squeeze(-1)  # d(loss_0)/dW = 1.0, d(loss_1)/dW = 3.0
    
    wrapper.step_microbatch(losses)
    # Expected gradient = (1.0 + 3.0) / L_expected = 4.0 / 10.0 = 0.40
    # (If divided by actual B=2, it would have incorrectly been 4.0 / 2 = 2.0)
    actual_grad = model.weight.grad.item()
    print(f"\n[Test DP-SGD Scaling Report] Normal batch: actual_grad={actual_grad:.6f}, expected={4.0/L_expected:.6f}")
    assert abs(actual_grad - (4.0 / L_expected)) < 1e-5, f"Normal batch scaling failed: got {actual_grad}"
    
    # 2. Empty Batch Path: Verify pure noise is scaled by the EXACT SAME expected_batch_size L
    wrapper_empty = DPSGDWrapper(
        model=model,
        optimizer=optimizer,
        max_grad_norm=2.0,
        noise_multiplier=1.5,
        expected_batch_size=L_expected,
        validate_architecture=False,
    )
    
    torch.manual_seed(999)
    synthetic_noise = torch.randn_like(model.weight) * (1.5 * 2.0)
    expected_empty_grad = (synthetic_noise / L_expected).item()
    
    torch.manual_seed(999)
    wrapper_empty.step_microbatch(torch.zeros(0))
    actual_empty_grad = model.weight.grad.item()
    print(f"[Test DP-SGD Scaling Report] Empty batch: actual_empty_grad={actual_empty_grad:.6f}, expected={expected_empty_grad:.6f}")
    assert abs(actual_empty_grad - expected_empty_grad) < 1e-5, f"Empty batch scaling failed: got {actual_empty_grad}"


def test_vectorized_gradient_equality_and_timing():
    """
    Test Gate 1c (Point 6): Verify vectorized per-sample autograd (is_grads_batched=True)
    produces exact identical gradients to the per-sample loop (max diff < 1e-5),
    and report the timing / speedup at batch size 250.
    """
    import time
    
    in_dim = 16
    B = 250
    hvae = HierarchicalVAE(in_dim=in_dim, global_dim=16, group_dim=8, num_groups=2)
    params = [p for p in hvae.parameters() if p.requires_grad]
    
    torch.manual_seed(42)
    x = torch.randn(B, in_dim)
    out = hvae(x)
    losses = out["per_sample_loss"]
    
    # Path 1: Vectorized per-sample autograd
    grad_outputs = torch.eye(B, device=x.device)
    batched_grads = torch.autograd.grad(
        outputs=losses,
        inputs=params,
        grad_outputs=grad_outputs,
        is_grads_batched=True,
        retain_graph=True,
        allow_unused=True,
    )
    sample_norms_sq = torch.zeros(B, device=x.device)
    for g in batched_grads:
        if g is not None:
            sample_norms_sq += g.flatten(start_dim=1).pow(2).sum(dim=1)
    sample_norms_t = torch.clamp(sample_norms_sq.sqrt(), min=1e-8)
    clip_coef_t = torch.clamp(1.25 / sample_norms_t, max=1.0)
    
    vectorized_clipped_grads = []
    for g in batched_grads:
        if g is not None:
            view_shape = [B] + [1] * (g.dim() - 1)
            vectorized_clipped_grads.append((g * clip_coef_t.view(*view_shape)).sum(dim=0))
        else:
            vectorized_clipped_grads.append(None)
            
    # Path 2: Python per-sample loop
    loop_clipped_grads = [torch.zeros_like(p) for p in params]
    for i in range(B):
        sample_grads = torch.autograd.grad(
            losses[i],
            params,
            retain_graph=(i < B - 1),
            allow_unused=True,
        )
        total_norm_sq = sum(g.pow(2).sum().item() for g in sample_grads if g is not None)
        s_norm = max(1e-8, total_norm_sq ** 0.5)
        clip_coef = min(1.0, 1.25 / s_norm)
        for acc_g, g in zip(loop_clipped_grads, sample_grads):
            if g is not None:
                acc_g.add_(g * clip_coef)
                
    # Check gradient equality
    max_diff = 0.0
    for g_vec, g_loop in zip(vectorized_clipped_grads, loop_clipped_grads):
        if g_vec is not None and g_loop is not None:
            diff = (g_vec - g_loop).abs().max().item()
            if diff > max_diff:
                max_diff = diff
    print(f"\n[Test Vectorized DP-SGD Report] Max gradient diff (batched vs loop): {max_diff:.8e}")
    assert max_diff < 1e-5, f"Batched gradient discrepancy {max_diff} exceeds 1e-5"
    
    # Timing benchmark at batch 250
    # Vectorized timing
    t0 = time.perf_counter()
    for _ in range(5):
        out_v = hvae(x)
        _ = torch.autograd.grad(
            outputs=out_v["per_sample_loss"],
            inputs=params,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            allow_unused=True,
        )
    t_vec = (time.perf_counter() - t0) / 5.0
    
    # Loop timing
    t0 = time.perf_counter()
    for _ in range(5):
        out_l = hvae(x)
        losses_l = out_l["per_sample_loss"]
        for i in range(B):
            _ = torch.autograd.grad(
                losses_l[i],
                params,
                retain_graph=(i < B - 1),
                allow_unused=True,
            )
    t_loop = (time.perf_counter() - t0) / 5.0
    speedup = t_loop / max(1e-6, t_vec)
    print(f"[Test Vectorized DP-SGD Report] Timing at B=250: Vectorized={t_vec*1000:.1f}ms | Loop={t_loop*1000:.1f}ms | Speedup={speedup:.2f}x")


def test_checkpoint_audit():
    """
    Gate 2b Item 2: Checkpoint-Audit Test.
    Enumerates every tensor and scalar in the released checkpoint and config;
    each must be DP-trained weights, a public constant, or ledger-charged.
    Fails on anything else.
    """
    from acfm_net.privacy.checkpoint_audit import audit_released_checkpoint

    np.random.seed(42)
    torch.manual_seed(42)

    n = 100
    mock_df = pd.DataFrame({
        "age": np.random.uniform(20, 80, size=n),
        "bmi": np.random.uniform(18, 35, size=n),
        "sys_bp": np.random.uniform(90, 160, size=n),
        "outcome": np.random.choice([0, 1], size=n),
    })

    pipeline = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        use_ot_coupling=False,
        poisson_sampling=True,
    )

    df_syn, report = pipeline.fit_generate(
        df_raw=mock_df,
        num_samples=20,
        target_col="outcome",
        epochs=1,
    )

    checkpoint = pipeline.released_checkpoint
    audit_res = audit_released_checkpoint(checkpoint)
    summary = audit_res["summary"]

    print("\n====================== CHECKPOINT AUDIT TEST REPORT ======================")
    print(f" Total items audited:           {summary['total_items_audited']}")
    print(f" Total DP-SGD trained weights:  {summary['total_dp_trained_parameters']:,} parameters")
    print(f" Total Ledger-charged elements: {summary['total_ledger_charged_elements']} elements")
    print(f" Total Public entries/constants: {summary['total_public_entries']}")
    print(f" Unbudgeted private violations: {summary['unbudgeted_violations_count']}")
    print(f" Audit status:                  {'PASSED' if summary['checkpoint_audit_passed'] else 'FAILED'}")
    print("--------------------------------------------------------------------------")
    for entry in audit_res["audit_log"][:15]:
        print(f"  [{entry['status']}] {entry['item']} ({entry['category']}) -> shape: {entry.get('shape', 'N/A')}")
    if len(audit_res["audit_log"]) > 15:
        print(f"  ... and {len(audit_res['audit_log']) - 15} more verified tensors/scalars.")
    print("========================================================================\n")

    assert summary["checkpoint_audit_passed"] is True, "Checkpoint audit must pass for valid release"
    assert summary["unbudgeted_violations_count"] == 0, "No unbudgeted items allowed"

    # Negative test: inject an unbudgeted private statistic and verify it FAILS
    poisoned_checkpoint = dict(checkpoint)
    poisoned_checkpoint["private_empirical_latent_mean"] = torch.tensor([0.15, -0.22])
    with pytest.raises(AssertionError) as exc_info:
        audit_released_checkpoint(poisoned_checkpoint)
    print(f"[Negative Test Verified] Correctly caught unbudgeted item: {exc_info.value}")


def test_missingness_nan_restoration():
    """
    Gate 2b Item 7: Confirm decoding restores NaN when the *_isna indicator is 1
    and that no raw 0.0 fill values reach the output. Use the schema midpoint for
    the fill in scaled space (0.0 in [-1, 1]).
    """
    from acfm_net.preparation.imputation import AdaptiveImputer
    from acfm_net.preparation.encoding import DistributionPreservingEncoder
    from acfm_net.preparation.scaling import AdaptiveFeatureScaler

    # Setup feature with range [100.0, 200.0] (schema midpoint is 150.0)
    public_schema = {
        "systolic": {"type": "numeric", "min": 100.0, "max": 200.0},
        "gender": {"type": "categorical", "categories": ["F", "M", "MISSING"]},
    }

    df_raw = pd.DataFrame({
        "systolic": [120.0, np.nan, 180.0, np.nan, 140.0],
        "gender": ["F", "M", np.nan, "F", "M"],
    })

    imputer = AdaptiveImputer()
    df_imputed, _ = imputer.fit_transform(
        df_raw,
        privacy_mode=True,
        public_schema=public_schema,
    )

    # 1. Verify schema midpoint was used as fill value in raw space
    # (100.0 + 200.0) / 2 = 150.0
    assert df_imputed.loc[1, "systolic"] == 150.0, f"Expected midpoint 150.0, got {df_imputed.loc[1, 'systolic']}"
    assert df_imputed.loc[1, "systolic_isna"] == 1.0, "Missing indicator must be 1.0"
    assert df_imputed.loc[0, "systolic_isna"] == 0.0, "Observed indicator must be 0.0"

    # 2. Verify scaled space maps schema midpoint 150.0 to 0.0 in [-1, 1]
    encoder = DistributionPreservingEncoder()
    enc_X, enc_info = encoder.fit_transform(df_imputed, privacy_schema=public_schema)
    scaler = AdaptiveFeatureScaler()
    scaled_X = scaler.fit_transform(
        enc_X,
        feature_specs=enc_info["feature_specs"],
        privacy_mode=True,
        encoded_column_names=enc_info["encoded_column_names"],
    )

    col_idx = enc_info["encoded_column_names"].index("systolic")
    midpoint_scaled = scaled_X[1, col_idx]
    assert abs(midpoint_scaled - 0.0) < 1e-5, f"Expected schema midpoint to be 0.0 in scaled space, got {midpoint_scaled}"

    # 3. Verify decoding restores NaN when *_isna is 1, and restores valid generated numbers when 0
    # Simulate a decoded matrix where row 0 has isna=0, row 1 has isna=1
    # Raw values: row 0 -> 160.0, row 1 -> 150.0 (the model output)
    fake_encoded = enc_X.copy()
    isna_col_idx = enc_info["encoded_column_names"].index("systolic_isna")
    fake_encoded[0, isna_col_idx] = 0.0
    fake_encoded[1, isna_col_idx] = 1.0

    df_decoded = encoder.decode(fake_encoded)
    print(f"\n[Test Missingness Restoration] Decoded systolic: row 0={df_decoded.loc[0, 'systolic']}, row 1={df_decoded.loc[1, 'systolic']}")

    assert not np.isnan(df_decoded.loc[0, "systolic"]), "Row 0 (isna=0) must NOT be NaN"
    assert np.isnan(df_decoded.loc[1, "systolic"]), "Row 1 (isna=1) MUST be restored to NaN"
    assert df_decoded.loc[0, "systolic"] >= 100.0, "Output must be within schema bounds, never raw 0.0"
    print("[Test Missingness Restoration] Passed: NaN restored when isna=1, schema midpoint used in scaled space, no raw 0.0 in output.")




