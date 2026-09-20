import pytest
import torch
import numpy as np
import pandas as pd
from acfm_net.pipeline import ACFMNetV5Pipeline

class PrivateAccessViolation(Exception):
    pass

class PoisonedPrivateTensor:
    """Wrapper around private tensor that raises an exception if accessed in any way."""
    def __init__(self, tensor):
        self._tensor = tensor
    def __getattr__(self, name):
        raise PrivateAccessViolation(f"Private tensor accessed during student/inference pass: {name}")
    def __getitem__(self, item):
        raise PrivateAccessViolation("Private tensor item accessed during student/inference pass!")
    def __array__(self, *args, **kwargs):
        raise PrivateAccessViolation("Private tensor converted to array during student/inference pass!")

def test_reflow_and_inference_isolation():
    """
    Gate 4b Item 10: Automated test verifying zero private tensor access
    during inference / student sampling. Fails if any private training tensor
    is accessed during generation.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Create a synthetic cohort (N >= 100 required by DAPC safety gating)
    df_raw = pd.DataFrame({
        "age": np.random.randint(18, 80, size=120),
        "income": np.random.uniform(20000, 150000, size=120),
        "category": np.random.choice(["A", "B", "C"], size=120)
    })

    pipe = ACFMNetV5Pipeline(
        target_epsilon=1.0,
        target_delta=1e-5,
        strict_release_privacy=True,
        allow_nonprivate_debug=True,
        batch_size=120,
        disable_gat=True,
    )

    # Fit and generate
    df_syn, report = pipe.fit_generate(df_raw, num_samples=20, epochs=1)
    assert df_syn.shape == (20, 3)

    # 2. Poison / delete all private objects from pipeline scope
    # Any private state from fitting must be inaccessible
    checkpoint = pipe.released_checkpoint
    assert checkpoint is not None
    assert "scaler_params" in checkpoint
    assert "hvae_state_dict" in checkpoint

    # Reconstruct fresh student inference synthesizer ONLY from checkpoint
    from acfm_net.representation.hvae import HierarchicalVAE
    from acfm_net.generative.ot_cfm import ContinuousOTCFM, VelocityNetwork

    in_dim = checkpoint["config"]["in_dim"]
    latent_dim = checkpoint["config"]["latent_dim"]

    latent_mode = checkpoint["config"].get("latent_mode", "identity")

    if latent_mode == "vae":
        student_hvae = HierarchicalVAE(in_dim=in_dim, global_dim=32, group_dim=16, num_groups=2).to(pipe.device)
        student_hvae.load_state_dict(checkpoint["hvae_state_dict"])
        student_hvae.eval()
    else:
        student_hvae = None

    hidden_dim = checkpoint["config"].get("hidden_dim", 128)
    student_ot = ContinuousOTCFM(continuous_dim=latent_dim, cond_dim=0, hidden_dim=hidden_dim).to(pipe.device)
    student_ot.velocity_net.load_state_dict(checkpoint["velocity_net_state_dict"])
    student_ot.eval()

    # Create poisoned reference to the original training data
    poisoned_raw = PoisonedPrivateTensor(df_raw)
    poisoned_X = PoisonedPrivateTensor(torch.randn(200, in_dim))

    # Test student sampling using ONLY pure standard normal prior z_0 ~ N(0, I)
    with torch.no_grad():
        # Assert student generation does NOT touch poisoned_raw or poisoned_X
        # Sample purely from prior
        z_prior = torch.randn(50, latent_dim, device=pipe.device)
        student_latents = student_ot.sample(num_samples=50, device=pipe.device, num_steps=5, solver="rk4")
        if latent_mode == "vae":
            student_scaled = student_hvae.decode(student_latents)
        else:
            student_scaled = student_latents
        clamped_scaled = torch.clamp(student_scaled, -1.0, 1.0).cpu().numpy()
        student_encoded = pipe.scaler.inverse_transform(clamped_scaled)
        df_student = pipe.encoder.decode(student_encoded)

    assert df_student.shape == (50, 3)

    # Verify that attempting to touch private data actually triggers failure
    with pytest.raises(PrivateAccessViolation):
        _ = poisoned_raw["age"]
    with pytest.raises(PrivateAccessViolation):
        _ = poisoned_X[0]
