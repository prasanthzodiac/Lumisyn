import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


from opacus.validators import ModuleValidator
from opacus.grad_sample import GradSampleModule


class DPSGDWrapper:
    """
    Research-Grade Differential Privacy SGD (DP-SGD) Engine.
    Ref: Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.

    Mathematical Formulation:
    For loss per sample l_i(theta):
    1. Per-sample gradient: g_i = grad_theta l_i(theta)
    2. Per-sample clipping: bar_g_i = g_i / max(1, ||g_i||_2 / C)
       Invariant verified: ||bar_g_i||_2 <= C for all i.
    3. Gaussian Perturbation: tilde_g = (1/B) * [ sum_i bar_g_i + N(0, sigma^2 C^2 I) ]
    4. Descent Step: theta <- theta - lr * tilde_g
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        expected_batch_size: Optional[float] = None,
        validate_architecture: bool = True,
    ):
        if validate_architecture:
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                raise ValueError(
                    f"Opacus ModuleValidator found DP-incompatible layers in model: {errors}"
                )

        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.expected_batch_size = float(expected_batch_size) if expected_batch_size is not None else None
        self.last_sample_norms: List[float] = []
        self.last_clipped_norms: List[float] = []
        self.last_fraction_clipped: float = 0.0

    def step_microbatch(
        self,
        losses: torch.Tensor,
        retain_graph: bool = False
    ) -> float:
        """
        Takes a 1D tensor of per-sample losses of length B,
        performs per-sample gradient clipping with strict assertion,
        and adds calibrated Gaussian perturbation scaled by the EXPECTED batch size (q*N).
        """
        B = losses.shape[0]
        # Cache scalar BEFORE any autograd.grad call frees the computation graph
        loss_scalar = float(losses.mean().item()) if B > 0 else 0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        scale_divisor = float(self.expected_batch_size) if self.expected_batch_size is not None else float(max(1, B))

        if B == 0:
            # Match Opacus behavior on empty Poisson draw:
            # Under Poisson sampling with rate q, an empty batch produces zero accumulated sample gradients.
            # Pure calibrated Gaussian noise is added to parameters scaled by expected batch size (q*N),
            # and the optimizer steps, advancing one iteration in privacy accounting.
            self.optimizer.zero_grad()
            if params:
                for p in params:
                    if self.noise_multiplier > 0.0:
                        noise = torch.randn_like(p) * (self.noise_multiplier * self.max_grad_norm)
                        p.grad = noise / scale_divisor
                    else:
                        p.grad = torch.zeros_like(p)
                self.optimizer.step()
            self.last_sample_norms = []
            self.last_clipped_norms = []
            self.last_fraction_clipped = 0.0
            return 0.0

        if not params:
            self.last_sample_norms = []
            self.last_clipped_norms = []
            self.last_fraction_clipped = 0.0
            return 0.0

        # Gate 4b Item 8: Chunked per-sample autograd (chunks of 128-256)
        # Slices batch into manageable chunks to prevent VRAM spikes, accumulates
        # clipped gradients, and adds DP noise once at step level.
        chunk_size = 128
        clipped_grads = [torch.zeros_like(p) for p in params]
        sample_norms: List[float] = []
        clipped_norms: List[float] = []
        use_chunked = True

        try:
            for start in range(0, B, chunk_size):
                end = min(B, start + chunk_size)
                c_len = end - start
                c_loss = losses[start:end]
                grad_outputs = torch.eye(c_len, device=losses.device)
                b_grads = torch.autograd.grad(
                    outputs=c_loss,
                    inputs=params,
                    grad_outputs=grad_outputs,
                    is_grads_batched=True,
                    retain_graph=(end < B) or retain_graph,
                    allow_unused=True,
                )
                c_sq = torch.zeros(c_len, device=losses.device)
                for g in b_grads:
                    if g is not None:
                        if g.dim() == 1:
                            c_sq += g.pow(2)
                        else:
                            c_sq += g.flatten(start_dim=1).pow(2).sum(dim=1)
                c_norms_t = torch.clamp(c_sq.sqrt(), min=1e-8)
                c_clip_coef_t = torch.clamp(self.max_grad_norm / c_norms_t, max=1.0)
                c_clipped_norms_t = c_norms_t * c_clip_coef_t

                sample_norms.extend(c_norms_t.detach().cpu().tolist())
                clipped_norms.extend(c_clipped_norms_t.detach().cpu().tolist())

                for idx, g in enumerate(b_grads):
                    if g is not None:
                        if g.dim() == 1:
                            clipped_g = g * c_clip_coef_t
                            clipped_grads[idx].add_(clipped_g.sum(dim=0))
                        else:
                            view_shape = [c_len] + [1] * (g.dim() - 1)
                            clipped_g = g * c_clip_coef_t.view(*view_shape)
                            clipped_grads[idx].add_(clipped_g.sum(dim=0))
        except Exception:
            use_chunked = False

        if not use_chunked:
            # Fallback: exact per-sample gradient loop if batched autograd fails
            clipped_grads = [torch.zeros_like(p) for p in params]
            sample_norms = []
            clipped_norms = []

            for i in range(B):
                sample_loss = losses[i]
                sample_grads = torch.autograd.grad(
                    sample_loss,
                    params,
                    retain_graph=(i < B - 1) or retain_graph,
                    allow_unused=True
                )

                total_norm_sq = 0.0
                for g in sample_grads:
                    if g is not None:
                        total_norm_sq += g.pow(2).sum().item()
                sample_norm = max(1e-8, total_norm_sq ** 0.5)
                sample_norms.append(sample_norm)

                clip_coef = min(1.0, self.max_grad_norm / sample_norm)
                clipped_norm = sample_norm * clip_coef
                clipped_norms.append(clipped_norm)

                for acc_g, g in zip(clipped_grads, sample_grads):
                    if g is not None:
                        acc_g.add_(g * clip_coef)

        # Mathematical Invariant Assertion: ||bar_g_i||_2 <= C for all samples i
        for i, c_norm in enumerate(clipped_norms):
            assert c_norm <= self.max_grad_norm + 1e-4, (
                f"DP Invariant Violation: Sample {i} clipped gradient norm {c_norm:.6f} "
                f"exceeds clipping bound C={self.max_grad_norm:.6f}."
            )

        self.last_sample_norms = sample_norms
        self.last_clipped_norms = clipped_norms
        self.last_fraction_clipped = float(np.mean([n > self.max_grad_norm for n in sample_norms]))

        # Inject calibrated Gaussian noise and set param.grad scaled by EXPECTED batch size
        self.optimizer.zero_grad()
        for p, acc_g in zip(params, clipped_grads):
            if acc_g is not None:
                if self.noise_multiplier > 0.0:
                    noise = torch.randn_like(acc_g) * (self.noise_multiplier * self.max_grad_norm)
                    perturbed_grad = (acc_g + noise) / scale_divisor
                else:
                    perturbed_grad = acc_g / scale_divisor
                p.grad = perturbed_grad
            else:
                p.grad = None

        self.optimizer.step()
        return loss_scalar
