import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
from scipy.optimize import linear_sum_assignment


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal frequency positional encoding for continuous time t in [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class VelocityNetwork(nn.Module):
    """
    Research-Grade Residual Vector Field v_θ(z_t, t, condition) for Continuous Flow Matching.
    Adheres to Tong et al. (2023) / Lipman et al. (2022) architecture.
    Integrates Stage 3 & 3b Domain Causal Graph relational inductive bias.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int = 0,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        num_layers: int = 3,
        adj_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        # Stage 3 & 3b Graph inductive bias: domain causal graph adjacency
        if adj_matrix is not None and adj_matrix.shape[0] == data_dim:
            # Symmetrically normalize A_tilde = D^{-1/2} (A + I) D^{-1/2}
            A_tilde = adj_matrix + torch.eye(data_dim, device=adj_matrix.device)
            deg = A_tilde.sum(dim=-1)
            deg_inv_sqrt = torch.pow(deg.clamp(min=1e-5), -0.5)
            norm_adj = deg_inv_sqrt.unsqueeze(-1) * A_tilde * deg_inv_sqrt.unsqueeze(0)
            self.register_buffer("adj_matrix", norm_adj.float())
            self.graph_weight = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        else:
            self.register_buffer("adj_matrix", torch.eye(data_dim, dtype=torch.float32))
            self.graph_weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Learnable affine transformation inside the flow network trained under DP-SGD (Gate 2b Fix)
        self.affine_scale = nn.Parameter(torch.ones(data_dim))
        self.affine_bias = nn.Parameter(torch.zeros(data_dim))

        in_features = data_dim + time_embed_dim + cond_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            )
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, data_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Apply DP-trained learnable affine transformation on input latents
        z_t_scaled = z_t * self.affine_scale + self.affine_bias

        # Stage 3/3b: Apply graph relational propagation if domain causal graph is registered
        if self.adj_matrix is not None:
            z_graph = torch.matmul(z_t, self.adj_matrix.to(z_t.device))
            z_t_scaled = z_t_scaled + self.graph_weight * z_graph

        t_emb = self.time_embed(t)
        inputs = [z_t_scaled, t_emb]
        if self.cond_dim > 0 and cond is not None:
            inputs.append(cond)

        h = self.input_layer(torch.cat(inputs, dim=-1))
        for block in self.blocks:
            h = h + block(h)  # Residual connection

        return self.output_layer(h)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # Tolerate missing/unexpected graph keys for backwards compatibility with legacy checkpoints
        for k in [prefix + "graph_weight", prefix + "adj_matrix"]:
            if k in missing_keys:
                missing_keys.remove(k)
            if k in unexpected_keys:
                unexpected_keys.remove(k)


class ContinuousOTCFM(nn.Module):
    """
    Exact Optimal Transport Conditional Flow Matching (OT-CFM).
    Ref: Tong et al., "Improving and Generalizing Flow-Based Generative Models
    with Minibatch Optimal Transport", ICML/NeurIPS 2023.

    Features:
    - Independent Flow Matching (I-CFM) default preserving per-example sensitivity
    - Domain Constraint Graph (DCGB) relational propagation in VelocityNetwork
    - Batch Hungarian optimal transport matching (non-private ablation only)
    - Straight trajectory interpolation with analytical target vector fields
    - Multi-step ODE integration (Euler, Midpoint, RK4)
    """

    def __init__(
        self,
        continuous_dim: int,
        cond_dim: int = 0,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        num_layers: int = 2,
        adj_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.cond_dim = cond_dim
        self.velocity_net = VelocityNetwork(
            data_dim=continuous_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            adj_matrix=adj_matrix,
        )

    def solve_minibatch_ot(self, z_0: torch.Tensor, z_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves exact Monge Optimal Transport assignment on Euclidean distance matrix.
        z_0: (B, D) prior noise samples
        z_1: (B, D) empirical data samples
        Returns:
            z_0, z_1_coupled: paired samples minimizing total transport cost.
        """
        B = z_0.shape[0]
        if B <= 1:
            return z_0, z_1

        with torch.no_grad():
            # Pairwise squared Euclidean distance matrix M_{ij} = ||z_0_i - z_1_j||^2
            dist_matrix = torch.cdist(z_0, z_1, p=2).pow(2).detach().cpu().numpy()

            # Hungarian algorithm (exact optimal matching in O(B^3))
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            z_1_coupled = z_1[col_ind]
            return z_0, z_1_coupled

    def compute_loss(
        self,
        z_1: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        use_ot_coupling: bool = False,
        return_per_sample: bool = False,
        allow_nonprivate_ot: bool = False,
    ) -> torch.Tensor:
        """
        Computes Flow Matching regression loss against straight trajectories.

        Parameters:
            z_1: (B, D) Empirical data / latent samples.
            cond: Optional conditional context.
            use_ot_coupling: If False (DEFAULT, I-CFM), uses independent coupling
                             z_0 ~ N(0, I) paired strictly per sample. This guarantees
                             independent per-example gradients required by DP-SGD.
                             If True (BATCH-OT ABLATION), uses Kuhn-Munkres matching
                             across the batch; this couples examples and is ONLY permitted
                             as a non-private ablation (requires allow_nonprivate_ot=True).
            return_per_sample: If True, returns (B,) per-example loss for DP-SGD clipping.
            allow_nonprivate_ot: Safety guard requiring explicit acknowledgement for batch OT.
        """
        B = z_1.shape[0]
        device = z_1.device

        # 1. Sample standard normal base prior z_0 ~ N(0, I) independently per record
        z_0 = torch.randn_like(z_1)

        # 2. Coupling
        if use_ot_coupling:
            if not allow_nonprivate_ot:
                raise ValueError(
                    "Batch-Hungarian Optimal Transport coupling (use_ot_coupling=True) violates "
                    "per-sample sensitivity bounds under DP-SGD. Set allow_nonprivate_ot=True "
                    "only if running a non-private comparison ablation."
                )
            z_0, z_1_coupled = self.solve_minibatch_ot(z_0, z_1)
        else:
            # Independent Flow Matching (I-CFM): strictly per-example independent pairing
            z_1_coupled = z_1

        # 3. Sample continuous time t ~ Uniform(0, 1) independently per record
        t = torch.rand((B, 1), device=device)

        # 4. Straight trajectory and target velocity
        z_t = (1.0 - t) * z_0 + t * z_1_coupled
        u_t = z_1_coupled - z_0

        # 5. Predict velocity field
        v_pred = self.velocity_net(z_t, t, cond)

        # 6. Mean squared error velocity regression loss
        per_sample_loss = F.mse_loss(v_pred, u_t, reduction="none").mean(dim=1)
        return per_sample_loss if return_per_sample else per_sample_loss.mean()

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        solver: str = "rk4",
        prior_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Simulates the continuous ODE dz/dt = v_theta(z_t, t) from t=0 (prior) to t=1 (data manifold).
        Supported solvers: 'euler', 'midpoint', 'rk4' (Runge-Kutta 4th order).
        """
        # Sample prior z_0 ~ N(0, s^2 I) with scale s -- Zero training data memorization!
        z = torch.randn((num_samples, self.continuous_dim), device=device) * float(prior_scale)
        dt = 1.0 / num_steps

        if solver == "euler":
            for i in range(num_steps):
                t_val = torch.full((num_samples, 1), i * dt, device=device)
                v = self.velocity_net(z, t_val, cond)
                z = z + v * dt

        elif solver == "midpoint":
            for i in range(num_steps):
                t_val = torch.full((num_samples, 1), i * dt, device=device)
                t_mid = torch.full((num_samples, 1), (i + 0.5) * dt, device=device)
                v1 = self.velocity_net(z, t_val, cond)
                z_mid = z + 0.5 * dt * v1
                v_mid = self.velocity_net(z_mid, t_mid, cond)
                z = z + dt * v_mid

        elif solver == "rk4":
            for i in range(num_steps):
                t0 = torch.full((num_samples, 1), i * dt, device=device)
                t_half = torch.full((num_samples, 1), (i + 0.5) * dt, device=device)
                t1 = torch.full((num_samples, 1), (i + 1.0) * dt, device=device)

                k1 = self.velocity_net(z, t0, cond)
                k2 = self.velocity_net(z + 0.5 * dt * k1, t_half, cond)
                k3 = self.velocity_net(z + 0.5 * dt * k2, t_half, cond)
                k4 = self.velocity_net(z + dt * k3, t1, cond)

                z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        else:
            raise ValueError(f"Unknown ODE solver: {solver}")

        return z
