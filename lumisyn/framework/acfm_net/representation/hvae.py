import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class HierarchicalVAE(nn.Module):
    """
    4.3 Hierarchical VAE (HVAE) - HCARLN Stage 4
    Global vector + group heads architecture:
    Level 1 — Global latent z_g (d=64): overall patient state
    Level 2 — Feature-group latents z_k (d=32 each): group subspaces
    Joint latent: Z = [z_g ‖ z_1 ‖ ... ‖ z_K]
    Note: Structured into subspaces; does NOT optimize a decorrelation penalty
    and does not claim statistical disentanglement.
    KL Annealing β_KL(t) reaches 1.0 to enforce standard normal latents ~ N(0, I).
    """

    def __init__(
        self,
        in_dim: int,
        global_dim: int = 64,
        group_dim: int = 32,
        num_groups: int = 4,
        beta_kl: float = 0.01,
        free_bits: float = 0.0,
        logvar_init_bias: float = 0.0,
        deterministic_encoder: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.global_dim = global_dim
        self.group_dim = group_dim
        self.num_groups = num_groups
        self.joint_dim = global_dim + num_groups * group_dim
        self.beta_kl = beta_kl
        self.free_bits = free_bits
        self.logvar_init_bias = logvar_init_bias
        self.deterministic_encoder = deterministic_encoder

        # Encoder Backbone
        self.encoder_backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU()
        )

        # Global Head (Level 1)
        self.fc_mu_g = nn.Linear(128, global_dim)
        self.fc_logvar_g = nn.Linear(128, global_dim)

        # Group Heads (Level 2) - conditioned on global latent z_g
        self.group_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 + global_dim, 64),
                nn.SiLU(),
                nn.Linear(64, group_dim * 2)  # mu and logvar
            ) for _ in range(num_groups)
        ])

        # Optional logvar bias initialization (Gate 4c Item 2a)
        if logvar_init_bias != 0.0:
            nn.init.constant_(self.fc_logvar_g.bias, logvar_init_bias)
            for head in self.group_heads:
                with torch.no_grad():
                    head[-1].bias[group_dim:].fill_(logvar_init_bias)

        # Decoder Backbone
        self.decoder = nn.Sequential(
            nn.Linear(self.joint_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, in_dim)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.deterministic_encoder:
            return mu
        # Clamp logvar to [-10.0, 4.0] to prevent numerical explosion and stabilize reconstruction
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=4.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder_backbone(x)

        # Level 1
        mu_g = self.fc_mu_g(h)
        logvar_g = torch.clamp(self.fc_logvar_g(h), min=-10.0, max=4.0)
        z_g = self.reparameterize(mu_g, logvar_g)

        # Level 2
        mus_group = [mu_g]
        logvars_group = [logvar_g]
        z_groups = [z_g]

        for head in self.group_heads:
            h_g = torch.cat([h, z_g], dim=-1)
            out = head(h_g)
            mu_k, logvar_k = torch.chunk(out, 2, dim=-1)
            logvar_k = torch.clamp(logvar_k, min=-10.0, max=4.0)
            z_k = self.reparameterize(mu_k, logvar_k)
            mus_group.append(mu_k)
            logvars_group.append(logvar_k)
            z_groups.append(z_k)

        joint_Z = torch.cat(z_groups, dim=-1)  # (B, global_dim + 4*group_dim)
        total_mu = torch.cat(mus_group, dim=-1)
        total_logvar = torch.cat(logvars_group, dim=-1)

        return joint_Z, total_mu, total_logvar

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.decoder(Z)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        beta_kl: Optional[float] = None,
        free_bits: Optional[float] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        Z, mu, logvar = self.encode(x)
        x_recon = self.decode(Z)

        # Keep per-example losses so the privacy wrapper can clip each example
        # before adding Gaussian noise. Target MUST be fixed X_scaled.
        # ELBO formulation: sum squared error over features for proper DP gradient SNR
        recon_target = target if target is not None else x
        recon_per_sample = F.mse_loss(x_recon, recon_target, reduction="none").sum(dim=1)
        
        # Free-bits formulation (Kingma et al., 2016): clamp per-dimension KL at free_bits threshold
        eff_free_bits = free_bits if free_bits is not None else self.free_bits
        raw_kl_elem = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        raw_total_kl_nats = raw_kl_elem.sum(dim=-1).mean()
        if eff_free_bits > 0.0:
            kl_elem = torch.clamp(raw_kl_elem, min=eff_free_bits)
        else:
            kl_elem = raw_kl_elem
        
        kl_per_sample = kl_elem.sum(dim=1)
        recon_loss = F.mse_loss(x_recon, recon_target, reduction="none").mean()
        kl_loss = kl_elem.mean()
        total_kl_nats = kl_elem.sum(dim=-1).mean()

        # Gate 4b & Stage 4 fix: beta_kl with free-bits protection against posterior collapse
        eff_beta_kl = beta_kl if beta_kl is not None else self.beta_kl
        per_sample_loss = recon_per_sample + eff_beta_kl * kl_per_sample
        total_loss = per_sample_loss.mean()

        return {
            "loss": total_loss,
            "per_sample_loss": per_sample_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_kl_nats": total_kl_nats,
            "raw_total_kl_nats": raw_total_kl_nats,
            "beta_kl": eff_beta_kl,
            "free_bits": eff_free_bits,
            "Z": Z,
            "x_recon": x_recon,
        }
