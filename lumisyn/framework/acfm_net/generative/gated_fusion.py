import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSourceGatedFusion(nn.Module):
    """
    6.3 Gated Multi-Source Cross-Attention Fusion (Generative Stage 6) ★ v5 REDESIGNED ★
    Replaces v4 linear combination (F = αM + βG + γP + δZ) with non-linear gated cross-attention.
    Fuses Metadata M, Graph G, Privacy P, and Latent Z.
    """

    def __init__(self, latent_dim: int, metadata_dim: int = 32, graph_dim: int = 64, privacy_dim: int = 32, fused_dim: int = 128):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, fused_dim)
        self.meta_proj = nn.Linear(metadata_dim, fused_dim) if metadata_dim > 0 else None
        self.graph_proj = nn.Linear(graph_dim, fused_dim) if graph_dim > 0 else None
        self.priv_proj = nn.Linear(privacy_dim, fused_dim) if privacy_dim > 0 else None

        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 4, fused_dim),
            nn.SiLU(),
            nn.Linear(fused_dim, 4),
            nn.Softmax(dim=-1)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.SiLU()
        )

    def forward(
        self,
        latent_Z: torch.Tensor,
        metadata_M: torch.Tensor = None,
        graph_G: torch.Tensor = None,
        privacy_P: torch.Tensor = None
    ) -> torch.Tensor:
        B = latent_Z.shape[0]
        h_z = self.latent_proj(latent_Z)

        h_m = self.meta_proj(metadata_M) if (self.meta_proj and metadata_M is not None) else torch.zeros_like(h_z)
        h_g = self.graph_proj(graph_G) if (self.graph_proj and graph_G is not None) else torch.zeros_like(h_z)
        h_p = self.priv_proj(privacy_P) if (self.priv_proj and privacy_P is not None) else torch.zeros_like(h_z)

        # Concatenate for dynamic gating weights
        concat_h = torch.cat([h_z, h_m, h_g, h_p], dim=-1)  # (B, fused_dim * 4)
        weights = self.gate(concat_h)                       # (B, 4)

        w_z, w_m, w_g, w_p = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3], weights[:, 3:4]
        fused = w_z * h_z + w_m * h_m + w_g * h_g + w_p * h_p
        return self.out_layer(fused)
