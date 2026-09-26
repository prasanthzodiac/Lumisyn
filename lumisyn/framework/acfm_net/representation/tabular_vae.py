import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class TabularVAE(nn.Module):
    """
    Research-Grade Mixed-Type Tabular Variational Autoencoder (T-VAE).
    Maps heterogeneous tabular records (continuous + high-cardinality categorical)
    into a unified, smooth continuous latent manifold Z ~ N(0, I).

    Mathematical Formulation:
    - Continuous features x_num in R^{D_num}
    - Categorical features x_cat_k in {0, ..., C_k - 1} with learned embeddings E_k
    - Joint Encoder: x -> h_enc -> mu(x), log_var(x) in R^{latent_dim}
    - Reparameterization: z = mu + sigma * eps, eps ~ N(0, I)
    - Decoder: z -> h_dec -> continuous reconstruction + categorical logits
    - Loss: L_total = L_MSE_num + sum_k L_CE_cat_k + beta * D_KL(q(z|x) || N(0, I))
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_dims: List[int],
        latent_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        embedding_dim_factor: float = 0.5,
        max_embedding_dim: int = 32
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_dims = categorical_dims
        self.latent_dim = latent_dim
        self.num_categoricals = len(categorical_dims)

        # 1. Categorical Embeddings
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for card in categorical_dims:
            # Rule of thumb for tabular embedding dimension: min(max_emb, max(4, int(card^0.5)))
            emb_dim = min(max_embedding_dim, max(4, int(card ** embedding_dim_factor)))
            self.embeddings.append(nn.Embedding(card, emb_dim))
            total_emb_dim += emb_dim

        self.input_dim = continuous_dim + total_emb_dim

        # 2. Encoder Backbone
        enc_layers = []
        in_d = self.input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_d, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(0.05)
            ])
            in_d = h_dim

        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_d, latent_dim)
        self.fc_logvar = nn.Linear(in_d, latent_dim)

        # 3. Decoder Backbone
        dec_layers = []
        in_d = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_d, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(0.05)
            ])
            in_d = h_dim

        self.decoder = nn.Sequential(*dec_layers)

        # 4. Reconstruction Heads
        if continuous_dim > 0:
            self.continuous_head = nn.Linear(in_d, continuous_dim)
        else:
            self.continuous_head = None

        self.categorical_heads = nn.ModuleList([
            nn.Linear(in_d, card) for card in categorical_dims
        ])

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard Gaussian reparameterization trick."""
        std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 10.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_inputs(self, x_cont: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """Combines continuous features and categorical embeddings into unified representation."""
        components = []
        if self.continuous_dim > 0 and x_cont is not None:
            components.append(x_cont)

        if self.num_categoricals > 0 and x_cat is not None:
            for k, emb_layer in enumerate(self.embeddings):
                cat_col = x_cat[:, k].long()
                components.append(emb_layer(cat_col))

        if not components:
            raise ValueError("Both x_cont and x_cat are empty.")

        return torch.cat(components, dim=-1)

    def encode(self, x_cont: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maps tabular data to latent representation z ~ q(z|x)."""
        x_in = self.encode_inputs(x_cont, x_cat)
        h = self.encoder(x_in)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """Decodes continuous latent z into continuous reconstructions and categorical logits."""
        h = self.decoder(z)
        x_cont_recon = self.continuous_head(h) if self.continuous_head is not None else None
        cat_logits = [head(h) for head in self.categorical_heads]
        return x_cont_recon, cat_logits

    def forward(
        self,
        x_cont: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        beta_kl: float = 1.0
    ) -> Dict[str, Any]:
        """
        Forward pass computing reconstruction and KL divergence losses.
        """
        z, mu, logvar = self.encode(x_cont, x_cat)
        x_cont_recon, cat_logits = self.decode(z)

        # 1. Continuous Reconstruction Loss (MSE)
        if self.continuous_dim > 0 and x_cont is not None and x_cont_recon is not None:
            loss_cont = F.mse_loss(x_cont_recon, x_cont, reduction="mean")
        else:
            loss_cont = torch.tensor(0.0, device=z.device)

        # 2. Categorical Reconstruction Loss (Cross-Entropy)
        loss_cat = torch.tensor(0.0, device=z.device)
        if self.num_categoricals > 0 and x_cat is not None:
            for k, logits in enumerate(cat_logits):
                target = x_cat[:, k].long()
                loss_cat = loss_cat + F.cross_entropy(logits, target, reduction="mean")

        # 3. Disentanglement / Prior Regularization (KL Divergence)
        # KL(q(z|x) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        total_loss = loss_cont + loss_cat + beta_kl * kl_div

        return {
            "loss": total_loss,
            "loss_cont": loss_cont,
            "loss_cat": loss_cat,
            "loss_kl": kl_div,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "x_cont_recon": x_cont_recon,
            "cat_logits": cat_logits
        }
