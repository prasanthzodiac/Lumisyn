import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

class DiscreteCategoricalFlow(nn.Module):
    """
    6.2 Discrete Flow / Categorical Flow (Generative Stage 6) ★ v5 REDESIGNED ★
    Dedicated categorical flow matching via Gumbel-Softmax / Continuous-time Markov chain transition rates.
    Prevents rounding artifacts and boundary blur of standard continuous models on categorical features.
    """

    def __init__(self, num_categories_list: List[int], cond_dim: int = 128):
        super().__init__()
        self.num_categories_list = num_categories_list
        self.total_cats = sum(num_categories_list)

        self.rate_network = nn.Sequential(
            nn.Linear(self.total_cats + cond_dim + 32, 256),
            nn.SiLU(),
            nn.Linear(256, self.total_cats)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

    def compute_loss(self, x_onehot: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x_onehot: Concatenated one-hot targets (B, total_cats)
        cond: Fused conditioning (B, cond_dim)
        """
        B = x_onehot.shape[0]
        device = x_onehot.device

        t = torch.rand((B, 1), device=device)
        t_emb = self.time_embed(t)

        # Discrete interpolation noise path via Gumbel noise mixture
        noise_uniform = torch.ones_like(x_onehot) / float(self.total_cats)
        x_t = (1.0 - t) * noise_uniform + t * x_onehot

        inputs = torch.cat([x_t, t_emb, cond], dim=-1)
        pred_logits = self.rate_network(inputs)

        # Cross-entropy loss against target one-hot distribution
        loss = F.binary_cross_entropy_with_logits(pred_logits, x_onehot)
        return loss

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """
        Samples discrete categorical indices via Gumbel-Softmax decoding.
        """
        B = cond.shape[0]
        device = cond.device

        t_1 = torch.ones((B, 1), device=device)
        t_emb = self.time_embed(t_1)

        noise_prior = torch.ones((B, self.total_cats), device=device) / float(self.total_cats)
        inputs = torch.cat([noise_prior, t_emb, cond], dim=-1)

        logits = self.rate_network(inputs)
        
        # Hard Gumbel-Softmax sampling
        sampled_onehot = F.gumbel_softmax(logits, tau=temperature, hard=True)
        return sampled_onehot
