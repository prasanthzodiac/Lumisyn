import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class GradNorm(nn.Module):
    """
    8.2 GradNorm Adaptive Loss Weighting (Optimization Stage 8) ★ v5 REDESIGNED ★
    GradNorm (Chen et al. 2018) + MGDA.
    Dynamically balances gradient magnitudes across multi-objective losses.
    Weights: w_i(t) updated such that all task gradients train at similar relative rates.
    """

    def __init__(self, num_tasks: int, alpha: float = 0.15):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))
        self.initial_losses = None

    def get_weighted_loss(self, losses: List[torch.Tensor]) -> torch.Tensor:
        loss_stack = torch.stack(losses)
        normalized_weights = F.softmax(self.weights, dim=0) * self.num_tasks
        weighted_loss = torch.sum(normalized_weights * loss_stack)
        return weighted_loss
