import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    4.4 Contrastive Representation Regularization (HCARLN Stage 4) - ★ NEW IN v5 ★
    Supervised Contrastive Loss (Khosla et al. 2020).
    Pulls together latent representations of patients with same diagnosis / risk group;
    Pushes apart latent representations of distinct clinical risk cohorts.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        features: (B, Z_dim)
        labels: (B,) class or risk group labels
        """
        if labels is None or len(labels) == 0:
            return torch.tensor(0.0, device=features.device)

        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # Compute similarity matrix
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        # Compute log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=features.device)
