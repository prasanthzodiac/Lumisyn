import torch
import torch.nn as nn
from typing import List, Dict

class PCGrad:
    """
    8.1 Gradient Conflict Resolution via PCGrad (Optimization Stage 8) ★ v5 REDESIGNED ★
    Projected Conflicting Gradients (Yu et al. 2020).
    Performs per-parameter gradient conflict projection:
    If gradients of two task losses conflict (g_i · g_j < 0):
    g_i ← g_i - (g_i · g_j / ||g_j||²) g_j
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, losses: List[torch.Tensor]):
        """
        losses: List of scalar loss tensors for each objective.
        """
        num_tasks = len(losses)
        if num_tasks == 0:
            return

        if num_tasks == 1:
            self.optimizer.zero_grad()
            losses[0].backward()
            self.optimizer.step()
            return

        # 1. Collect per-parameter gradients for each task
        task_param_grads: List[Dict[nn.Parameter, torch.Tensor]] = []
        
        for i, loss in enumerate(losses):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True if i < num_tasks - 1 else False)

            p_grads = {}
            for param in self.optimizer.param_groups[0]["params"]:
                if param.grad is not None:
                    p_grads[param] = param.grad.clone().detach()
            task_param_grads.append(p_grads)

        # 2. Resolve gradient conflicts per parameter
        self.optimizer.zero_grad()

        for param in self.optimizer.param_groups[0]["params"]:
            param_task_grads = [t_g[param] for t_g in task_param_grads if param in t_g]
            if not param_task_grads:
                continue

            num_p_tasks = len(param_task_grads)
            if num_p_tasks == 1:
                param.grad = param_task_grads[0]
                continue

            # PCGrad projection for this parameter
            p_grads_proj = [g.clone() for g in param_task_grads]
            shuffled_idx = torch.randperm(num_p_tasks)

            for i in shuffled_idx:
                for j in shuffled_idx:
                    if i != j:
                        gi = p_grads_proj[i]
                        gj = param_task_grads[j]
                        dot_prod = torch.sum(gi * gj)
                        if dot_prod < 0:
                            norm_sq = torch.sum(gj * gj) + 1e-8
                            p_grads_proj[i] -= (dot_prod / norm_sq) * gj

            # Average projected gradients across tasks for parameter update
            param.grad = torch.stack(p_grads_proj).mean(dim=0)

        self.optimizer.step()
