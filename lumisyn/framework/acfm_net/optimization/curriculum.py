import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional


class SlicedWassersteinDistance(nn.Module):
    """
    Sliced Wasserstein Distance (SWD) — differentiable 1D Monte Carlo approximation.
    """
    def __init__(self, num_projections: int = 64):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, X_real: torch.Tensor, X_syn: torch.Tensor) -> torch.Tensor:
        if X_real.dim() == 1: X_real = X_real.unsqueeze(-1)
        if X_syn.dim()  == 1: X_syn  = X_syn.unsqueeze(-1)
        d = X_real.shape[1]
        device = X_real.device
        proj = F.normalize(torch.randn(self.num_projections, d, device=device), p=2, dim=1)
        pr = torch.matmul(X_real, proj.T)
        ps = torch.matmul(X_syn,  proj.T)
        pr_s, _ = torch.sort(pr, dim=0)
        ps_s, _ = torch.sort(ps, dim=0)
        min_b = min(pr_s.shape[0], ps_s.shape[0])
        return torch.mean(torch.abs(pr_s[:min_b] - ps_s[:min_b]))


class CurriculumTrainer:
    """
    8.3  3-Phase Curriculum Training Engine (Stage 8) — v5.1 Spec Compliant
    Phase 1 (Epochs 1-20)  : Representation & Graph warm-up
    Phase 2 (Epochs 21-70) : Flow Matching & Privacy warm-up
    Phase 3 (Epochs 71+)   : Constraint Repair, Fairness & Utility fine-tuning

    5-loss objective (full spec):
      L_rep    — HVAE reconstruction + KL
      L_flow   — OT-CFM flow matching
      L_repair — Gradient repair residual
      L_swd    — Sliced Wasserstein Distance
      L_fair   — Demographic Parity gap (Fairness)
      L_clin   — Clinical constraint violations
      L_priv   — Privacy noise regularisation proxy
    """

    def __init__(self, total_epochs: int = 100):
        self.total_epochs = total_epochs
        self.swd = SlicedWassersteinDistance(num_projections=64)

    def get_phase(self, epoch: int) -> int:
        if epoch <= 20:  return 1
        if epoch <= 70:  return 2
        return 3

    def compute_fairness_loss(
        self,
        x_gen: torch.Tensor,
        sensitive_col_idx: Optional[int] = None,
        target_col_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Demographic Parity Gap: |E[Y|A=0] - E[Y|A=1]|
        If no sensitive column is identifiable, returns a proxy:
        variance of mean predictions across random halves of the batch.
        """
        if x_gen.shape[0] < 4:
            return torch.tensor(0.0, device=x_gen.device)

        if sensitive_col_idx is not None and sensitive_col_idx < x_gen.shape[1]:
            sens_vals = x_gen[:, sensitive_col_idx]
            median_s  = sens_vals.median().detach()
            mask_a = (sens_vals >= median_s)
            mask_b = ~mask_a

            tgt = x_gen[:, target_col_idx] if (
                target_col_idx is not None and target_col_idx < x_gen.shape[1]
            ) else x_gen.mean(dim=1)

            if mask_a.sum() > 0 and mask_b.sum() > 0:
                dp_gap = torch.abs(tgt[mask_a].mean() - tgt[mask_b].mean())
                return dp_gap

        # Proxy: variance of row-means across random split (always differentiable)
        half = x_gen.shape[0] // 2
        mean_a = x_gen[:half].mean()
        mean_b = x_gen[half:].mean()
        return torch.abs(mean_a - mean_b)

    def compute_clinical_loss(
        self,
        x_gen: torch.Tensor,
        sys_idx: Optional[int] = None,
        dia_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Penalise systolic < diastolic + 10 (BP constraint).
        If column indices not known, returns 0.
        """
        if (sys_idx is None or dia_idx is None or
                sys_idx >= x_gen.shape[1] or dia_idx >= x_gen.shape[1]):
            return torch.tensor(0.0, device=x_gen.device)
        diff = x_gen[:, sys_idx] - (x_gen[:, dia_idx] + 10.0)
        return F.relu(-diff).mean()

    def compute_privacy_loss(
        self,
        Z: torch.Tensor,
        noise_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Privacy regularisation: penalise large latent norms relative to noise.
        High noise_scale should accompany small ||Z|| to reduce signal leakage.
        L_priv = noise_scale_mean * ||Z||_2_mean
        """
        return noise_scales.mean() * Z.norm(dim=1).mean()

    def compute_curriculum_loss(
        self,
        epoch: int,
        rep_loss: torch.Tensor,
        flow_loss: torch.Tensor,
        repair_loss: torch.Tensor,
        swd_loss: torch.Tensor,
        fair_loss: Optional[torch.Tensor] = None,
        clin_loss: Optional[torch.Tensor] = None,
        priv_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        phase = self.get_phase(epoch)

        # Default missing losses to zero
        f_l = fair_loss if fair_loss is not None else torch.tensor(0.0, device=rep_loss.device)
        c_l = clin_loss if clin_loss is not None else torch.tensor(0.0, device=rep_loss.device)
        p_l = priv_loss if priv_loss is not None else torch.tensor(0.0, device=rep_loss.device)

        if phase == 1:
            # Phase 1: Representation & Graph warm-up
            w = dict(rep=1.0, flow=0.1, repair=0.0, swd=0.0, fair=0.0, clin=0.0, priv=0.0)
            active = [rep_loss, 0.1 * flow_loss]

        elif phase == 2:
            # Phase 2: Flow Matching & Privacy warm-up
            w = dict(rep=0.5, flow=1.0, repair=0.1, swd=0.2, fair=0.0, clin=0.05, priv=0.1)
            active = [0.5 * rep_loss, flow_loss, 0.2 * swd_loss, 0.05 * c_l, 0.1 * p_l]

        else:
            # Phase 3: Repair, Fairness & Utility fine-tuning
            w = dict(rep=0.2, flow=0.8, repair=1.0, swd=1.0, fair=0.5, clin=0.5, priv=0.2)
            active = [
                0.2  * rep_loss,
                0.8  * flow_loss,
                repair_loss,
                swd_loss,
                0.5  * f_l,
                0.5  * c_l,
                0.2  * p_l,
            ]

        total_loss = (
            w["rep"]    * rep_loss    +
            w["flow"]   * flow_loss   +
            w["repair"] * repair_loss +
            w["swd"]    * swd_loss    +
            w["fair"]   * f_l         +
            w["clin"]   * c_l         +
            w["priv"]   * p_l
        )

        info = {
            "phase":   phase,
            "epoch":   epoch,
            "weights": w,
            "loss_components": {
                "rep":    float(rep_loss.item()),
                "flow":   float(flow_loss.item()),
                "repair": float(repair_loss.item()),
                "swd":    float(swd_loss.item()),
                "fair":   float(f_l.item()),
                "clin":   float(c_l.item()),
                "priv":   float(p_l.item()),
            }
        }
        return total_loss, active, info
