import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from acfm_net.privacy.prv_accountant import PRVAccountant
from acfm_net.privacy.rdp_accountant import RDPAccountant

class AdaptiveBudgetAllocator(nn.Module):
    """
    Stage 5: Dataset-Adaptive Privacy Budget Controller (DAPC).

    Both the release gate and epsilon selection read ONLY public metadata:
    rounded cohort size n, encoded dimension d (fixed by the declared schema),
    declared column sensitivity (column-name rules, not data values), and a
    target-name high-stakes flag. DAPC therefore consumes zero privacy budget,
    and the selected epsilon needs no adaptive-composition argument: it is a
    fixed function of public inputs, chosen before any private data is touched
    by a DP mechanism.

    Public-N assumption: cohort size is treated as public — the same
    assumption DP-SGD already makes when it sets the sample rate q = B/N
    (report/PRIVACY_PROOF.md, A2). N is additionally rounded to the nearest 50.

    Risk vector r = [r_sens, r_scale, r_dim, r_task] in [0, 1]^4:
      r_sens  = mean(s_i^2)          declared column sensitivity
      r_scale = exp(-n / 600)        smaller cohort -> more leverage per record
      r_dim   = 1 - exp(-d / 100)    more released dimensions per record
      r_task  = 0.85 high-stakes target, else 0.30
    R_P  = max(r_sens, 1 - prod_k (1 - w_k r_k)^{gamma_k})
    eps* = eps_max - (eps_max - eps_min) * (1 - e^{-beta R_P}) / (1 - e^{-beta})
    so eps* is monotone non-increasing in risk and always within [eps_min, eps_max].
    Sigma is then solved so the PRV accountant meets eps* exactly.
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        eps_min: float = 0.40,
        eps_max: Optional[float] = None,
        beta: float = 2.5,
        eps_meta: float = 0.0,
        fingerprint_dim: int = 256
    ):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max if eps_max is not None else target_epsilon
        self.target_epsilon = self.eps_max
        self.target_delta = target_delta
        self.beta = beta
        # No private metadata query exists any more, so nothing to reserve by default.
        self.eps_meta = eps_meta

        # [r_sens, r_scale, r_dim, r_task]
        self.weights = np.array([0.35, 0.30, 0.20, 0.15])
        self.gammas = np.array([1.5, 1.0, 1.0, 1.0])

        self.prv_solver = PRVAccountant(target_epsilon=self.target_epsilon, target_delta=self.target_delta)
        self.rdp_monitor = RDPAccountant(target_epsilon=self.target_epsilon, target_delta=self.target_delta)

    def compute_risk_vector(
        self,
        sensitivity_vector: np.ndarray,
        num_samples: int,
        num_features: int,
        is_high_stakes_task: bool = False,
    ) -> np.ndarray:
        """Public-metadata risk vector [r_sens, r_scale, r_dim, r_task] in [0, 1]^4."""
        s_vec = np.clip(np.asarray(sensitivity_vector, dtype=np.float64), 0.0, 1.0)
        r_sens = float(np.mean(s_vec ** 2)) if len(s_vec) > 0 else 0.30
        r_scale = float(np.exp(-float(num_samples) / 600.0))
        r_dim = float(1.0 - np.exp(-float(num_features) / 100.0))
        r_task = 0.85 if is_high_stakes_task else 0.30
        return np.clip(np.array([r_sens, r_scale, r_dim, r_task], dtype=np.float64), 0.0, 1.0)

    def aggregate_risk(self, r_vector: np.ndarray) -> float:
        """R_P = max(r_sens, 1 - prod_k (1 - w_k r_k)^{gamma_k}): declared sensitivity can dominate alone."""
        smooth = float(1.0 - np.prod((1.0 - self.weights * r_vector) ** self.gammas))
        return float(np.clip(max(float(r_vector[0]), smooth), 0.0, 1.0))

    def select_policy_bounded_epsilon(self, R_P: float) -> float:
        """
        Risk-Averse Non-Linear Budget Mapping: ε*(R_P) = ε_max - (ε_max - ε_min) * Φ_β(R_P)
        where Φ_β(R_P) = (1 - e^{-β R_P}) / (1 - e^{-β}) with β = 2.5
        """
        phi_beta = (1.0 - np.exp(-self.beta * R_P)) / (1.0 - np.exp(-self.beta))
        eps_star = self.eps_max - (self.eps_max - self.eps_min) * phi_beta
        return float(np.clip(eps_star, self.eps_min, self.eps_max))

    MIN_COHORT_SIZE: int = 100

    def evaluate_defensive_safety(
        self,
        num_samples: int,
        num_features: int,
        public_schema: Optional[Dict[str, Any]] = None,
        rounded_n: Optional[int] = None,
    ) -> Tuple[str, List[str]]:
        """
        Public-Input Defensive Safety Gate:
        The release gate evaluates ONLY public inputs (explicit cohort size threshold n >= 100,
        feature dimension d, and public schema definition).
        Private values, unique ratios, and sample values are strictly excluded
        from decision branches to prevent side-channel privacy leakage.
        """
        reasons = []

        # Policy Gate: Explicit Minimum Cohort Size (Intended n >= 100)
        # Avoids python round() round-to-even artifacts where n=75..99 could round up to 100
        effective_n = int(num_samples)
        if effective_n < self.MIN_COHORT_SIZE:
            reasons.append(
                f"Cohort size too small for differential privacy guarantees (n={effective_n} < {self.MIN_COHORT_SIZE}). Policy action: ABSTAIN."
            )
            return "ABSTAIN", reasons

        # Policy Gate: Excessive dimensionality relative to cohort size
        if num_features > 500:
            reasons.append(
                f"Feature dimensionality (d={num_features}) exceeds standard policy envelope (500). Policy action: REJECT."
            )
            return "REJECT", reasons

        reasons.append("Public cohort size and schema dimensions meet policy requirements.")
        return "PROCEED", reasons

    def calibrate_noise_multiplier_accountant(
        self,
        target_eps_sgd: float,
        sample_rate: float = 0.00486,
        steps: int = 500,
        delta: float = 1e-5,
        extra_history: Tuple[Tuple[float, float, int], ...] = (),
    ) -> float:
        """
        Solves σ* = min σ subject to accountant(σ, q, T, δ) <= target_eps_sgd.

        A bounded monotone search is used instead of a root solver. Opacus' PRV
        FFT can legitimately overflow at very small σ; the accountant wrapper
        treats those candidates as +∞ and the search safely moves upward.
        """
        target_eps_sgd = max(float(target_eps_sgd), 1e-6)
        sample_rate = min(max(float(sample_rate), 1e-8), 1.0)
        steps = max(int(steps), 1)

        def epsilon(sigma: float) -> float:
            # extra_history: other Gaussian mechanisms released alongside DP-SGD (e.g. Stage 7c
            # marginals), composed jointly so sigma fills exactly the remaining budget.
            acc = PRVAccountant(target_delta=delta)
            return acc._epsilon_from_history([(sigma, sample_rate, steps), *extra_history])

        low, high = 0.05, 1.0
        while epsilon(high) > target_eps_sgd and high < 128.0:
            high *= 2.0

        for _ in range(48):
            mid = (low + high) / 2.0
            if epsilon(mid) > target_eps_sgd:
                low = mid
            else:
                high = mid
        return float(high)

    def analyze_dataset_dapc(
        self,
        sensitivity_vector: torch.Tensor,
        num_samples: int,
        num_features: int,
        is_high_stakes: bool = False,
        dataset_name: str = "dataset.csv",
        sample_rate: Optional[float] = None,
        steps: Optional[int] = None,
        public_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Gate + risk-adaptive epsilon, both from public metadata only (see class docstring).
        Zero privacy cost; no private statistic is read, so none can leak through the choice.
        """
        rounded_n = int(round(num_samples / 50.0) * 50)
        decision, reasons = self.evaluate_defensive_safety(
            num_samples=num_samples,
            num_features=num_features,
            public_schema=public_schema,
            rounded_n=rounded_n,
        )

        sens_np = sensitivity_vector.detach().cpu().numpy() if isinstance(sensitivity_vector, torch.Tensor) else np.asarray(sensitivity_vector)
        r_vector = self.compute_risk_vector(
            sensitivity_vector=sens_np,
            num_samples=rounded_n,
            num_features=num_features,
            is_high_stakes_task=bool(is_high_stakes),
        )
        risk_score_R_P = self.aggregate_risk(r_vector)

        eps_star = self.select_policy_bounded_epsilon(risk_score_R_P)
        eps_sgd = max(0.05, eps_star - self.eps_meta)

        sample_rate = float(sample_rate if sample_rate is not None else min(1.0, 256.0 / max(1, num_samples)))
        steps = int(steps if steps is not None else 500)
        sigma_star = self.calibrate_noise_multiplier_accountant(
            target_eps_sgd=eps_sgd,
            sample_rate=sample_rate,
            steps=steps,
            delta=self.target_delta
        )

        return {
            "certificate_id": f"PDC-{np.random.randint(100000, 999999)}",
            "dataset_name": dataset_name,
            "sample_count": num_samples,
            "rounded_sample_count": rounded_n,
            "feature_count": num_features,
            "policy_envelope": [self.eps_min, self.eps_max],
            "selected_target_epsilon": round(eps_star, 4),
            "meta_selection_budget": self.eps_meta,
            "training_budget_eps_sgd": round(eps_sgd, 4),
            "target_delta": self.target_delta,
            "calibrated_noise_multiplier_sigma": round(sigma_star, 4),
            "clipping_threshold_C": 1.25,
            "decision": decision,
            "decision_reasons": reasons,
            "risk_vector": {
                "r_sens": round(float(r_vector[0]), 4),
                "r_scale": round(float(r_vector[1]), 4),
                "r_dim": round(float(r_vector[2]), 4),
                "r_task": round(float(r_vector[3]), 4),
            },
            "risk_inputs_public_only": True,
            "risk_score_R_P": round(risk_score_R_P, 4),
        }

    def forward(
        self,
        sensitivity_vector: torch.Tensor,
        dataset_fingerprint: torch.Tensor,
        epoch: Optional[int] = None,
        sample_rate: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Allocate the configured target budget across features.

        ``epoch`` is accepted for compatibility with the training scheduler;
        the current policy is intentionally epoch-invariant. Returning an
        explicit per-feature allocation makes the contract auditable instead
        of exposing only a feature scaling vector.
        """
        d = len(sensitivity_vector)
        # This is deliberately deterministic. The previous untrained MLP made
        # privacy allocations depend on random initialization rather than on
        # measured dataset risk, and it was not itself trained privately.
        fingerprint_factor = 1.0 + 0.1 * torch.tanh(dataset_fingerprint.abs().mean())
        raw_weights = sensitivity_vector.clamp(0.0, 1.0) * fingerprint_factor + 1e-3

        feature_scales = torch.exp(1.2 * sensitivity_vector * raw_weights)
        eps_per_feature = self.target_epsilon * F.softmax(raw_weights, dim=0)
        eps_sgd = max(0.05, self.target_epsilon - self.eps_meta)
        sigma_star = self.calibrate_noise_multiplier_accountant(
            target_eps_sgd=eps_sgd,
            sample_rate=sample_rate if sample_rate is not None else 0.00486,
            steps=steps if steps is not None else 500,
            delta=self.target_delta,
        )

        info = {
            "target_epsilon": self.target_epsilon,
            "eps_meta": self.eps_meta,
            "eps_sgd": eps_sgd,
            "target_delta": self.target_delta,
            "noise_multiplier": sigma_star,
            "epoch": epoch,
            "allocated_sum": float(eps_per_feature.sum().detach().cpu()),
            "training_budget": eps_sgd,
        }

        return eps_per_feature, torch.tensor([sigma_star], device=sensitivity_vector.device), info

    def marginal_noise_multiplier(
        self,
        num_samples: int,
        cells_per_table: int,
        num_tables: int,
        total_epsilon: float,
        delta: float,
        share: Optional[float] = None,
        target_relative_noise: float = 0.25,
        max_share: float = 0.6,
    ) -> float:
        """
        DAPC budget split between Stage 7c marginals and DP-SGD (public inputs only).

        share=float: fixed split — z such that the marginals alone cost share * total_epsilon.
        share=None (adaptive): pick the noise that keeps the per-cell noise std at
        `target_relative_noise` of the average cell count n / K, i.e.
            z sqrt(M) = r n / K   =>   z = r n / (K sqrt(M)),
        so large cohorts buy accurate marginals cheaply and hand the rest to the flow, while
        small cohorts spend more on marginals — capped at max_share of the budget. n, K, M are
        public (A2 + public schema), so the split itself costs nothing.
        """
        acc = PRVAccountant(target_delta=delta)

        def z_for(eps_target: float) -> float:
            lo, hi = 0.05, 1.0
            while acc.compute_epsilon(hi, 1.0, 1) > eps_target and hi < 1e5:
                hi *= 2.0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                lo, hi = (mid, hi) if acc.compute_epsilon(mid, 1.0, 1) > eps_target else (lo, mid)
            return hi

        if share is not None:
            return z_for(float(share) * total_epsilon)
        z_target = target_relative_noise * num_samples / (cells_per_table * np.sqrt(max(1, num_tables)))
        return float(max(z_target, z_for(max_share * total_epsilon)))
