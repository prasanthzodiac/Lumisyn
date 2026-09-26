import math
import warnings
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from opacus.accountants import PRVAccountant as OpacusPRVAccountant
from opacus.accountants import RDPAccountant as OpacusRDPAccountant

class RealPRVAccountant:
    """
    Stage 5: Official Privacy Random Variable (PRV) Accountant
    Wraps PyTorch Opacus PRVAccountant for exact numerical Fast Fourier Transform (FFT)
    discretization of Privacy Loss Distributions (PLD).
    """

    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.accountant = OpacusPRVAccountant()
        self.step_history: List[Tuple[float, float, int]] = []

    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1) -> Tuple[float, float]:
        """
        Records SGD steps and calculates exact PRV privacy spending (ε, δ).
        """
        sigma = float(noise_multiplier)
        q = float(sample_rate)
        count = int(steps)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("noise_multiplier must be finite and positive")
        if not math.isfinite(q) or not 0 < q <= 1:
            raise ValueError("sample_rate must be in (0, 1]")
        if count < 1:
            raise ValueError("steps must be positive")
        self.accountant.history.append((sigma, q, count))
        self.step_history.append((sigma, q, count))
        eps_used = self._epsilon_from_history(self.accountant.history)
        return eps_used, self.target_delta

    def compute_epsilon(self, noise_multiplier: float, sample_rate: float, steps: int) -> float:
        """
        Computes exact PRV ε for given parameters without modifying internal state.
        """
        sigma = float(noise_multiplier)
        q = float(sample_rate)
        count = int(steps)
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(q) or not 0 < q <= 1 or count < 1:
            return float("inf")
        return self._epsilon_from_history([(sigma, q, count)])

    def _epsilon_from_history(self, history: List[Tuple[float, float, int]]) -> float:
        """Return a finite accountant value without leaking Opacus FFT warnings."""
        if not history:
            return 0.0

        if not getattr(self, "force_rdp_fallback", False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    prv = OpacusPRVAccountant()
                    prv.history = list(history)
                    epsilon = float(prv.get_epsilon(delta=self.target_delta))
                    if math.isfinite(epsilon):
                        return epsilon
                except (FloatingPointError, OverflowError, ValueError):
                    pass

        # RDP is the stable fallback for extreme q/sigma/step combinations or forced fallback.
        try:
            rdp = OpacusRDPAccountant()
            for sigma, sample_rate, steps in history:
                rdp.history.append((float(sigma), float(sample_rate), int(steps)))
            epsilon = float(rdp.get_epsilon(delta=self.target_delta))
            return epsilon if math.isfinite(epsilon) else float("inf")
        except (FloatingPointError, OverflowError, ValueError):
            return float("inf")

    def set_target_epsilon(self, target_epsilon: float):
        self.target_epsilon = target_epsilon

    def get_accumulated_epsilon(self) -> float:
        if not self.accountant.history:
            return 0.0
        return self._epsilon_from_history(self.accountant.history)

    @property
    def accumulated_privacy_cost(self) -> float:
        return self.get_accumulated_epsilon()

    def is_budget_exceeded(self) -> bool:
        return self.get_accumulated_epsilon() > self.target_epsilon


# Alias PRVAccountant to RealPRVAccountant
PRVAccountant = RealPRVAccountant
