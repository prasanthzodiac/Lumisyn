import numpy as np
from typing import Tuple, Dict, Any, List
from opacus.accountants import RDPAccountant as OpacusRDPAccountant

class RealRDPAccountant:
    """
    Stage 5: Analytical Rényi Differential Privacy (RDP) Accountant
    Wraps PyTorch Opacus RDPAccountant for fast analytical moment composition across Rényi orders α.
    Used for real-time dynamic early stopping during training.
    """

    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.accountant = OpacusRDPAccountant()

    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1) -> Tuple[float, float]:
        self.accountant.history.append((float(noise_multiplier), float(sample_rate), int(steps)))
        eps_used = float(self.accountant.get_epsilon(delta=self.target_delta))
        return eps_used, self.target_delta

    def compute_epsilon(self, noise_multiplier: float, sample_rate: float, steps: int) -> float:
        acc = OpacusRDPAccountant()
        acc.history = [(float(noise_multiplier), float(sample_rate), int(steps))]
        return float(acc.get_epsilon(delta=self.target_delta))

    def get_accumulated_epsilon(self) -> float:
        if not self.accountant.history:
            return 0.0
        return float(self.accountant.get_epsilon(delta=self.target_delta))

RDPAccountant = RealRDPAccountant
