from typing import Dict, Any, List, Optional
import numpy as np

class ParetoModelSelector:
    """
    8.4 Pareto Model Selector (Optimization Stage 8) ★ v5 REDESIGNED ★
    Selects the optimal model checkpoint based on:
    max U(X_syn, X_real) subject to ε_used <= ε*
    Utility score U is computed across statistical similarity, correlation retention, and machine learning efficacy.
    """

    def __init__(self, target_epsilon: float = 1.0):
        self.target_epsilon = target_epsilon
        self.best_checkpoint_info = None
        self.best_utility_score = -1.0

    def evaluate_checkpoint(
        self,
        epoch: int,
        model_state_dict: Dict[str, Any],
        eps_used: float,
        swd_metric: float,
        corr_error: float,
        ml_utility_score: float = 0.85
    ) -> bool:
        """
        Evaluates model checkpoint at given epoch.
        Returns True if this checkpoint sets a new Pareto optimal record.
        """
        if eps_used > self.target_epsilon:
            # DP violation - reject checkpoint
            return False

        # Composite Utility Score U ∈ [0, 1]
        # Higher score means better utility
        stat_utility = max(0.0, 1.0 - swd_metric)
        corr_utility = max(0.0, 1.0 - corr_error)
        overall_utility = 0.4 * stat_utility + 0.3 * corr_utility + 0.3 * ml_utility_score

        if overall_utility > self.best_utility_score:
            self.best_utility_score = overall_utility
            self.best_checkpoint_info = {
                "epoch": epoch,
                "utility_score": float(overall_utility),
                "eps_used": float(eps_used),
                "swd_metric": float(swd_metric),
                "corr_error": float(corr_error),
                "model_state_dict": model_state_dict
            }
            return True

        return False
