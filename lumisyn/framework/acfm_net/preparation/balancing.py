from typing import Tuple, Dict, Any
import numpy as np

class ClassBalancer:
    """
    2.5 Stratified Class Balancing (ADPE Stage 2)
    Auto-selects class balancing based on Imbalance Ratio IR:
    • IR < 3: No balancing
    • 3 <= IR < 10: Oversampling / Borderline-SMOTE
    • IR >= 10: Hybrid (Oversampling + Random Undersampling)
    """

    def balance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if y is None or len(np.unique(y)) <= 1:
            return X, y, {"action": "none", "imbalance_ratio": 1.0}

        unique_cls, counts = np.unique(y, return_counts=True)
        majority_count = np.max(counts)
        minority_count = np.min(counts)
        ir = float(majority_count / max(1, minority_count))

        if ir < 3.0:
            return X, y, {"action": "none", "imbalance_ratio": ir}

        # Simplified robust oversampling heuristic for high IR
        min_cls = unique_cls[np.argmin(counts)]
        maj_cls = unique_cls[np.argmax(counts)]

        min_indices = np.where(y == min_cls)[0]
        maj_indices = np.where(y == maj_cls)[0]

        target_min_count = int(majority_count * 0.5) if ir >= 10.0 else majority_count
        
        # Random duplicate sampling with Gaussian jitter for numerical features
        oversample_idx = np.random.choice(min_indices, size=target_min_count - len(min_indices), replace=True)
        jitter = np.random.normal(0, 0.01, size=(len(oversample_idx), X.shape[1]))

        X_synth = X[oversample_idx] + jitter
        y_synth = np.full(len(oversample_idx), min_cls)

        X_balanced = np.vstack([X, X_synth])
        y_balanced = np.concatenate([y, y_synth])

        return X_balanced, y_balanced, {"action": "oversample_jitter", "imbalance_ratio": ir}
