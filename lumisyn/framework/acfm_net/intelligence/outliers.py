from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class OutlierDetectionEngine:
    """
    1.5 Multi-method Outlier Detection Engine (UDIE Stage 1)
    Ensemble outlier detection: Z-score ∪ IQR ∪ Isolation Forest ∪ LOF ∪ ECOD (Empirical CDF)
    Distinguishes Clinical Outliers (real signal) vs Statistical Outliers vs Entry Errors.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        clinical_ranges: Dict[str, Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty or len(numeric_df) < 5:
            return {
                "outlier_scores": np.zeros(len(df)),
                "outlier_mask": np.zeros(len(df), dtype=bool),
                "clinical_outliers": [],
                "statistical_outliers": [],
                "entry_errors": []
            }

        num_samples = len(df)
        filled_df = numeric_df.fillna(numeric_df.median())

        # Method 1: Z-score
        z_scores = np.abs((filled_df - filled_df.mean()) / (filled_df.std() + 1e-8))
        z_outliers = (z_scores > 3.0).any(axis=1).astype(float)

        # Method 2: IQR
        q1 = filled_df.quantile(0.25)
        q3 = filled_df.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = ((filled_df < (q1 - 1.5 * iqr)) | (filled_df > (q3 + 1.5 * iqr))).any(axis=1).astype(float)

        # Method 3: Isolation Forest
        try:
            iso = IsolationForest(contamination=0.05, random_state=42)
            iso_preds = iso.fit_predict(filled_df)
            iso_outliers = (iso_preds == -1).astype(float)
        except Exception:
            iso_outliers = np.zeros(num_samples)

        # Method 4: Local Outlier Factor
        try:
            lof = LocalOutlierFactor(n_neighbors=min(20, num_samples - 1), novelty=False)
            lof_preds = lof.fit_predict(filled_df)
            lof_outliers = (lof_preds == -1).astype(float)
        except Exception:
            lof_outliers = np.zeros(num_samples)

        # Method 5: ECOD (Empirical CDF Outlier Detection)
        ecod_scores = np.zeros(num_samples)
        for col in filled_df.columns:
            vals = filled_df[col].values
            ranks = pd.Series(vals).rank(pct=True).values
            tail_probs = np.minimum(ranks, 1.0 - ranks)
            ecod_scores += -np.log(tail_probs + 1e-6)
        ecod_outliers = (ecod_scores > np.percentile(ecod_scores, 95)).astype(float)

        # Weighted Ensemble Voting
        final_scores = (
            0.20 * z_outliers +
            0.20 * iqr_outliers +
            0.25 * iso_outliers +
            0.20 * lof_outliers +
            0.15 * ecod_outliers
        )
        outlier_mask = final_scores > 0.45

        # Differentiate Clinical vs Statistical vs Data Entry Errors
        clinical_outliers = []
        statistical_outliers = []
        entry_errors = []

        clinical_ranges = clinical_ranges or {
            "age": (0, 120),
            "glucose": (20, 600),
            "heart_rate": (20, 250),
            "systolic_bp": (40, 300),
            "diastolic_bp": (20, 200),
            "bmi": (10, 80)
        }

        for col in numeric_df.columns:
            col_lower = col.lower()
            vals = numeric_df[col].dropna()
            if col_lower in clinical_ranges:
                low_b, high_b = clinical_ranges[col_lower]
                err_idx = numeric_df[(numeric_df[col] < low_b) | (numeric_df[col] > high_b)].index.tolist()
                entry_errors.extend([(idx, col) for idx in err_idx])

            # Statistical outliers within bounds
            stat_idx = numeric_df[(z_scores[col] > 3.5) & (~numeric_df[col].isna())].index.tolist()
            for idx in stat_idx:
                if (idx, col) not in entry_errors:
                    statistical_outliers.append((idx, col))

        return {
            "outlier_scores": final_scores,
            "outlier_mask": outlier_mask,
            "clinical_outliers": clinical_outliers,
            "statistical_outliers": statistical_outliers,
            "entry_errors": entry_errors
        }
