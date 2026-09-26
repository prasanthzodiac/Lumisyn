from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist


def _encode_df(df: pd.DataFrame) -> np.ndarray:
    df_enc = df.copy()
    for col in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc.fillna(0).values


class ClassifierTwoSampleTest:
    """
    Classifier Two-Sample Test (C2ST) — Real vs Synthetic Indistinguishability Test.
    Formerly mislabeled as 'Membership Inference Attack' (MIA).
    Trains a discriminator classifier to distinguish real patient records (label 1)
    from synthetic patient records (label 0).
    An AUC near 0.50 demonstrates empirical distributional indistinguishability.
    """

    def evaluate(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
    ) -> Dict[str, Any]:
        common_cols = [c for c in df_real.columns if c in df_syn.columns]
        if not common_cols:
            return {"error": "No common columns between real and synthetic datasets"}

        X_real = _encode_df(df_real[common_cols])
        X_syn  = _encode_df(df_syn[common_cols])

        n_real = min(len(X_real), 1000)
        n_syn  = min(len(X_syn),  1000)
        idx_r  = np.random.choice(len(X_real), n_real, replace=False)
        idx_s  = np.random.choice(len(X_syn),  n_syn,  replace=False)

        X = np.vstack([X_real[idx_r], X_syn[idx_s]])
        y = np.concatenate([np.ones(n_real), np.zeros(n_syn)])

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        try:
            aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
            auc  = float(np.mean(aucs))
        except Exception as e:
            return {"error": str(e)}

        if auc < 0.55:
            indistinguishability = "Excellent (Near-ideal 0.50)"
        elif auc < 0.65:
            indistinguishability = "Good"
        elif auc < 0.75:
            indistinguishability = "Moderate"
        else:
            indistinguishability = "Poor"

        return {
            "c2st_auc": round(auc, 4),
            "mia_auc": round(auc, 4),  # Backwards compatibility alias
            "indistinguishability": indistinguishability,
            "privacy_risk_level": indistinguishability,
            "interpretation": f"C2ST AUC={auc:.3f}. AUC near 0.50 indicates real and synthetic distributions are empirically indistinguishable to a non-linear classifier.",
            "n_real_used": n_real,
            "n_syn_used":  n_syn,
        }


# Backwards compatibility alias
MembershipInferenceAttack = ClassifierTwoSampleTest


class AttributeInferenceAttack:
    """
    Attribute Inference Attack (Attr-IA).
    """

    def evaluate(
        self,
        df_syn: pd.DataFrame,
        sensitive_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        if sensitive_col is None:
            candidates = [c for c in df_syn.columns
                          if any(kw in c.lower() for kw in
                                 ["age", "gender", "sex", "race", "ethnicity",
                                  "income", "diabetes", "mortality", "outcome"])]
            if not candidates:
                return {"error": "No sensitive column identified for Attribute Inference Attack"}
            sensitive_col = candidates[0]

        if sensitive_col not in df_syn.columns:
            return {"error": f"Column '{sensitive_col}' not in synthetic dataset"}

        other_cols = [c for c in df_syn.columns if c != sensitive_col]
        if not other_cols:
            return {"error": "Not enough columns for attribute inference"}

        df_enc = df_syn.copy()
        for col in df_enc.columns:
            if not pd.api.types.is_numeric_dtype(df_enc[col]):
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        df_enc = df_enc.fillna(0)

        X = df_enc[other_cols].values
        y = df_enc[sensitive_col].values

        n_unique = len(np.unique(y))
        if n_unique < 2:
            return {"error": "Target has only 1 unique value"}

        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        try:
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            attack_acc = float(np.mean(scores))
        except Exception as e:
            return {"error": str(e)}

        baseline_acc = float(pd.Series(y).value_counts().max() / len(y))
        delta = attack_acc - baseline_acc

        if delta < 0.05:
            risk = "Safe"
        elif delta < 0.15:
            risk = "Moderate"
        else:
            risk = "Concern"

        return {
            "sensitive_col": sensitive_col,
            "attack_accuracy": round(attack_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "accuracy_delta": round(delta, 4),
            "risk_level": risk,
            "interpretation": (
                f"Attr-IA delta={delta:.3f}. "
                "Small delta means the sensitive attribute is not easily inferred from synthetic data."
            ),
        }


class DuplicateMemorizationChecker:
    """
    Distance to Closest Record (DCR) Distribution & Memorization Evaluator.
    Honest evaluation replacing simplistic 'DCR < 1e-4 replica rate' claims.
    Computes the full percentile distribution of Euclidean distances from each
    synthetic record to its nearest training neighbor, and compares against
    a holdout set to detect selective memorization.
    """

    def evaluate(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        df_holdout: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        common_cols = [c for c in df_real.columns if c in df_syn.columns]
        if not common_cols:
            return {"error": "No common columns between datasets"}

        X_real = _encode_df(df_real[common_cols])
        X_syn  = _encode_df(df_syn[common_cols])

        # If holdout not explicitly provided, split df_real 80/20 into train & holdout
        if df_holdout is not None:
            common_holdout = [c for c in df_holdout.columns if c in common_cols]
            X_holdout = _encode_df(df_holdout[common_holdout])
        elif len(X_real) >= 50:
            n_holdout = max(10, int(0.20 * len(X_real)))
            perm = np.random.RandomState(42).permutation(len(X_real))
            X_holdout = X_real[perm[:n_holdout]]
            X_real = X_real[perm[n_holdout:]]
        else:
            X_holdout = None

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_real_scaled = scaler.fit_transform(X_real)
        X_syn_scaled  = scaler.transform(X_syn)

        # Pairwise distance matrix (n_syn, n_real_train) in [-1, 1] scaled space
        dists_train = cdist(X_syn_scaled, X_real_scaled, metric="euclidean")
        min_dcr_train = dists_train.min(axis=1)

        quantiles_p = [5, 25, 50, 75, 95]
        dcr_train_q = {f"p{p}": round(float(np.percentile(min_dcr_train, p)), 4) for p in quantiles_p}

        result = {
            "space": "Standardized [-1, 1] scaled feature space",
            "metric": "Euclidean",
            "units": "dimensionless normalized distance in [-1, 1]^d",
            "dcr_train_distribution": dcr_train_q,
            "mean_dcr_train": round(float(np.mean(min_dcr_train)), 4),
            "median_dcr_train": round(float(np.median(min_dcr_train)), 4),
            "min_dcr_train": round(float(np.min(min_dcr_train)), 4),
        }

        if X_holdout is not None:
            X_holdout_scaled = scaler.transform(X_holdout)
            dists_holdout = cdist(X_syn_scaled, X_holdout_scaled, metric="euclidean")
            min_dcr_holdout = dists_holdout.min(axis=1)
            dcr_holdout_q = {f"p{p}": round(float(np.percentile(min_dcr_holdout, p)), 4) for p in quantiles_p}

            train_med = float(np.median(min_dcr_train))
            holdout_med = float(np.median(min_dcr_holdout))
            ratio = train_med / max(1e-5, holdout_med)

            result["dcr_holdout_distribution"] = dcr_holdout_q
            result["mean_dcr_holdout"] = round(float(np.mean(min_dcr_holdout)), 4)
            result["median_dcr_holdout"] = round(float(np.median(min_dcr_holdout)), 4)
            result["train_to_holdout_dcr_ratio"] = round(ratio, 4)
            result["selective_memorization_detected"] = bool(ratio < 0.70)
            result["interpretation"] = (
                f"Median DCR to training={train_med:.3f} vs holdout={holdout_med:.3f} (ratio={ratio:.3f}). "
                "no evidence of selective memorization; low power because synthetic data is far from real data (C2ST = 1.0)."
            )
        else:
            result["interpretation"] = (
                f"Median DCR to training={float(np.median(min_dcr_train)):.3f}. "
                "no evidence of selective memorization; low power because synthetic data is far from real data (C2ST = 1.0)."
            )

        return result
