"""
Shared evaluation helpers for L-OTFM synthetic data: downstream utility (TSTR/TRTR),
anti-memorization (DCR), and fidelity (correlation error). Used by both the benchmark
runner (run_hard_stress_test.py) and the terminal dashboard so the two never drift.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from scipy.spatial.distance import cdist


def evaluate_tstr(
    df_real_train: pd.DataFrame,
    df_real_test: pd.DataFrame,
    df_syn: pd.DataFrame,
    target_col: str,
    task: str,
) -> Dict[str, Any]:
    """Train on Synthetic, Test on Real (TSTR) vs Train on Real, Test on Real (TRTR)."""
    feature_cols = [c for c in df_real_train.columns if c != target_col]

    syn_cols = [c for c in feature_cols if c in df_syn.columns]
    df_combined = pd.concat([df_real_train[syn_cols], df_real_test[syn_cols], df_syn[syn_cols]], axis=0, ignore_index=True)
    df_dummies = pd.get_dummies(df_combined, drop_first=True)

    n_train = len(df_real_train)
    n_test = len(df_real_test)

    X_train_real = np.nan_to_num(df_dummies.iloc[:n_train].values, nan=0.0)
    y_train_real = df_real_train[target_col].values

    X_test_real = np.nan_to_num(df_dummies.iloc[n_train:n_train + n_test].values, nan=0.0)
    y_test_real = df_real_test[target_col].values

    X_train_syn = np.nan_to_num(df_dummies.iloc[n_train + n_test:].values, nan=0.0)
    y_train_syn = df_syn[target_col].values if target_col in df_syn.columns else np.zeros(len(df_syn))

    if task == "classification":
        # Non-numeric (string) categorical targets (e.g. Adult's "<=50K"/">50K") must be
        # mapped to a shared 0/1 encoding derived from the REAL train labels before the
        # numeric coercion below — otherwise pd.to_numeric silently NaNs every string
        # value and fillna(0) collapses the whole column to a single class.
        if not pd.api.types.is_numeric_dtype(pd.Series(y_train_real)):
            cats = sorted(pd.Series(y_train_real).astype(str).unique())
            cat_map = {c: i for i, c in enumerate(cats)}
            y_train_real = pd.Series(y_train_real).astype(str).map(cat_map).values
            y_test_real = pd.Series(y_test_real).astype(str).map(cat_map).values
            y_train_syn = pd.Series(y_train_syn).astype(str).map(cat_map).values

        y_train_real_clean = pd.to_numeric(pd.Series(y_train_real), errors="coerce").fillna(0).astype(int).values
        y_test_real_clean = pd.to_numeric(pd.Series(y_test_real), errors="coerce").fillna(0).astype(int).values
        y_train_syn_clean = pd.to_numeric(pd.Series(y_train_syn), errors="coerce").fillna(0).astype(int).values

        if len(np.unique(y_train_syn_clean)) < 2:
            thresh = np.median(y_train_syn)
            y_train_syn_clean = (y_train_syn >= thresh).astype(int)

        clf_real = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_real.fit(X_train_real, y_train_real_clean)
        preds_trtr = clf_real.predict(X_test_real)
        probs_trtr = clf_real.predict_proba(X_test_real)[:, 1] if len(np.unique(y_train_real_clean)) == 2 else None
        trtr_acc = accuracy_score(y_test_real_clean, preds_trtr)
        trtr_auc = roc_auc_score(y_test_real_clean, probs_trtr) if probs_trtr is not None else 0.0

        clf_syn = RandomForestClassifier(n_estimators=100, random_state=42)
        try:
            clf_syn.fit(X_train_syn, y_train_syn_clean)
            preds_tstr = clf_syn.predict(X_test_real)
            probs_tstr = clf_syn.predict_proba(X_test_real)[:, 1] if len(np.unique(y_train_syn_clean)) == 2 else None
            tstr_acc = accuracy_score(y_test_real_clean, preds_tstr)
            tstr_auc = roc_auc_score(y_test_real_clean, probs_tstr) if probs_tstr is not None else 0.0
        except Exception:
            tstr_acc = 0.50
            tstr_auc = 0.50

        retention = f"{tstr_acc / max(1e-5, trtr_acc) * 100:.1f}%"
        return {
            "trtr_metric": f"AUC={trtr_auc:.3f}, Acc={trtr_acc*100:.1f}%",
            "tstr_metric": f"AUC={tstr_auc:.3f}, Acc={tstr_acc*100:.1f}%",
            "retention": retention,
            "tstr_acc": round(float(tstr_acc), 4),
            "trtr_acc": round(float(trtr_acc), 4),
            "tstr_auc": round(float(tstr_auc), 4),
            "trtr_auc": round(float(trtr_auc), 4),
        }
    else:
        y_train_real_num = pd.to_numeric(pd.Series(y_train_real), errors="coerce").fillna(0.0).values
        y_test_real_num = pd.to_numeric(pd.Series(y_test_real), errors="coerce").fillna(0.0).values
        y_train_syn_num = pd.to_numeric(pd.Series(y_train_syn), errors="coerce").fillna(0.0).values

        reg_real = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_real.fit(X_train_real, y_train_real_num)
        trtr_r2 = r2_score(y_test_real_num, reg_real.predict(X_test_real))

        reg_syn = RandomForestRegressor(n_estimators=100, random_state=42)
        try:
            reg_syn.fit(X_train_syn, y_train_syn_num)
            tstr_r2 = r2_score(y_test_real_num, reg_syn.predict(X_test_real))
        except Exception:
            tstr_r2 = 0.0

        retention = f"{max(0.0, tstr_r2) / max(1e-5, trtr_r2) * 100:.1f}%"
        return {
            "trtr_metric": f"R2={trtr_r2:.3f}",
            "tstr_metric": f"R2={tstr_r2:.3f}",
            "retention": retention,
            "tstr_r2": round(float(tstr_r2), 4),
            "trtr_r2": round(float(trtr_r2), 4),
        }


def compute_anti_memorization_dcr(df_real: pd.DataFrame, df_syn: pd.DataFrame) -> Tuple[float, float]:
    """Distance to Closest Record (mean_dcr, exact_match_percentage)."""
    num_cols = [c for c in df_real.columns if c in df_syn.columns and pd.api.types.is_numeric_dtype(df_real[c])]
    if len(num_cols) == 0:
        return 1.0, 0.0

    real_sub = df_real[num_cols].fillna(0.0)
    syn_sub = df_syn[num_cols].fillna(0.0)

    means = real_sub.mean()
    stds = real_sub.std().replace(0.0, 1.0) + 1e-5

    real_mat = ((real_sub - means) / stds).values
    syn_mat = ((syn_sub - means) / stds).values

    n_eval = min(500, len(real_mat), len(syn_mat))
    distances = cdist(syn_mat[:n_eval], real_mat[:n_eval], metric="euclidean")
    min_distances = np.min(distances, axis=1)

    exact_matches = np.sum(min_distances < 1e-4) / float(len(min_distances))
    return float(np.mean(min_distances)), float(exact_matches * 100.0)


def correlation_frobenius_error(df_real: pd.DataFrame, df_syn: pd.DataFrame) -> float:
    """Frobenius-norm error between real and synthetic numeric correlation matrices."""
    num_cols = [c for c in df_real.columns if c in df_syn.columns and pd.api.types.is_numeric_dtype(df_real[c])]
    if len(num_cols) <= 1:
        return 0.0
    corr_real = df_real[num_cols].corr().fillna(0).values
    corr_syn = df_syn[num_cols].corr().fillna(0).values
    return float(np.linalg.norm(corr_real - corr_syn, ord="fro") / (len(num_cols) ** 2))
