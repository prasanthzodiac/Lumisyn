import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    root_mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import StratifiedKFold, KFold


def _encode_for_eval(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    df_enc = df.copy()
    for col in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        else:
            df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce").fillna(0.0)
    return df_enc.values.astype(np.float32), list(df.columns)


class MarginalErrorEvaluator:
    """
    Phase 3 Fidelity Module: 1-way and 2-way marginal distribution error.
    Metric: Total Variation Distance (TVD) = 0.5 * sum_i |p_i - q_i| in [0, 1].
    """

    def evaluate_1way(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        df_real_holdout: Optional[pd.DataFrame] = None,
        n_bins: int = 20,
    ) -> Dict[str, Any]:
        common_cols = [c for c in df_real.columns if c in df_syn.columns and not c.endswith("_isna")]
        per_col_tvd = {}

        for col in common_cols:
            s_real = df_real[col].dropna()
            s_syn = df_syn[col].dropna()
            if len(s_real) == 0 or len(s_syn) == 0:
                continue

            if pd.api.types.is_numeric_dtype(s_real):
                lo = min(float(s_real.min()), float(s_syn.min()))
                hi = max(float(s_real.max()), float(s_syn.max()))
                if hi <= lo:
                    hi = lo + 1.0
                bins = np.linspace(lo, hi, n_bins + 1)
                p, _ = np.histogram(s_real, bins=bins, density=False)
                q, _ = np.histogram(s_syn, bins=bins, density=False)
                p = p / max(1, p.sum())
                q = q / max(1, q.sum())
                tvd = 0.5 * float(np.sum(np.abs(p - q)))
            else:
                cats = sorted(set(s_real.astype(str).unique()).union(set(s_syn.astype(str).unique())))
                p = s_real.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).values
                q = s_syn.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).values
                tvd = 0.5 * float(np.sum(np.abs(p - q)))

            per_col_tvd[col] = round(tvd, 4)

        # Gate 4d Item 4: Bin-independent Kolmogorov-Smirnov (KS) statistic
        from scipy.stats import ks_2samp
        per_col_ks = {}
        for col in common_cols:
            s_real = df_real[col].dropna()
            s_syn = df_syn[col].dropna()
            if pd.api.types.is_numeric_dtype(s_real):
                try:
                    ks_val = float(ks_2samp(s_real.values, s_syn.values).statistic)
                    per_col_ks[col] = round(ks_val, 4)
                except Exception:
                    per_col_ks[col] = 0.0

        mean_tvd = float(np.mean(list(per_col_tvd.values()))) if per_col_tvd else 0.0
        mean_ks = float(np.mean(list(per_col_ks.values()))) if per_col_ks else 0.0

        res = {
            "mean_1way_marginal_error_tvd": round(mean_tvd, 4),
            "mean_ks_statistic": round(mean_ks, 4),
            "per_feature_1way_tvd": per_col_tvd,
            "per_feature_ks": per_col_ks,
            "eval_bins": n_bins,
            "metric": "Total Variation Distance (0.5 * sum |p - q|) and KS statistic",
            "num_features_evaluated": len(per_col_tvd),
        }

        # Real-vs-real holdout floor normalization (Gate 4b Item 4)
        if df_real_holdout is not None and len(df_real_holdout) > 0:
            floor_eval = self.evaluate_1way(df_real, df_real_holdout, df_real_holdout=None, n_bins=n_bins)
            floor_tvd = floor_eval["mean_1way_marginal_error_tvd"]
            floor_ks = floor_eval.get("mean_ks_statistic", 0.01)
            res["real_vs_real_floor_1way_tvd"] = floor_tvd
            res["real_vs_real_floor_ks"] = floor_ks
            res["normalized_1way_marginal_error"] = round(mean_tvd / max(1e-4, floor_tvd), 4)
            res["normalized_ks_statistic"] = round(mean_ks / max(1e-4, floor_ks), 4)

        return res

    def evaluate_2way(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        n_bins: int = 10,
        max_pairs: int = 50,
        seed: int = 42,
    ) -> Dict[str, Any]:
        common_cols = [c for c in df_real.columns if c in df_syn.columns and not c.endswith("_isna")]
        d = len(common_cols)
        if d < 2:
            return {"error": "Need at least 2 features for 2-way marginal evaluation"}

        pairs = []
        for i in range(d):
            for j in range(i + 1, d):
                pairs.append((common_cols[i], common_cols[j]))

        if len(pairs) > max_pairs:
            rng = np.random.default_rng(seed)
            selected_idx = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[idx] for idx in selected_idx]

        pair_tvds = {}
        for c1, c2 in pairs:
            r1 = df_real[c1].dropna()
            r2 = df_real[c2].loc[r1.index].dropna()
            common_idx_r = r2.index
            s1 = df_syn[c1].dropna()
            s2 = df_syn[c2].loc[s1.index].dropna()
            common_idx_s = s2.index

            if len(common_idx_r) < 10 or len(common_idx_s) < 10:
                continue

            r_x = df_real.loc[common_idx_r, c1]
            r_y = df_real.loc[common_idx_r, c2]
            s_x = df_syn.loc[common_idx_s, c1]
            s_y = df_syn.loc[common_idx_s, c2]

            # Categorize / bin
            def _to_discrete(sr, ss):
                if pd.api.types.is_numeric_dtype(sr):
                    lo = min(float(sr.min()), float(ss.min()))
                    hi = max(float(sr.max()), float(ss.max()))
                    if hi <= lo:
                        hi = lo + 1.0
                    bins = np.linspace(lo, hi, n_bins + 1)
                    r_b = np.digitize(sr, bins)
                    s_b = np.digitize(ss, bins)
                    return r_b, s_b, n_bins + 2
                else:
                    le = LabelEncoder()
                    cats = list(set(sr.astype(str)).union(set(ss.astype(str))))
                    le.fit(cats)
                    r_b = le.transform(sr.astype(str))
                    s_b = le.transform(ss.astype(str))
                    return r_b, s_b, len(cats)

            r_disc1, s_disc1, k1 = _to_discrete(r_x, s_x)
            r_disc2, s_disc2, k2 = _to_discrete(r_y, s_y)

            # 2D Joint contingency matrix
            H_r, _, _ = np.histogram2d(r_disc1, r_disc2, bins=[k1, k2])
            H_s, _, _ = np.histogram2d(s_disc1, s_disc2, bins=[k1, k2])

            P = H_r / max(1.0, H_r.sum())
            Q = H_s / max(1.0, H_s.sum())
            tvd = 0.5 * float(np.sum(np.abs(P - Q)))
            pair_tvds[f"{c1}__x__{c2}"] = round(tvd, 4)

        mean_2way = float(np.mean(list(pair_tvds.values()))) if pair_tvds else 0.0
        return {
            "mean_2way_marginal_error_tvd": round(mean_2way, 4),
            "pair_tvds_sample": dict(list(pair_tvds.items())[:10]),
            "num_pairs_evaluated": len(pair_tvds),
            "metric": "Joint 2-Way Total Variation Distance",
        }


class CorrelationErrorEvaluator:
    """
    Phase 3 Fidelity Module: Correlation matrix Frobenius error and off-diagonal MAE.
    """

    def evaluate(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        df_real_holdout: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c]) and c in df_syn.columns and not c.endswith("_isna")]
        if len(num_cols) < 2:
            return {"error": "Need at least 2 numerical columns"}

        corr_real = df_real[num_cols].corr(method="pearson").fillna(0.0).values
        corr_syn  = df_syn[num_cols].corr(method="pearson").fillna(0.0).values

        # Frobenius norm difference
        frob_err = float(np.linalg.norm(corr_real - corr_syn, ord="fro"))

        # Off-diagonal Mean Absolute Error (MAE)
        d = len(num_cols)
        mask = ~np.eye(d, dtype=bool)
        mae_offdiag = float(np.mean(np.abs(corr_real[mask] - corr_syn[mask]))) if d > 1 else 0.0

        # Spearman rank correlation
        spear_real = df_real[num_cols].corr(method="spearman").fillna(0.0).values
        spear_syn  = df_syn[num_cols].corr(method="spearman").fillna(0.0).values
        spear_frob = float(np.linalg.norm(spear_real - spear_syn, ord="fro"))
        spear_mae  = float(np.mean(np.abs(spear_real[mask] - spear_syn[mask]))) if d > 1 else 0.0

        res = {
            "pearson_frobenius_error": round(frob_err, 4),
            "pearson_offdiag_mae": round(mae_offdiag, 4),
            "spearman_frobenius_error": round(spear_frob, 4),
            "spearman_offdiag_mae": round(spear_mae, 4),
            "num_features": d,
        }

        # Real-vs-real holdout floor normalization (Gate 4b Item 4)
        if df_real_holdout is not None and len(df_real_holdout) > 0:
            floor_eval = self.evaluate(df_real, df_real_holdout, df_real_holdout=None)
            floor_frob = floor_eval.get("pearson_frobenius_error", 0.0)
            res["real_vs_real_floor_pearson_frobenius"] = floor_frob
            res["normalized_pearson_frobenius_error"] = round(frob_err / max(1e-4, floor_frob), 4)

        return res


class ClassifierTwoSampleTestHarness:
    """
    Phase 3 Fidelity Module: Classifier Two-Sample Test (C2ST).
    Trains non-linear discriminator to distinguish real vs synthetic records.
    Ideal: AUC = 0.50 (indistinguishable).
    Reports 5-fold CV AUC and 95% Confidence Interval.
    """

    def evaluate(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        n_splits: int = 5,
        max_samples: int = 2000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        common_cols = [c for c in df_real.columns if c in df_syn.columns and not c.endswith("_isna")]
        X_real_enc, _ = _encode_for_eval(df_real[common_cols])
        X_syn_enc, _  = _encode_for_eval(df_syn[common_cols])

        n_sub = min(len(X_real_enc), len(X_syn_enc), max_samples)
        rng = np.random.default_rng(seed)
        idx_r = rng.choice(len(X_real_enc), size=n_sub, replace=False)
        idx_s = rng.choice(len(X_syn_enc),  size=n_sub, replace=False)

        X = np.vstack([X_real_enc[idx_r], X_syn_enc[idx_s]])
        y = np.concatenate([np.ones(n_sub, dtype=np.int32), np.zeros(n_sub, dtype=np.int32)])

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aucs = []

        for train_idx, val_idx in cv.split(X, y):
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y[val_idx], probs)))

        mean_auc = float(np.mean(fold_aucs))
        se_auc = float(np.std(fold_aucs, ddof=1) / math.sqrt(n_splits)) if n_splits > 1 else 0.0
        ci_95 = [round(max(0.0, mean_auc - 1.96 * se_auc), 4), round(min(1.0, mean_auc + 1.96 * se_auc), 4)]

        return {
            "c2st_auc": round(mean_auc, 4),
            "c2st_se": round(se_auc, 4),
            "c2st_95_ci": ci_95,
            "n_samples_per_class": n_sub,
            "ideal_baseline": 0.5000,
            "interpretation": f"C2ST AUC={mean_auc:.4f} (95% CI {ci_95}). 0.50 indicates ideal indistinguishability; 1.0 indicates easily separable distributions.",
        }


class TSTRHarness:
    """
    Phase 3 Fidelity Module: Train on Synthetic, Test on Real (TSTR)
    Evaluates both classification (AUROC, F1-weighted, F1-macro, accuracy)
    and regression (RMSE, MAE, R2). Compares against TRTR (Train on Real, Test on Real) baseline.
    """

    def evaluate(
        self,
        df_real_train: pd.DataFrame,
        df_real_test: pd.DataFrame,
        df_syn: pd.DataFrame,
        target_col: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        assert target_col in df_real_test.columns, f"Target '{target_col}' missing from test set"
        assert target_col in df_syn.columns, f"Target '{target_col}' missing from synthetic set"

        feature_cols = [c for c in df_syn.columns if c != target_col and not c.endswith("_isna") and c in df_real_test.columns]

        # Prepare X, y
        def _prep(df):
            X_enc, _ = _encode_for_eval(df[feature_cols])
            y_raw = df[target_col].values
            return X_enc, y_raw

        X_syn, y_syn = _prep(df_syn)
        X_r_train, y_r_train = _prep(df_real_train)
        X_r_test, y_r_test = _prep(df_real_test)

        is_clf = len(np.unique(y_r_test)) <= 10

        if is_clf:
            # Encode binary targets properly to 0 and 1
            all_classes = np.unique(y_r_train)
            is_binary = len(all_classes) == 2
            if is_binary:
                le_target = LabelEncoder()
                y_r_train = le_target.fit_transform(y_r_train)
                y_r_test = le_target.transform(y_r_test)
                # Map synthetic labels safely
                y_syn_clean = []
                for val in y_syn:
                    if val in le_target.classes_:
                        y_syn_clean.append(le_target.transform([val])[0])
                    else:
                        y_syn_clean.append(0)
                y_syn = np.array(y_syn_clean, dtype=np.int32)

            # TSTR
            clf_tstr = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf_tstr.fit(X_syn, y_syn)
            pred_tstr = clf_tstr.predict(X_r_test)
            acc_tstr = float(accuracy_score(y_r_test, pred_tstr))
            f1_w_tstr = float(f1_score(y_r_test, pred_tstr, average="weighted", zero_division=0))
            f1_m_tstr = float(f1_score(y_r_test, pred_tstr, average="macro", zero_division=0))

            if is_binary:
                try:
                    classes_tstr = list(clf_tstr.classes_)
                    proba_raw_tstr = clf_tstr.predict_proba(X_r_test)
                    if 1 in classes_tstr:
                        idx_pos = classes_tstr.index(1)
                    else:
                        idx_pos = 1 if proba_raw_tstr.shape[1] > 1 else 0
                    proba_tstr = proba_raw_tstr[:, idx_pos]
                    auc_tstr = float(roc_auc_score(y_r_test, proba_tstr))
                    auprc_tstr = float(average_precision_score(y_r_test, proba_tstr))
                except Exception:
                    auc_tstr = 0.5000
                    auprc_tstr = float(np.mean(y_r_test))
            else:
                try:
                    proba_raw_tstr = clf_tstr.predict_proba(X_r_test)
                    y_r_test_bin = label_binarize(y_r_test, classes=clf_tstr.classes_)
                    auc_tstr = float(roc_auc_score(y_r_test_bin, proba_raw_tstr, multi_class="ovr", average="weighted"))
                    auprc_tstr = float(average_precision_score(y_r_test_bin, proba_raw_tstr, average="weighted"))
                except Exception:
                    auc_tstr = 0.5000
                    auprc_tstr = float(np.mean(y_r_test == y_r_test[0]))

            # TRTR Baseline
            clf_trtr = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf_trtr.fit(X_r_train, y_r_train)
            pred_trtr = clf_trtr.predict(X_r_test)
            acc_trtr = float(accuracy_score(y_r_test, pred_trtr))
            f1_w_trtr = float(f1_score(y_r_test, pred_trtr, average="weighted", zero_division=0))
            f1_m_trtr = float(f1_score(y_r_test, pred_trtr, average="macro", zero_division=0))

            if is_binary:
                try:
                    classes_trtr = list(clf_trtr.classes_)
                    proba_raw_trtr = clf_trtr.predict_proba(X_r_test)
                    if 1 in classes_trtr:
                        idx_pos = classes_trtr.index(1)
                    else:
                        idx_pos = 1 if proba_raw_trtr.shape[1] > 1 else 0
                    proba_trtr = proba_raw_trtr[:, idx_pos]
                    auc_trtr = float(roc_auc_score(y_r_test, proba_trtr))
                    auprc_trtr = float(average_precision_score(y_r_test, proba_trtr))
                except Exception:
                    auc_trtr = 0.5000
                    auprc_trtr = float(np.mean(y_r_test))
            else:
                try:
                    proba_raw_trtr = clf_trtr.predict_proba(X_r_test)
                    y_r_test_bin = label_binarize(y_r_test, classes=clf_trtr.classes_)
                    auc_trtr = float(roc_auc_score(y_r_test_bin, proba_raw_trtr, multi_class="ovr", average="weighted"))
                    auprc_trtr = float(average_precision_score(y_r_test_bin, proba_raw_trtr, average="weighted"))
                except Exception:
                    auc_trtr = 0.5000
                    auprc_trtr = float(np.mean(y_r_test == y_r_test[0]))

            return {
                "task": "classification",
                "target_column": target_col,
                "tstr": {
                    "auroc": round(auc_tstr, 4),
                    "auprc": round(auprc_tstr, 4),
                    "f1_weighted": round(f1_w_tstr, 4),
                    "f1_macro": round(f1_m_tstr, 4),
                    "accuracy": round(acc_tstr, 4),
                },
                "trtr_baseline": {
                    "auroc": round(auc_trtr, 4),
                    "auprc": round(auprc_trtr, 4),
                    "f1_weighted": round(f1_w_trtr, 4),
                    "f1_macro": round(f1_m_trtr, 4),
                    "accuracy": round(acc_trtr, 4),
                },
                "utility_ratio_auroc": round(auc_tstr / max(1e-4, auc_trtr), 4),
                "utility_ratio_auprc": round(auprc_tstr / max(1e-4, auprc_trtr), 4),
                "utility_ratio_f1": round(f1_w_tstr / max(1e-4, f1_w_trtr), 4),
            }
        else:
            # Regression: RandomForest
            # TSTR
            reg_tstr = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            reg_tstr.fit(X_syn, y_syn)
            pred_tstr = reg_tstr.predict(X_r_test)
            rmse_tstr = float(root_mean_squared_error(y_r_test, pred_tstr))
            mae_tstr = float(mean_absolute_error(y_r_test, pred_tstr))
            r2_tstr = float(r2_score(y_r_test, pred_tstr))

            # TRTR
            reg_trtr = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            reg_trtr.fit(X_r_train, y_r_train)
            pred_trtr = reg_trtr.predict(X_r_test)
            rmse_trtr = float(root_mean_squared_error(y_r_test, pred_trtr))
            mae_trtr = float(mean_absolute_error(y_r_test, pred_trtr))
            r2_trtr = float(r2_score(y_r_test, pred_trtr))

            return {
                "task": "regression",
                "target_column": target_col,
                "tstr": {
                    "rmse": round(rmse_tstr, 4),
                    "mae": round(mae_tstr, 4),
                    "r2": round(r2_tstr, 4),
                },
                "trtr_baseline": {
                    "rmse": round(rmse_trtr, 4),
                    "mae": round(mae_trtr, 4),
                    "r2": round(r2_trtr, 4),
                },
                "rmse_gap": round(rmse_tstr - rmse_trtr, 4),
            }


class ClinicalConstraintEvaluator:
    """
    Phase 3 Fidelity Module: Clinical constraint violation audit.
    Checks physical range validity and logical clinical invariants (e.g. systolic > diastolic + 10).
    """

    def evaluate(
        self,
        df_syn: pd.DataFrame,
        schema_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        N = len(df_syn)
        if N == 0:
            return {"error": "Synthetic dataframe is empty"}

        violations_per_record = np.zeros(N, dtype=np.int32)
        rule_violations = {}

        # Default clinical ranges if not supplied
        bounds = schema_bounds or {
            "age": (0.0, 120.0),
            "bmi": (10.0, 70.0),
            "systolic": (60.0, 250.0),
            "diastolic": (30.0, 150.0),
            "glucose": (20.0, 600.0),
            "creatinine": (0.1, 25.0),
        }

        # 1. Range bounds checks
        for col, (lo, hi) in bounds.items():
            if col in df_syn.columns:
                series = pd.to_numeric(df_syn[col], errors="coerce").dropna()
                viol = ((series < lo) | (series > hi)).values
                viol_count = int(np.sum(viol))
                viol_indices = series.index[viol]
                violations_per_record[viol_indices] += 1
                rule_violations[f"range_{col}_[{lo},{hi}]"] = {
                    "violation_count": viol_count,
                    "violation_rate": round(float(viol_count / max(1, len(series))), 4),
                }

        # 2. Logical consistency rule: systolic BP > diastolic BP + 10 mmHg
        if "systolic" in df_syn.columns and "diastolic" in df_syn.columns:
            valid_idx = df_syn[["systolic", "diastolic"]].dropna().index
            sys_val = pd.to_numeric(df_syn.loc[valid_idx, "systolic"], errors="coerce")
            dia_val = pd.to_numeric(df_syn.loc[valid_idx, "diastolic"], errors="coerce")
            pulse_viol = (sys_val <= dia_val + 10.0).values
            pulse_count = int(np.sum(pulse_viol))
            violations_per_record[valid_idx[pulse_viol]] += 1
            rule_violations["logical_systolic_gt_diastolic_plus_10"] = {
                "violation_count": pulse_count,
                "violation_rate": round(float(pulse_count / max(1, len(valid_idx))), 4),
            }

        total_violating_records = int(np.sum(violations_per_record > 0))
        overall_violation_rate = float(total_violating_records / N)

        return {
            "overall_record_violation_rate": round(overall_violation_rate, 4),
            "overall_record_pass_rate": round(1.0 - overall_violation_rate, 4),
            "total_records_evaluated": N,
            "violating_records_count": total_violating_records,
            "rule_breakdown": rule_violations,
        }


class SlicedWassersteinEvaluator:
    """
    Evaluates Sliced Wasserstein Distance between real and synthetic continuous representations.
    Gate 4d Item A2: Standardizes each column by the real-train std before computing SWD,
    and reports per-column 1D W1 in std units.
    """
    def __init__(self, num_projections: int = 100):
        self.num_projections = num_projections

    def evaluate(
        self,
        X_real: np.ndarray,
        X_syn: np.ndarray,
        seed: int = 42,
        standardize: bool = True,
        real_std: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        d = X_real.shape[1]
        assert X_syn.shape[1] == d, f"Dimension mismatch: real has {d}, syn has {X_syn.shape[1]}"

        if standardize:
            if real_std is None:
                std = np.std(X_real, axis=0)
            else:
                std = np.array(real_std, dtype=np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            Xr = X_real / std
            Xs = X_syn / std
        else:
            Xr = X_real
            Xs = X_syn

        rng = np.random.RandomState(seed)
        proj = rng.randn(self.num_projections, d)
        proj /= np.maximum(1e-9, np.linalg.norm(proj, axis=1, keepdims=True))

        p_real = np.sort(Xr @ proj.T, axis=0)
        p_syn = np.sort(Xs @ proj.T, axis=0)
        min_n = min(len(p_real), len(p_syn))
        swd = float(np.mean(np.abs(p_real[:min_n] - p_syn[:min_n])))

        # Gate 4d Item A2: Per-column 1D W1 in std units
        per_col_w1 = {}
        for j in range(d):
            c_real = np.sort(Xr[:, j])
            c_syn = np.sort(Xs[:, j])
            w1_j = float(np.mean(np.abs(c_real[:min_n] - c_syn[:min_n])))
            cname = feature_names[j] if feature_names and j < len(feature_names) else f"col_{j}"
            per_col_w1[cname] = round(w1_j, 4)

        mean_1d_w1 = float(np.mean(list(per_col_w1.values()))) if per_col_w1 else 0.0

        return {
            "sliced_wasserstein_distance": round(swd, 6),
            "mean_1d_w1": round(mean_1d_w1, 4),
            "per_column_1d_w1": per_col_w1,
            "projections": self.num_projections,
            "standardized": standardize,
        }
