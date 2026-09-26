from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


CLINICAL_RANGES = [
    ("age",                  0,   130,  True),
    ("glucose",             20,   600,  False),
    ("fasting_glucose",     20,   600,  False),
    ("hba1c",                2,    20,  False),
    ("bmi",                 10,    80,  False),
    ("systolic_bp",         50,   250,  False),
    ("diastolic_bp",        30,   150,  False),
    ("heart_rate",          20,   300,  False),
    ("cholesterol",         50,   500,  False),
    ("hdl",                 10,   150,  False),
    ("ldl",                 20,   400,  False),
    ("creatinine",        0.2,    15,   False),
    ("lactate",           0.1,    20,   False),
    ("sofa_score",           0,    24,  True),
    ("pregnancies",          0,    20,  True),
    ("skin_thickness",       0,   100,  False),
    ("insulin",              0,  1000,  False),
    ("pedigree",           0.0,    5.0, False),
    ("diabetes_pedigree",  0.0,    5.0, False),
    ("outcome",              0,     1,  True),
    ("mortality",            0,     1,  True),
    ("diabetes",             0,     1,  True),
]

LOGICAL_RULES: List[Tuple[str, str, str, str]] = [
    ("systolic_bp", "diastolic_bp", "a_gt_b", "Systolic BP must be > Diastolic BP"),
    ("fasting_glucose", "hba1c", "a_positive", "Fasting glucose must be positive"),
]

# Known medical dependency rules: (feature_a, feature_b, expected_correlation_sign)
CLINICAL_DEPENDENCIES = [
    ("glucose", "outcome", 1),       # Higher Glucose -> Higher Diabetes Risk
    ("bmi", "outcome", 1),           # Higher BMI -> Higher Diabetes Risk
    ("age", "outcome", 1),           # Higher Age -> Higher Risk
    ("insulin", "glucose", 1),       # Positive correlation
    ("systolic_bp", "bmi", 1),       # Positive correlation
]


class ClinicalConstraintAuditor:
    """
    Stage 7 / Evaluation: Clinical Constraint Auditor & Dependency Validator — v5.1
    Validates:
      1. Clinical ranges & integer constraints (Issues 19, 20)
      2. Cross-feature logical rules (systolic > diastolic)
      3. Medical dependency directionality (Glucose->Outcome, BMI->Outcome, etc.) (Issues 4, 5, 18, 20)
    """

    def _find_col(self, df: pd.DataFrame, keyword: str) -> Optional[str]:
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
        return None

    def _check_range(
        self, series: pd.Series, lo: Optional[float], hi: Optional[float], integer: bool
    ) -> Dict[str, Any]:
        n = len(series)
        result: Dict[str, Any] = {"n_total": n}
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            result["skipped"] = True
            return result

        violations = 0
        if lo is not None:
            below = int((numeric < lo).sum())
            result["below_min"] = below
            violations += below
        if hi is not None:
            above = int((numeric > hi).sum())
            result["above_max"] = above
            violations += above
        if integer:
            frac = int((numeric != numeric.round()).sum())
            result["fractional_values"] = frac
            violations += frac

        result["violations"] = violations
        result["pass_rate"] = round(1.0 - violations / max(len(numeric), 1), 4)
        result["passed"] = violations == 0
        return result

    def _validate_dependencies(
        self, df_real: pd.DataFrame, df_syn: pd.DataFrame
    ) -> Dict[str, Any]:
        dep_results = {}
        validated = 0
        passed_count = 0

        for feat_a, feat_b, expected_sign in CLINICAL_DEPENDENCIES:
            col_a_r = self._find_col(df_real, feat_a)
            col_b_r = self._find_col(df_real, feat_b)
            col_a_s = self._find_col(df_syn, feat_a)
            col_b_s = self._find_col(df_syn, feat_b)

            if not col_a_r or not col_b_r or not col_a_s or not col_b_s:
                continue

            r_a = pd.to_numeric(df_real[col_a_r], errors="coerce")
            r_b = pd.to_numeric(df_real[col_b_r], errors="coerce")
            s_a = pd.to_numeric(df_syn[col_a_s], errors="coerce")
            s_b = pd.to_numeric(df_syn[col_b_s], errors="coerce")

            valid_r = r_a.notna() & r_b.notna()
            valid_s = s_a.notna() & s_b.notna()

            if valid_r.sum() < 5 or valid_s.sum() < 5:
                continue

            corr_real = float(np.corrcoef(r_a[valid_r], r_b[valid_r])[0, 1])
            corr_syn  = float(np.corrcoef(s_a[valid_s], s_b[valid_s])[0, 1])

            # Preserved if sign matches and magnitude difference <= 0.25
            sign_match = (corr_real * corr_syn) > 0 or abs(corr_real) < 0.05
            mag_diff   = abs(corr_real - corr_syn)
            is_preserved = sign_match and mag_diff <= 0.25

            validated += 1
            if is_preserved:
                passed_count += 1

            dep_results[f"{feat_a}_vs_{feat_b}"] = {
                "real_correlation": round(corr_real, 4),
                "syn_correlation": round(corr_syn, 4),
                "sign_matched": sign_match,
                "corr_diff": round(mag_diff, 4),
                "dependency_preserved": is_preserved,
            }

        return {
            "dependencies": dep_results,
            "total_validated": validated,
            "total_preserved": passed_count,
            "dependency_pass_rate": round(passed_count / max(1, validated), 4),
        }

    def audit(self, df_syn: pd.DataFrame, df_real: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        per_constraint: Dict[str, Any] = {}
        total_checks = 0
        total_passed = 0

        # 1. Range & integer checks
        for kw, lo, hi, integer in CLINICAL_RANGES:
            col = self._find_col(df_syn, kw)
            if col is None:
                continue
            res = self._check_range(df_syn[col], lo, hi, integer)
            per_constraint[f"{col}_range"] = {
                "column": col,
                "min_allowed": lo,
                "max_allowed": hi,
                "integer_required": integer,
                **res,
            }
            total_checks += 1
            if res.get("passed", False):
                total_passed += 1

        # 2. Logical rules
        logical_results: Dict[str, Any] = {}
        for kw_a, kw_b, rule, desc in LOGICAL_RULES:
            col_a = self._find_col(df_syn, kw_a)
            col_b = self._find_col(df_syn, kw_b)
            if col_a is None or col_b is None:
                continue
            a = pd.to_numeric(df_syn[col_a], errors="coerce")
            b = pd.to_numeric(df_syn[col_b], errors="coerce")
            valid = a.notna() & b.notna()
            a_v, b_v = a[valid], b[valid]
            n_valid = int(valid.sum())

            violations = int((a_v <= b_v).sum()) if rule == "a_gt_b" else int((a_v <= 0).sum())
            passed = violations == 0
            logical_results[f"{col_a}_vs_{col_b}"] = {
                "rule": desc,
                "n_evaluated": n_valid,
                "violations": violations,
                "pass_rate": round(1.0 - violations / max(n_valid, 1), 4),
                "passed": passed,
            }
            total_checks += 1
            total_passed += int(passed)

        # 3. Clinical dependencies (if df_real provided)
        dep_info = {}
        if df_real is not None:
            dep_info = self._validate_dependencies(df_real, df_syn)

        overall_rate = round(total_passed / max(total_checks, 1), 4)

        return {
            "range_constraints": per_constraint,
            "logical_constraints": logical_results,
            "clinical_dependencies": dep_info,
            "summary": {
                "total_constraints_checked": total_checks,
                "total_passed": total_passed,
                "constraint_satisfaction_rate": overall_rate,
                "clinical_consistency_score": round(overall_rate * 100, 2),
                "fully_compliant": overall_rate == 1.0,
            }
        }
