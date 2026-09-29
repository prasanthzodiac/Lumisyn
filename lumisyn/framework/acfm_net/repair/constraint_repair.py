import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import rankdata, norm


class ConstrainedGradientRepairLoop:
    """
    STAGE 7: Constrained Gradient Descent Repair Loop — v5.1
    Enforces hard clinical and physical domain constraints with FINAL dtype enforcement.
    """

    def __init__(
        self,
        max_repair_steps: int = 15,
        lr: float = 0.05,
        clinical_bounds: Dict[int, Tuple[float, float]] = None,
        relational_pairs: List[Tuple[int, int]] = None,
        column_kinds: Dict[int, str] = None,
    ):
        self.max_repair_steps = max_repair_steps
        self.lr = lr
        self.clinical_bounds  = clinical_bounds  or {}
        self.relational_pairs = relational_pairs or []
        self.column_kinds     = column_kinds     or {}

    def repair(
        self,
        x_gen: torch.Tensor,
        column_kinds: Optional[Dict[int, str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        kinds = column_kinds if column_kinds is not None else self.column_kinds

        x_repaired = x_gen.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x_repaired], lr=self.lr)

        initial_violations = self._compute_violations(x_repaired)

        for step in range(self.max_repair_steps):
            optimizer.zero_grad()
            penalty = self._penalty_function(x_repaired)
            if penalty.item() < 1e-6:
                break
            penalty.backward()
            optimizer.step()

            with torch.no_grad():
                for idx, (lo, hi) in self.clinical_bounds.items():
                    if idx < x_repaired.shape[1]:
                        x_repaired[:, idx].clamp_(lo, hi)

        final_violations = self._compute_violations(x_repaired)

        stats = {
            "initial_violations": initial_violations,
            "final_violations": final_violations,
            "repaired_count": initial_violations - final_violations,
        }
        return x_repaired.detach(), stats

    def _penalty_function(self, x: torch.Tensor) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=x.device)
        for idx, (lo, hi) in self.clinical_bounds.items():
            if idx < x.shape[1]:
                col = x[:, idx]
                under = F.relu(lo - col)
                over  = F.relu(col - hi)
                penalty = penalty + torch.mean(under.pow(2) + over.pow(2))

        for sys_idx, dia_idx in self.relational_pairs:
            if sys_idx < x.shape[1] and dia_idx < x.shape[1]:
                diff = x[:, sys_idx] - (x[:, dia_idx] + 10.0)
                penalty = penalty + torch.mean(F.relu(-diff).pow(2))

        return penalty

    def _compute_violations(self, x: torch.Tensor) -> int:
        violations = 0
        with torch.no_grad():
            for idx, (lo, hi) in self.clinical_bounds.items():
                if idx < x.shape[1]:
                    mask = (x[:, idx] < lo) | (x[:, idx] > hi)
                    violations += int(mask.sum().item())
            for sys_idx, dia_idx in self.relational_pairs:
                if sys_idx < x.shape[1] and dia_idx < x.shape[1]:
                    mask = x[:, sys_idx] < (x[:, dia_idx] + 10.0)
                    violations += int(mask.sum().item())
        return violations


# =============================================================================
# MULTIVARIATE RANK-PRESERVING COPULA ALIGNMENT (MRPCA)
# Preserves BOTH 1D Empirical Marginals AND Full 2D Joint Feature Correlations
# =============================================================================

def align_empirical_distributions(
    df_syn: pd.DataFrame,
    df_real: pd.DataFrame,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    [QUARANTINED - UNREACHABLE FROM RELEASE ENTRYPOINT]
    Multivariate Rank-Preserving Copula Alignment (MRPCA).
    Strictly forbidden under differential privacy guarantees.
    """
    raise RuntimeError(
        "align_empirical_distributions is strictly forbidden on the release entrypoint "
        "and under differential privacy guarantees (privacy invariant violation)."
    )


def enforce_domain_constraints(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Final post-processing pass enforcing domain constraints and empirical QRM alignment.
    Deliberately unreachable in practice: align_empirical_distributions() below always
    raises (quarantined, forbidden under DP), so this whole non-strict fallback path is a
    hard safety gate rather than a usable code path. See test_enforce_public_domain_constraints_kinds.
    """
    from acfm_net.preparation.encoding import _detect_column_kind

    # Perform Multivariate Rank-Preserving Copula Alignment
    df_clean = align_empirical_distributions(df, original_df, target_col=target_col)

    # Secondary safety pass: enforce hard clinical bounds (Age >= 0, Glucose >= 20, etc.)
    for col in df_clean.columns:
        if col not in original_df.columns:
            continue

        orig_series = original_df[col].dropna()
        if not pd.api.types.is_numeric_dtype(orig_series):
            continue

        kind = _detect_column_kind(orig_series)
        lo   = float(orig_series.min())
        hi   = float(orig_series.max())

        syn_col = pd.to_numeric(df_clean[col], errors="coerce").fillna(lo)

        if kind == "binary":
            syn_col = syn_col.clip(0.0, 1.0).round().astype(int)

        elif kind == "binary_int":
            mid = (lo + hi) / 2.0
            syn_col = (syn_col >= mid).astype(int)
            syn_col = syn_col.map({0: int(lo), 1: int(hi)})

        elif kind == "count":
            floor = max(0.0, lo)
            syn_col = syn_col.clip(floor, hi).round().astype(int)

        elif kind == "integer":
            syn_col = syn_col.clip(lo, hi).round().astype(int)

        else:
            syn_col = syn_col.clip(lo, hi)

        df_clean[col] = syn_col

    return df_clean


BP_PAIRS = [("systolic", "diastolic"), ("ap_hi", "ap_lo")]


def enforce_relational_rules(df: pd.DataFrame, schema: Dict[str, Any], gap: float = 10.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Public clinical rule on decoded rows: systolic >= diastolic + gap. Post-processing only
    (reads synthetic rows and declared bounds), so it costs no privacy budget."""
    out = df.copy()
    cols = {c.lower(): c for c in out.columns}
    stats = {}
    for s_key, d_key in BP_PAIRS:
        s_col = next((c for k, c in cols.items() if s_key in k), None)
        d_col = next((c for k, c in cols.items() if d_key in k), None)
        if s_col is None or d_col is None:
            continue
        s = pd.to_numeric(out[s_col], errors="coerce")
        d = pd.to_numeric(out[d_col], errors="coerce")
        s_lo, s_hi = (schema.get(s_col, {}).get("bounds") or [s.min(), s.max()])
        d_lo, _ = (schema.get(d_col, {}).get("bounds") or [d.min(), d.max()])
        bad = s < d + gap
        before = int(bad.sum())
        d = d.where(~bad, np.maximum(s - gap, d_lo))
        s = s.where(s >= d + gap, np.minimum(d + gap, s_hi))
        out[s_col] = s.astype(out[s_col].dtype, errors="ignore")
        out[d_col] = d.astype(out[d_col].dtype, errors="ignore")
        stats[f"{s_col} >= {d_col} + {gap:g}"] = {"violations_before": before,
                                                  "violations_after": int((s < d + gap).sum())}
    return out, stats


def enforce_public_domain_constraints(df: pd.DataFrame, feature_specs: Dict[str, Any]) -> pd.DataFrame:
    """Apply only declared schema bounds; never read cohort extrema."""
    from acfm_net.preparation.encoding import _detect_column_kind

    result = df.copy()
    for col, spec in feature_specs.items():
        if col not in result.columns or spec.get("type") != "numeric":
            continue
        lo = float(spec.get("orig_min", -1.0))
        hi = float(spec.get("orig_max", 1.0))
        kind = spec.get("kind", "continuous")
        values = pd.to_numeric(result[col], errors="coerce").fillna(lo)
        if kind == "binary":
            result[col] = values.clip(0.0, 1.0).round().astype(int)
        elif kind in {"count", "integer", "binary_int"}:
            result[col] = values.clip(lo, hi).round().astype(int)
        else:
            result[col] = values.clip(lo, hi).astype(float)
    return result
