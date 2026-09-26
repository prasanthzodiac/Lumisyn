"""
Stage 7c: DP class-conditional marginals (label x feature), Gaussian mechanism.

Measurement (the MST/AIM recipe: noisy low-order marginals on public domains; McKenna et al.
2021, 2022). One (classes x bins) count table per non-target column, on PUBLIC bins/categories.
Under add/remove adjacency one record changes exactly one cell of each of the M tables by 1,
so the concatenated count vector has L2 sensitivity sqrt(M). Adding N(0, (z sqrt(M))^2) per
cell is a Gaussian mechanism with noise multiplier z; the pipeline composes it with DP-SGD in
the PRV accountant as the history entry (z, q=1, steps=1).

Recalibration (post-processing; Sklar's theorem): class sizes come from the noisy tables;
within each class every column is quantile-mapped onto its noisy conditional marginal while
keeping the flow's within-class ranks, i.e. the flow supplies the dependence structure
(copula) and the DP tables supply the label-feature marginals. copula="independent" discards
the flow's ranks (ablation: marginals only).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _heavy_tailed(lo: float, hi: float) -> bool:
    return lo >= 0 and hi / max(lo, 1.0) > 1000


def _edges(spec: Dict[str, Any], n_bins: int) -> np.ndarray:
    lo, hi = (float(b) for b in spec["bounds"])
    if _heavy_tailed(lo, hi):
        return np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), n_bins + 1))
    return np.linspace(lo, hi, n_bins + 1)


def _is_discrete_numeric(spec: Dict[str, Any]) -> bool:
    return spec.get("kind") in ("binary", "binary_int", "count", "integer")


class ConditionalMarginals:
    def __init__(self, schema: Dict[str, Any], target: str, columns: List[str], n_bins: int = 32):
        self.schema = schema.get("columns", schema)
        self.target = target
        self.columns = [c for c in columns if c != target and c in self.schema]
        self.n_bins = n_bins
        self.tables: Optional[Dict[str, np.ndarray]] = None

    # ---------------------------------------------------------------- public domain
    def _small_int_domain(self, spec: Dict[str, Any]) -> bool:
        lo, hi = spec["bounds"]
        return _is_discrete_numeric(spec) and hi - lo + 1 <= self.n_bins

    def _levels(self, col: str) -> int:
        spec = self.schema[col]
        if spec.get("type") == "categorical":
            return len(spec["categories"])
        lo, hi = spec["bounds"]
        return int(hi - lo + 1) if self._small_int_domain(spec) else self.n_bins

    def _codes(self, series: pd.Series, col: str) -> np.ndarray:
        spec = self.schema[col]
        if spec.get("type") == "categorical":
            cats = [str(c) for c in spec["categories"]]
            fallback = cats.index("MISSING") if "MISSING" in cats else 0
            idx = {c: i for i, c in enumerate(cats)}
            return series.fillna("MISSING").astype(str).map(lambda v: idx.get(v, fallback)).to_numpy(dtype=int)
        lo, hi = (float(b) for b in spec["bounds"])
        x = pd.to_numeric(series, errors="coerce").fillna(lo).clip(lo, hi).to_numpy()
        if self._small_int_domain(spec):
            return np.round(x - lo).astype(int)
        return np.clip(np.searchsorted(_edges(spec, self.n_bins), x, side="right") - 1, 0, self.n_bins - 1)

    @property
    def num_tables(self) -> int:
        return len(self.columns)

    @property
    def max_cells(self) -> int:
        k = self._levels(self.target)
        return max(k * self._levels(c) for c in self.columns)

    # ---------------------------------------------------------------- mechanism
    def release(self, df: pd.DataFrame, noise_multiplier: float) -> Dict[str, Any]:
        rng = np.random.default_rng()  # unseeded: the noise must be unpredictable
        std = float(noise_multiplier) * np.sqrt(self.num_tables)
        y = self._codes(df[self.target], self.target)
        k = self._levels(self.target)
        self.tables = {}
        for col in self.columns:
            t = np.zeros((k, self._levels(col)))
            np.add.at(t, (y, self._codes(df[col], col)), 1.0)
            self.tables[col] = t + rng.normal(0.0, std, size=t.shape)
        return {"mechanism": "Gaussian", "adjacency": "add_remove", "num_tables": self.num_tables,
                "l2_sensitivity": float(np.sqrt(self.num_tables)), "noise_multiplier": float(noise_multiplier),
                "noise_std_per_cell": std, "max_cells_per_table": self.max_cells}

    def _class_probs(self) -> np.ndarray:
        sums = np.mean([np.clip(t, 0, None).sum(axis=1) for t in self.tables.values()], axis=0)
        return sums / sums.sum() if sums.sum() > 0 else np.full(len(sums), 1.0 / len(sums))

    # ---------------------------------------------------------------- post-processing
    @staticmethod
    def _quota(p: np.ndarray, n: int) -> np.ndarray:
        raw = p * n
        q = np.floor(raw).astype(int)
        q[np.argsort(raw - q)[::-1][: n - q.sum()]] += 1
        return q

    @staticmethod
    def _assign(current: np.ndarray, quota: np.ndarray, rng) -> np.ndarray:
        """Keep each record's current level while its quota lasts; fill the rest randomly."""
        out = np.full(len(current), -1)
        left = quota.copy()
        for i in rng.permutation(len(current)):
            c = current[i]
            if 0 <= c < len(left) and left[c] > 0:
                out[i] = c
                left[c] -= 1
        out[out < 0] = rng.permutation(np.repeat(np.arange(len(left)), left))
        return out

    def recalibrate(self, df_syn: pd.DataFrame, copula: str = "flow", seed: Optional[int] = None) -> pd.DataFrame:
        assert self.tables is not None, "release() first"
        rng = np.random.default_rng(seed)
        out = df_syn.copy()
        y = self._assign(self._codes(out[self.target], self.target), self._quota(self._class_probs(), len(out)), rng)
        target_vals = self._decode(self.target, y, out[self.target], rng, copula)
        for col in self.columns:
            t = np.clip(self.tables[col], 0, None)
            cur = self._codes(out[col], col)
            vals = np.empty(len(out), dtype=object)
            for c in range(t.shape[0]):
                idx = np.where(y == c)[0]
                if len(idx) == 0:
                    continue
                row = t[c] if t[c].sum() > 0 else t.sum(axis=0) + 1e-9
                start = cur[idx] if copula == "flow" else rng.integers(0, t.shape[1], len(idx))
                levels = self._assign(start, self._quota(row / row.sum(), len(idx)), rng)
                vals[idx] = self._decode(col, levels, out[col].iloc[idx], rng, copula)
            out[col] = vals if self.schema[col].get("type") == "categorical" else pd.to_numeric(vals)
        out[self.target] = target_vals
        return out

    def _decode(self, col, levels, original, rng, copula) -> np.ndarray:
        spec = self.schema[col]
        if spec.get("type") == "categorical":
            cats = list(spec["categories"])
            return np.array([cats[i] for i in levels], dtype=object)
        lo, hi = (float(b) for b in spec["bounds"])
        if self._small_int_domain(spec):
            return (levels + lo).astype(int)
        e = _edges(spec, self.n_bins)
        # Place each value inside its bin by its within-bin rank in the flow output (copula), else uniform.
        x = pd.to_numeric(original, errors="coerce").to_numpy(dtype=float)
        u = rng.uniform(size=len(levels))
        if copula == "flow":
            for b in np.unique(levels):
                m = levels == b
                r = np.argsort(np.argsort(x[m] + rng.normal(0, 1e-9, m.sum())))
                u[m] = (r + 0.5) / m.sum()
        vals = e[levels] + u * (e[levels + 1] - e[levels])
        if _is_discrete_numeric(spec) or _heavy_tailed(lo, hi):
            vals = np.round(vals)
        return np.clip(vals, lo, hi)
