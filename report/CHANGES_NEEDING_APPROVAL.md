# Changes Needing Approval (Class C Items)

This document lists all **Class (c) Output-Altering Changes** proposed during the Stage 0-9 and Generation Path fix-and-test pass.

Pursuant to mandatory guardrails:
- Every Class (c) item is placed behind a configuration flag that **defaults to the OLD behavior** (`False`).
- No Class (c) item is enabled in default or release execution paths.
- Approval from clinical/scientific leadership is required prior to activation.

---

## 1. Optional Public Schema-Level `log1p` Transform (`enable_schema_log1p`)

- **Component**: `lumisyn/framework/acfm_net/preparation/encoding.py` (`DistributionPreservingEncoder`)
- **Config Flag**: `enable_schema_log1p: bool = False` (Default: `False` / Disabled)
- **Classification**: **Class (c) - Output-Altering**
- **Associated Unit Test**: `lumisyn/backend/tests/test_stage2_adpe_strict.py::test_stage2_log1p_flag_gated`

### Rationale and Motivation
Skewed continuous numerical features (such as `capital-gain`, `capital-loss`, `fnlwgt` in Census/Adult, and heavy-tailed clinical biomarkers such as triglycerides or insulin) have extreme positive dynamic ranges ($[0, 99999]$). Under standard linear min-max scaling to $[-1, 1]$, non-zero values are compressed into an infinitesimal band, hindering flow matching velocity estimation. Applying $\log(1 + x)$ mapping derived strictly from declared public schema specifications maps skewed distributions into an approximately normal distribution before flow matching.

### Metric Impact (Anticipated Movement if Enabled)
1. **Marginal Fidelity (1-way TVD)**: Expected reduction of 15% to 35% on extreme heavy-tailed columns (`capital-gain`, `capital-loss`).
2. **Standardized Sliced Wasserstein Distance (SWD)**: Expected 0.10 to 0.25 improvement on standardized continuous manifolds.
3. **Correlation Error (Frobenius norm)**: Preserved rank correlation with reduced outlier leverage.
4. **Safety & Privacy Invariant**: Governed strictly by declared public schema bounds (`lo`, `hi`), ensuring 0 epsilon expenditure and zero private data leakage.

### Current Implementation State
- Code implemented in `DistributionPreservingEncoder.fit_transform()`:
  ```python
  transform = schema.get("transform") if self.enable_schema_log1p else None
  if transform == "log1p":
      vals = np.log1p(np.maximum(0.0, vals_raw)).reshape(-1, 1)
  ```
- Defaults to `enable_schema_log1p=False` across all pipeline invocations. Verified by test suite.
