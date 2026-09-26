# Lumisyn UI and Pipeline Execution Audit

Audit date: 2026-08-22

## Release finding

The execution visualizer now follows the backend’s measured execution sequence and reports 11 stage records. Stage 9 no longer renders generated-looking placeholder values from a repeating ticker. It reveals a bounded preview of the actual synthetic release rows returned by `/api/generate`, once, in order, through the visible decode → repair → release path.

Live verification used the NHANES preset with six requested output rows and one flow epoch. The response contained six synthetic preview rows and these stage records:

| Execution order | Stage | Live status | UI evidence |
|---:|---|---|---|
| 1 | Stage 0 — Governance | PASSED | Governance/de-identification evidence and audit handle |
| 2 | Stage 1 — Intelligence | PASSED | Sampled rows, task, fingerprint dimension |
| 3 | Stage 2 — Preparation | PASSED | Encoded shape and post-imputation counts |
| 4 | Stage 3 — Domain Constraint Graph | PASSED | Node/edge counts and DP graph-statistics mode |
| 5 | Stage 5 — Privacy Controller | PASSED | Selected epsilon, DP-SGD budget and planned steps |
| 6 | Stage 4 — Representation | PASSED | Latent shape, DP-SGD components and steps |
| 7 | Stage 3b — Post-Latent Refinement | PASSED | Initial/refined edge counts and latent rows used |
| 8 | Stage 6 — Latent Generator | PASSED | Trained epochs and final flow diagnostics |
| 9 | Stage 8 — DP Optimization | PASSED | Accounted epsilon, SWD and correlation error |
| 10 | Stage 7 — Constrained Repair | PASSED | Rows repaired and repair statistics |
| 11 | Stage 9 — Synthetic Data Output | READY | Generated rows, columns, missing cells and privacy status |

The numeric order is intentionally not ascending because Stage 5 is a pre-flight privacy decision before Stage 4 training, while Stage 3b depends on the completed latent representation. The UI and API now use the same order.

## Page inventory

All 14 sidebar routes resolve to an existing component and are imported by `App.jsx`:

1. Dashboard
2. Upload Dataset & Governance
3. Dataset Intelligence
4. Data Visualization
5. Clinical Knowledge Graph
6. Pipeline Overview
7. HVAE Stage Details
8. Privacy Controller
9. Flow Matching
10. Original vs Synthetic Comparison
11. Statistical Validation
12. Clinical Consistency
13. Experiment History
14. Reports & Downloads

The following unreferenced legacy modules were removed because they were not reachable from navigation and contained stale hard-coded dashboard values:

- `frontend/src/components/screens/AIAssistantScreen.jsx`
- `frontend/src/components/screens/PowerBIReportsScreen.jsx`
- `frontend/src/components/PowerBIDashboardEmbed.jsx`

No missing routed component was found. Stage-specific execution evidence for all 11 computational records is intentionally consolidated in Pipeline Overview and the upload execution screen rather than duplicated across 11 separate navigation pages.

## Animation audit

- Pipeline cards: now distinguish `WAITING`, `QUEUED`, `RUNNING`, `PASSED`, `INITIALIZED`, `UNVERIFIED`, and `READY`; they no longer turn green merely because a local timer elapsed.
- Stage 3b: now has an explicit topology-refinement animation and is placed after Stage 4.
- Stage 6: now has a dedicated entry in both execution sequences and is marked passed only after flow training finishes.
- Stage 9: uses the actual backend synthetic preview, reveals each row once, and stops at `READY`; no `Math.random`, synthetic ticker values, or infinite data-loop animation remains in the Stage 9 visualizer.
- Reduced motion and animation-token cleanup remain a follow-up accessibility pass; the current output-row motion is limited to opacity and translation.

## Remaining research-quality caveat

All dataset-dependent pages now use measured session data or explicit empty states. The Dashboard sparkline decorations and Flow Matching page remain explanatory visualizations; they are not presented as empirical run metrics. Publication claims should continue to use the run-backed comparison, evaluation, stage-report, and privacy-composition responses.
