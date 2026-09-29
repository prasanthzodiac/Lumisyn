# Lumisyn Demo Runbook

## Start (two terminals, from the repository root)

```bash
python -m uvicorn backend.app_backend:app --host 127.0.0.1 --port 8000 --app-dir lumisyn
```

```bash
npm --prefix lumisyn/frontend run dev
```

Open http://localhost:3001.

Check before a demo: `python lumisyn/backend/tests/demo_smoke.py nhanes`. It needs the backend running and takes about 2 minutes. It exercises every endpoint and every stage PDF, and must end with `DEMO SMOKE TEST PASSED`.

## Demo path (about 5 minutes)

1. **Landing page → NHANES preset.**
   - The real backend run starts at once and takes about 80 s on the RTX 4060 Ti.
   - The stage panel cycles while the run is in progress. Cards read PENDING or RUNNING until the measured statuses arrive.
2. **Completion screen.**
   - The public schema source is shown.
   - The Stage Reports panel has one PDF per stage and an "All stages (PDF)" button.
3. **Dashboard → Data Visualization.** Real vs synthetic distributions. The target positive rate matches within a few points (about 22% vs 22–24%).
4. **Pipeline Overview.** Measured status of all 12 records (0, 1, 2, 3, 3b, 5, 4, 6, 8, 7, 7c, 9); Stage 4 is SKIPPED in identity mode. The stage PDFs are also available here.
5. **Privacy Controller.** Shows how DAPC picks ε* from public metadata, plus the PRV spending bar.
6. **Orig vs Synthetic / Statistical Validation.**
   - Per-feature means and standard deviations, correlation difference, TSTR.
   - Evaluation takes about 20 s.
   - C2ST is a fidelity score, not a privacy attack.
7. **Clinical Consistency.** Systolic ≥ diastolic + 10 is enforced as a public post-processing rule, so violations are 0.
8. **Reports & Downloads.** Evaluation PDF, synthetic CSV, Power BI bundle, privacy certificate and per-stage PDFs.

## What the demo configuration does

| Setting | Value |
|---|---|
| Presets | Declared public schemas (the generator's clip bounds) |
| Uploads | Bounds inferred from the file and labelled **UNVERIFIED** (declare a schema for a formal DP guarantee) |
| Stage 7c (classification targets) | DP label × feature marginals with a 0.9 budget share, jointly accounted with DP-SGD |
| Flow | Warm-started from the marginals (paper §2.6, §3.7) |
| Stage 7c bins | Scaled with the public row count: 8–32 bins |
| Clinical rule | `relational_rules=True`: systolic ≥ diastolic + 10 after decoding |
| Research scripts | All three flags stay at their defaults, so the paper results are unchanged |
