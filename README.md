# Lumisyn

Lumisyn is a local research platform for privacy-aware synthetic clinical/tabular data generation: a differentially private flow-matching pipeline (`acfm_net`), a FastAPI backend, and a React dashboard.

## Repository layout

- `lumisyn/` — the application: `framework/` (`acfm_net` pipeline), `backend/` (API, benchmarks, tests), `frontend/` (React/Vite UI), `docs/`, `tools/`. See [`lumisyn/README.md`](lumisyn/README.md) for setup, running and paper reproduction.
- `paper/` — the paper source (`main.tex`, `refs.bib`).
- `report/` — verification and benchmark reports.
- `runs/` — raw logs and JSON results cited by the paper and reports.
- `scripts/` — experiment and diagnostic runners.

## Quick check

From the repository root (`pytest.ini` sets the import paths):

```bash
pip install -r lumisyn/requirements.txt pytest httpx
python -m pytest -q
```

The pinned environment is `lumisyn/requirements-lock.txt` (Python 3.12).
