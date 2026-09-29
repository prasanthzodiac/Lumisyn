# Lumisyn

Lumisyn is a local research platform for privacy-aware synthetic clinical/tabular data generation.

## Layout

- `backend/` — FastAPI service, persistence, benchmarks, and integration tests.
- `frontend/` — React/Vite user interface.
- `framework/` — `acfm_net` generation and evaluation modules.
- `data/` — benchmark and sample datasets.
- `docs/` — architecture and research documents.
- `tools/` — dataset/report utilities.

## Local development

From the `lumisyn` directory:

```powershell
$env:PYTHONPATH = "$PWD\framework;$PWD"
python -m uvicorn backend.app_backend:app --host 127.0.0.1 --port 8000
```

On Linux/macOS:

```bash
export PYTHONPATH="$PWD/framework:$PWD"
python -m uvicorn backend.app_backend:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The UI runs on `http://localhost:3001` and proxies `/api` to the backend on port `8000`. Set `VITE_API_BASE_URL` only when the API is hosted on a different origin.

## Verification

```powershell
$env:PYTHONPATH = "$PWD\framework;$PWD"
python -m pytest backend/tests -q
cd frontend
npm run build
```

On Linux/macOS, `pytest` can also be run from the repository root, where `pytest.ini` sets the paths:

```bash
python -m pytest -q
cd lumisyn/frontend && npm run build
```

`test_schema_invariance.py` is skipped unless `lumisyn/data/kaggle_datasets/adult.csv` is present (datasets are not redistributed; see below).

## Reproducing the paper (`paper/main.tex`)

Run from the repository root (one level above `lumisyn/`). Each script writes its raw results to `runs/`.

| Paper element | Command | Output |
|---|---|---|
| Table I (sensitivity, random + screened) | `python lumisyn/backend/verify_icfm_sensitivity.py --worst-case-only` | `runs/icfm_worst_case_sensitivity.json` |
| Table I (hill-climbed), Table II (white-box audit), black-box canary audit | `python lumisyn/backend/audit_privacy.py` | `runs/privacy_audit.json` |
| Table III (I-CFM vs OT, 10 seeds) | `python lumisyn/backend/verify_icfm_sensitivity.py --utility-only` | `runs/icfm_sensitivity_verification.json` |
| Table IV (matched-ε baselines) | `python lumisyn/backend/baselines_eval.py --prepare`, then `.venv_baselines/Scripts/python lumisyn/backend/baselines_gen.py`, then `python lumisyn/backend/baselines_eval.py` | `runs/baselines/results.json` |
| Small-cohort results | `python lumisyn/backend/run_hard_stress_test.py --seeds 5` | `runs/hard_stress_test_results.json` |
| DAPC robustness | `python lumisyn/backend/dapc_sensitivity.py` | `runs/dapc_sensitivity.json` |
| Cardio audit | `python lumisyn/backend/audit_privacy.py --dataset cardio --part A` | `runs/privacy_audit.json` (key `stronger_adversary_cardio`) |
| Budget split / flow ablation (Table V) | `python lumisyn/backend/dapc_split_eval.py --dataset adult\|cardio\|pima [--seeds 10] [--eps E] [--configs ...]` | `runs/dapc_split/results.jsonl` |
| 10-seed equivalence test | `python lumisyn/backend/flow_ablation_stats.py` | `runs/dapc_split/flow_ablation_stats.json` |
| Target-aware AIM | `.venv_baselines/Scripts/python lumisyn/backend/baselines_gen.py aim_target`, then `baselines_eval.py` | `runs/baselines/results.json` |

- Pinned environments: `lumisyn/requirements-lock.txt` (main, Python 3.12) and `lumisyn/requirements-baselines-lock.txt` (smartnoise venv).
- Datasets are not redistributed: UCI Adult (doi:10.24432/C5XW20) and Kaggle cardiovascular-disease, Pima and Breast-Cancer-Wisconsin go in `lumisyn/data/kaggle_datasets/`.

- The baselines run in an isolated environment (`python -m venv .venv_baselines`, then `pip install smartnoise-synth packaging`), because smartnoise pins an old Opacus.
- The privacy proof and threat model are in `report/PRIVACY_PROOF.md`; the full sensitivity write-up is in `report/ICFM_SENSITIVITY_PROOF.md`.
- Not part of the paper artifact: `framework/acfm_net/optimization/`, `representation/contrastive.py` and `backend/full_audit_acfmv5.py` (legacy, unused by the pipeline), and HVAE mode (`latent_mode="vae"`, experimental).
