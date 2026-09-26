"""
Lumisyn FastAPI Backend
Wraps the ACFM-Net research pipeline (acfm_net) into REST APIs for the Lumisyn UI.
Provides dataset upload, real-time synthetic data generation, statistical comparison,
full evaluation suite (/api/evaluate), PDF export (/api/report/pdf), Power BI export,
and 21st.dev MCP integration.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "framework")))
import io
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

import hashlib
try:
    from .db_store import save_session_db, get_session_db, list_runs_db
except ImportError:  # Supports running this file directly from the backend folder.
    from db_store import save_session_db, get_session_db, list_runs_db

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from acfm_net.pipeline import ACFMNetV5Pipeline
from acfm_net.evaluation import run_full_evaluation, build_pdf_report

app = FastAPI(
    title="Lumisyn API",
    description="Synthetic clinical-data generation and evaluation API.",
    version="0.1.0"
)

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "LUMISYN_FRONTEND_ORIGINS",
        "http://localhost:3001,http://127.0.0.1:3001"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session database for storing uploaded/generated datasets & reports
SESSIONS_DB: Dict[str, Dict[str, Any]] = {}


# --- PRESET CLINICAL DATASETS GENERATORS ---
def get_preset_nhanes(num_samples: int = 1000) -> Tuple[pd.DataFrame, str]:
    np.random.seed(42)
    age = np.random.randint(18, 80, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples)
    poverty_ratio = np.random.uniform(0.5, 5.0, size=num_samples)
    bmi = np.random.normal(28.5, 6.2, size=num_samples).clip(14.0, 55.0)
    sys_bp = np.random.normal(124.0, 16.0, size=num_samples).clip(85.0, 200.0)
    dia_bp = np.minimum(sys_bp - 12.0, np.random.normal(76.0, 10.0, size=num_samples).clip(50.0, 120.0))
    glucose = np.random.normal(105.0, 35.0, size=num_samples).clip(60.0, 300.0)
    hba1c = np.random.normal(5.8, 1.2, size=num_samples).clip(4.0, 14.0)
    cholesterol = np.random.normal(195.0, 38.0, size=num_samples).clip(100.0, 380.0)
    hdl = np.random.normal(52.0, 14.0, size=num_samples).clip(20.0, 110.0)
    smoking = np.random.choice(["Never", "Former", "Current"], size=num_samples)
    diabetes = (np.random.rand(num_samples) < (1.0 / (1.0 + np.exp(-(-5.0 + 0.03*age + 0.08*bmi + 0.02*glucose + 0.5*hba1c))))).astype(int)

    df = pd.DataFrame({
        "age": age, "gender": gender, "poverty_ratio": np.round(poverty_ratio, 2),
        "bmi": np.round(bmi, 1), "systolic_bp": np.round(sys_bp, 1), "diastolic_bp": np.round(dia_bp, 1),
        "fasting_glucose": np.round(glucose, 1), "hba1c": np.round(hba1c, 1),
        "cholesterol": np.round(cholesterol, 1), "hdl": np.round(hdl, 1),
        "smoking_status": smoking, "diabetes": diabetes
    })
    return df, "diabetes"


def get_preset_mimic_iv(num_samples: int = 1000) -> Tuple[pd.DataFrame, str]:
    np.random.seed(43)
    age = np.random.randint(18, 90, size=num_samples)
    gender = np.random.choice(["M", "F"], size=num_samples)
    sofa_score = np.random.randint(0, 20, size=num_samples)
    heart_rate = np.random.normal(85, 15, size=num_samples).clip(40, 160)
    sys_bp = np.random.normal(120, 20, size=num_samples).clip(70, 210)
    dia_bp = np.minimum(sys_bp - 15, np.random.normal(75, 12, size=num_samples).clip(40, 120))
    creatinine = np.random.exponential(1.2, size=num_samples).clip(0.4, 10.0)
    lactate = np.random.exponential(1.8, size=num_samples).clip(0.5, 15.0)
    mortality = (np.random.rand(num_samples) < (1.0 / (1.0 + np.exp(-(-3.0 + 0.15*sofa_score + 0.2*lactate))))).astype(int)

    df = pd.DataFrame({
        "age": age, "gender": gender, "sofa_score": sofa_score, "heart_rate": np.round(heart_rate, 1),
        "systolic_bp": np.round(sys_bp, 1), "diastolic_bp": np.round(dia_bp, 1),
        "creatinine": np.round(creatinine, 2), "lactate": np.round(lactate, 2),
        "mortality": mortality
    })
    return df, "mortality"


PRESET_DATASETS = {
    "nhanes": ("NHANES Clinical Survey Dataset", get_preset_nhanes),
    "mimic_iv": ("MIMIC-IV Critical Care EHR", get_preset_mimic_iv)
}


# --- Pydantic Request Models ---
class GenerateRequest(BaseModel):
    session_id: str
    num_samples: int = 1000
    target_col: Optional[str] = None
    target_epsilon: float = 1.0
    target_delta: float = 1e-5
    epochs: int = 10
    protocol_id: str = "IRB-ACFMV5-2026"


# --- REST ENDPOINTS ---

@app.get("/api/health")
def healthcheck():
    return {
        "status": "online",
        "system": "Lumisyn Synthetic Healthcare Intelligence",
        "architecture_version": "research-prototype",
        "stages": "Governance, profiling, preparation, graph, representation, privacy, generation, repair, evaluation",
        "framework_connected": True,
        "ollama": "optional-local-reviewer"
    }


@app.get("/api/datasets/preset")
def list_preset_datasets():
    return [
        {"id": key, "name": name, "description": f"Pre-loaded benchmark dataset for {name}"}
        for key, (name, _) in PRESET_DATASETS.items()
    ]


@app.post("/api/upload")
async def upload_dataset(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Form(None)
):
    """
    Stage 0 & 1 API: Uploads dataset CSV or loads preset, initializes session,
    and returns initial data summary & governance check status.
    """
    session_id = str(uuid.uuid4())
    filename = "uploaded_dataset.csv"

    if preset_id:
        if preset_id not in PRESET_DATASETS:
            raise HTTPException(status_code=400, detail="Invalid preset dataset ID.")
        name, gen_fn = PRESET_DATASETS[preset_id]
        df_real, target_col = gen_fn()
        filename = f"{preset_id}.csv"
    elif file:
        filename = file.filename
        content = await file.read()
        try:
            if filename.endswith(".csv"):
                df_real = pd.read_csv(io.BytesIO(content))
            elif filename.endswith((".xls", ".xlsx")):
                df_real = pd.read_excel(io.BytesIO(content))
            elif filename.endswith(".parquet"):
                df_real = pd.read_parquet(io.BytesIO(content))
            else:
                df_real = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse uploaded dataset file: {str(e)}")
        target_col = None
    else:
        raise HTTPException(status_code=400, detail="Either file upload or preset_id must be provided.")

    if len(df_real) == 0:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    # Initialize ACFM-Net v5 pipeline instance for this session
    pipeline = ACFMNetV5Pipeline()

    SESSIONS_DB[session_id] = {
        "session_id": session_id,
        "filename": filename,
        "df_real": df_real,
        "df_syn": None,
        "target_col": target_col,
        "pipeline": pipeline,
        "created_at": time.time(),
        "evaluation_results": None,
    }

    columns_summary = []
    for col in df_real.columns:
        columns_summary.append({
            "name": col,
            "dtype": str(df_real[col].dtype),
            "missing_count": None if pipeline.strict_release_privacy else int(df_real[col].isna().sum()),
            "missing_pct": None if pipeline.strict_release_privacy else round(float(df_real[col].isna().mean()) * 100, 2),
            "unique_values": None if pipeline.strict_release_privacy else int(df_real[col].nunique())
        })

    num_cols = [col for col in df_real.columns if pd.api.types.is_numeric_dtype(df_real[col])]
    cat_cols = [col for col in df_real.columns if col not in num_cols]
    total_missing = None if pipeline.strict_release_privacy else int(df_real.isna().sum().sum())

    # Fast Stage 0 check for governance certificate metadata
    cert, _, _ = pipeline.cdgl.process_governance(df_real, run_deidentification=False)

    save_session_db(
        session_id=session_id,
        filename=filename,
        target_col=target_col or "",
        row_count=len(df_real),
        col_count=len(df_real.columns),
        status="READY",
        created_at=SESSIONS_DB[session_id]["created_at"]
    )

    df_sample = df_real.head(min(10, len(df_real)))
    return {
        "status": "SUCCESS",
        "session_id": session_id,
        "filename": filename,
        "total_rows": len(df_real),
        "total_columns": len(df_real.columns),
        "row_count": len(df_real),
        "column_count": len(df_real.columns),
        "missing_cells": total_missing,
        "suggested_target": target_col or (df_real.columns[-1] if len(df_real.columns) else None),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "re_id_risk": None,
        "re_id_risk_status": "withheld from upload response; released only through the DP composition report",
        "columns": columns_summary,
        "preview": [] if pipeline.strict_release_privacy else df_sample.to_dict(orient="records"),
        "preview_status": "withheld under strict release privacy" if pipeline.strict_release_privacy else "raw preview",
        "governance": {
            "protocol_id": cert.protocol_id,
            "dataset_hash": None if pipeline.strict_release_privacy else cert.dataset_hash,
            "dataset_hash_status": "withheld from release privacy response" if pipeline.strict_release_privacy else "audit hash",
            "re_id_risk": None,
            "re_id_risk_status": "withheld from upload response",
            "passed_irb": cert.passed_irb,
            "passed_deid": cert.passed_deid,
            "hipaa_compliant": cert.hipaa_compliant,
            "gdpr_compliant": cert.gdpr_compliant
        }
    }


@app.post("/api/dapc/analyze/{session_id}")
@app.get("/api/dapc/analyze/{session_id}")
def analyze_dataset_dapc(session_id: str):
    """
    Stage 5 DAPC Pre-flight API: Analyzes dataset privacy risk vector, calculates Soft-Max R_P,
    selects policy-bounded ε*, evaluates defensive safety (PROCEED/ABSTAIN/REJECT), and returns certified PDC.
    """
    if session_id not in SESSIONS_DB:
        # Check SQLite db fallback
        db_sess = get_session_db(session_id)
        if not db_sess:
            raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    df_real = session["df_real"]
    pipeline: ACFMNetV5Pipeline = session["pipeline"]

    # Run Stage 0 & 1 profiling if not already cached
    df_sample = df_real.sample(n=min(500, len(df_real)), random_state=42).reset_index(drop=True)
    sensitive_info = pipeline.sensitive_detector.detect(df_sample)
    dep_info = pipeline.dependency_engine.analyze(df_sample)
    cert, _, _ = pipeline.cdgl.process_governance(df_real, run_deidentification=False)

    S_vec = torch.tensor(sensitive_info["sensitivity_vector"], dtype=torch.float32)
    target_col = session.get("target_col") or df_real.columns[-1]

    dapc_cert = pipeline.budget_allocator.analyze_dataset_dapc(
        sensitivity_vector=S_vec,
        num_samples=len(df_real),
        num_features=len(df_real.columns),
        is_high_stakes=(target_col and any(k in str(target_col).lower() for k in ["mortality", "death", "cancer", "icu"])),
        dataset_name=session["filename"]
    )

    # Calculate dataset SHA-256 hash & certificate content SHA-256 hash
    if pipeline.strict_release_privacy:
        dapc_cert["dataset_hash"] = None
        dapc_cert["content_hash"] = None
        dapc_cert["hash_status"] = "withheld from release privacy response"
    else:
        ds_bytes = df_real.to_csv(index=False).encode('utf-8')
        dapc_cert["dataset_hash"] = hashlib.sha256(ds_bytes).hexdigest()
        dapc_cert["content_hash"] = hashlib.sha256(json.dumps(dapc_cert, sort_keys=True).encode('utf-8')).hexdigest()

    session["dapc_certificate"] = dapc_cert
    pipeline.target_epsilon = dapc_cert["selected_target_epsilon"]

    save_session_db(
        session_id=session_id,
        filename=session["filename"],
        target_col=target_col or "",
        row_count=len(df_real),
        col_count=len(df_real.columns),
        dapc_certificate=dapc_cert,
        status="READY",
        created_at=session["created_at"]
    )

    return dapc_cert


@app.get("/api/dapc/certificate/{session_id}")
def get_dapc_certificate(session_id: str):
    """
    Returns the Privacy Decision Certificate (PDC) for a session.
    """
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    dapc_cert = session.get("dapc_certificate")
    if not dapc_cert:
        return analyze_dataset_dapc(session_id)
    return dapc_cert


@app.get("/api/graph/{session_id}")
def get_clinical_knowledge_graph(session_id: str):
    """
    Stage 3 & 3b API: Returns multi-relational clinical graph nodes and adjacency edge weights.
    """
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    df_real = session["df_real"]
    pipeline: ACFMNetV5Pipeline = session["pipeline"]

    df_sample = df_real.sample(n=min(300, len(df_real)), random_state=42).reset_index(drop=True)
    sensitive_info = pipeline.sensitive_detector.detect(df_sample)
    dep_info = pipeline.dependency_engine.analyze(df_sample)

    cols = list(df_real.columns[:32])
    d = len(cols)
    dep_mat = dep_info.get("dependency_matrix", np.eye(d))[:d, :d]

    nodes = [
        {
            "id": col,
            "name": col,
            "sensitivity": round(float(sensitive_info["sensitivity_vector"][i] if i < len(sensitive_info["sensitivity_vector"]) else 0.5), 3),
            "is_sensitive": col in sensitive_info.get("sensitive_columns", []),
            "missing_pct": round(float(df_real[col].isna().mean()) * 100, 1)
        }
        for i, col in enumerate(cols)
    ]

    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            val = float(abs(dep_mat[i, j]))
            if val > 0.15:
                edges.append({
                    "source": cols[i],
                    "target": cols[j],
                    "weight": round(val, 3),
                    "type": "mutual_info" if val > 0.4 else "pearson"
                })

    return {
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges)
    }


@app.get("/api/runs")
def list_experiment_runs():
    """
    Returns persistent list of experiment runs stored in SQLite database.
    """
    return list_runs_db()


@app.post("/api/generate")
def generate_synthetic_data(req: GenerateRequest):
    """
    Stages 0 to 8 Execution API: Triggers full ACFM-Net v5 generation workflow.
    Enforces server-side DAPC safety abstention check.
    """
    if req.session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found. Upload dataset first.")

    session = SESSIONS_DB[req.session_id]

    # Server-Side Abstention & Safety Gate Enforcement
    dapc_cert = session.get("dapc_certificate")
    if dapc_cert and dapc_cert.get("decision") in ["ABSTAIN", "REJECT"]:
        raise HTTPException(
            status_code=403,
            detail=f"Training Blocked: DAPC Security Abstention triggered ({dapc_cert.get('decision')}). Rationale: {', '.join(dapc_cert.get('decision_reasons', []))}"
        )

    df_real = session["df_real"]
    target_col = req.target_col or session["target_col"] or df_real.columns[-1]
    session["target_col"] = target_col

    pipeline: ACFMNetV5Pipeline = session["pipeline"]
    pipeline.target_epsilon = req.target_epsilon
    pipeline.target_delta = req.target_delta

    t0 = time.time()
    try:
        df_syn, report = pipeline.fit_generate(
            df_raw=df_real,
            num_samples=req.num_samples,
            target_col=target_col,
            epochs=req.epochs,
            protocol_id=req.protocol_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthetic data generation failed: {str(e)}")

    t_elapsed = time.time() - t0
    session["df_syn"] = df_syn
    session["pipeline_report"] = report

    eps_used = float(pipeline.prv_accountant.accumulated_privacy_cost)
    privacy_training = report.get("privacy_training", {})
    privacy_training_verified = bool(privacy_training.get("verified", False))
    privacy_status = report.get("privacy_status", "UNVERIFIED")
    release_privacy_verified = bool(report.get("release_privacy_verified", False))
    # Synthetic release rows are safe to show in the UI and make the Stage 9
    # animation evidence-based. Raw source rows remain withheld by the strict
    # public-schema release contract.
    synthetic_preview = json.loads(
        df_syn.head(8).to_json(orient="records", date_format="iso")
    )

    return {
        "status": "COMPLETED",
        "session_id": req.session_id,
        "generated_samples": len(df_syn),
        "generation_time_sec": round(t_elapsed, 2),
        "stages_passed": len([
            s for s in report.get("stage_reports", [])
            if s.get("status") in {"PASSED", "READY"}
        ]),
        "stage_report": report.get("stage_reports", []),
        "pipeline_status": report.get("pipeline_status"),
        "dapc_certificate": report.get("dapc_certificate"),
        "dapc_decision": report.get("dapc_decision", "PROCEED"),
        "synthetic_preview": synthetic_preview,
        "report_summary": {
            "privacy_guaranteed": release_privacy_verified,
            "privacy_training_verified": privacy_training_verified,
            "privacy_status": privacy_status,
            "privacy_note": "Verified under the declared public-schema assumption: preprocessing uses public bounds, empirical alignment is disabled, graph statistics use bounded Laplace noise, and DP-SGD composition is checked before release.",
            "privacy_composition": report.get("privacy_composition"),
            "prv_eps_used": round(eps_used, 4),
            "dp_steps": privacy_training.get("steps"),
            "dp_sample_rate": privacy_training.get("sample_rate"),
            "dp_noise_multiplier": privacy_training.get("noise_multiplier"),
            "synthetic_missing_cells": int(df_syn.isna().sum().sum())
        }
    }


@app.get("/api/stages/{session_id}")
def get_stage_report(session_id: str):
    """Return measured stage evidence for the latest generation run."""
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    session = SESSIONS_DB[session_id]
    report = session.get("pipeline_report")
    if not report:
        return {
            "session_id": session_id,
            "pipeline_status": "READY",
            "stage_reports": [],
            "message": "No generation run has completed for this session.",
        }
    return {
        "session_id": session_id,
        "pipeline_status": report.get("pipeline_status"),
        "generation_runtime_seconds": report.get("generation_runtime_seconds"),
        "stage_reports": report.get("stage_reports", []),
        "privacy_training": report.get("privacy_training", {}),
        "privacy_status": report.get("privacy_status", "UNVERIFIED"),
        "release_privacy_verified": report.get("release_privacy_verified", False),
        "privacy_contract": report.get("privacy_contract", {}),
        "privacy_composition": report.get("privacy_composition", {}),
    }


@app.get("/api/evaluate/{session_id}")
def evaluate_synthetic_dataset(session_id: str):
    """
    Stage 9 / Final Evaluation Module API.
    Runs full evaluation across 5 dimensions: Statistical, Utility (TSTR/TRTS), Privacy (MIA/Attr-IA),
    Clinical Constraints, and Fairness.
    """
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    if session["df_syn"] is None:
        raise HTTPException(status_code=400, detail="Synthetic data not yet generated for this session.")

    df_real = session["df_real"]
    df_syn  = session["df_syn"]
    target_col = session["target_col"]
    pipeline: ACFMNetV5Pipeline = session["pipeline"]

    eps_used = float(pipeline.prv_accountant.accumulated_privacy_cost)
    target_eps = float(pipeline.target_epsilon)

    eval_results = run_full_evaluation(
        df_real=df_real,
        df_syn=df_syn,
        target_col=target_col,
        eps_used=eps_used,
        target_epsilon=target_eps,
        dataset_name=session["filename"],
        pipeline_report=session.get("pipeline_report"),
        privacy_verified=bool((session.get("pipeline_report") or {}).get("release_privacy_verified", False)),
        privacy_scope=(session.get("pipeline_report") or {}).get("privacy_status", "UNVERIFIED"),
    )

    session["evaluation_results"] = eval_results
    return eval_results



@app.get("/api/report/pdf/{session_id}")
def get_pdf_report(session_id: str):
    """
    Generates and downloads the official ACFM-Net v5 PDF Evaluation Report.
    """
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    if session["df_syn"] is None:
        raise HTTPException(status_code=400, detail="Synthetic data not yet generated.")

    eval_results = session.get("evaluation_results")
    if eval_results is None:
        # Run evaluation lazily
        df_real = session["df_real"]
        df_syn  = session["df_syn"]
        target_col = session["target_col"]
        pipeline: ACFMNetV5Pipeline = session["pipeline"]
        eval_results = run_full_evaluation(
            df_real=df_real,
            df_syn=df_syn,
            target_col=target_col,
            eps_used=float(pipeline.prv_accountant.accumulated_privacy_cost),
            target_epsilon=float(pipeline.target_epsilon),
            dataset_name=session["filename"],
            pipeline_report=session.get("pipeline_report"),
            privacy_verified=bool((session.get("pipeline_report") or {}).get("release_privacy_verified", False)),
            privacy_scope=(session.get("pipeline_report") or {}).get("privacy_status", "UNVERIFIED"),
        )
        session["evaluation_results"] = eval_results

    pdf_bytes = build_pdf_report(
        eval_results=eval_results,
        dataset_name=session["filename"],
        eps_used=float(eval_results["metadata"]["eps_used"]),
        target_epsilon=float(eval_results["metadata"]["target_epsilon"]),
    )

    response = StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=ACFM_Net_v5_Report_{session_id[:8]}.pdf"
    return response


@app.get("/api/compare/{session_id}")
def get_statistical_comparison(session_id: str):
    """
    Returns side-by-side comparative statistical metrics (D_real vs D_syn).
    """
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    if session["df_syn"] is None:
        raise HTTPException(status_code=400, detail="Synthetic data not yet generated for this session.")

    df_real = session["df_real"]
    df_syn = session["df_syn"]
    target_col = session["target_col"]

    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]

    # Compute Feature-wise summary stats & visualizer distributions (Histograms, KDE, Boxplots)
    feature_metrics = []
    histograms = {}
    kde_curves = {}
    box_plots = {}
    quantile_preservation = {}

    for col in num_cols:
        r = df_real[col].dropna().values.astype(float)
        s = df_syn[col].dropna().values.astype(float)

        r_mean, s_mean = float(np.mean(r)), float(np.mean(s))
        r_std,  s_std  = float(np.std(r)),  float(np.std(s))
        r_min,  r_max  = float(np.min(r)),  float(np.max(r))
        s_min,  s_max  = float(np.min(s)),  float(np.max(s))

        feature_metrics.append({
            "feature": col,
            "real_mean": round(r_mean, 3),
            "syn_mean": round(s_mean, 3),
            "mean_abs_error": round(abs(r_mean - s_mean), 3),
            "real_std": round(r_std, 3),
            "syn_std": round(s_std, 3),
            "std_abs_error": round(abs(r_std - s_std), 3),
            "real_min": round(r_min, 2),
            "real_max": round(r_max, 2),
            "syn_min": round(s_min, 2),
            "syn_max": round(s_max, 2),
        })

        # 1. Binned Histograms (15 bins)
        lo, hi = min(r_min, s_min), max(r_max, s_max)
        bins = np.linspace(lo, hi + 1e-6, 16)
        r_counts, _ = np.histogram(r, bins=bins)
        s_counts, _ = np.histogram(s, bins=bins)
        bin_labels = [round(float((bins[i] + bins[i+1])/2), 2) for i in range(len(bins)-1)]

        histograms[col] = {
            "bin_centers": bin_labels,
            "real_counts": r_counts.tolist(),
            "syn_counts": s_counts.tolist(),
        }

        # 2. KDE Points (30 points)
        grid = np.linspace(lo, hi, 30)
        from scipy import stats
        try:
            r_kde = stats.gaussian_kde(r)(grid)
            s_kde = stats.gaussian_kde(s)(grid)
            kde_curves[col] = {
                "grid": [round(float(x), 2) for x in grid],
                "real_density": [round(float(y), 4) for y in r_kde],
                "syn_density": [round(float(y), 4) for y in s_kde],
            }
        except Exception:
            kde_curves[col] = {}

        # 3. Box Plots
        box_plots[col] = {
            "real": {
                "min": round(r_min, 2), "q25": round(float(np.quantile(r, 0.25)), 2),
                "median": round(float(np.median(r)), 2), "q75": round(float(np.quantile(r, 0.75)), 2),
                "max": round(r_max, 2)
            },
            "synthetic": {
                "min": round(s_min, 2), "q25": round(float(np.quantile(s, 0.25)), 2),
                "median": round(float(np.median(s)), 2), "q75": round(float(np.quantile(s, 0.75)), 2),
                "max": round(s_max, 2)
            }
        }

        quantile_preservation[col] = {
            "real_extremes": [r_min, r_max],
            "syn_extremes": [s_min, s_max],
            "tail_preservation_ratio": round(min(1.0, (s_max - s_min)/max(1e-5, (r_max - r_min))), 4)
        }

    # Correlation Matrix comparison
    r_corr_df = df_real[num_cols].corr().fillna(0).round(3)
    s_corr_df = df_syn[num_cols].corr().fillna(0).round(3)
    diff_corr_df = (r_corr_df - s_corr_df).round(3)

    r_corr = r_corr_df.to_dict()
    s_corr = s_corr_df.to_dict()
    diff_corr = diff_corr_df.to_dict()

    frobenius_diff = float(np.linalg.norm(r_corr_df.values - s_corr_df.values, 'fro'))

    # Outcome class distribution comparison
    class_ratio_summary = {}
    if target_col and target_col in df_real.columns and target_col in df_syn.columns:
        r_target = df_real[target_col].dropna()
        s_target = df_syn[target_col].dropna()
        class_ratio_summary = {
            "target_col": target_col,
            "real_positive_ratio": round(float((r_target == r_target.max()).mean()), 4),
            "syn_positive_ratio": round(float((s_target == s_target.max()).mean()), 4),
            "ratio_matched": abs(float((r_target == r_target.max()).mean() - (s_target == s_target.max()).mean())) < 0.02
        }

    # Stage 7 Clinical Rule Repair check
    sys_col = [c for c in df_syn.columns if "systolic" in c.lower()]
    dia_col = [c for c in df_syn.columns if "diastolic" in c.lower()]
    if sys_col and dia_col:
        violations = int((df_syn[sys_col[0]] < df_syn[dia_col[0]]).sum())
    else:
        violations = 0

    eps_used = float(session["pipeline"].prv_accountant.accumulated_privacy_cost)
    target_eps = float(session["pipeline"].target_epsilon)
    pipeline_report = session.get("pipeline_report") or {}
    privacy_training = pipeline_report.get("privacy_training", {})
    privacy_status = pipeline_report.get("privacy_status", "UNVERIFIED")

    return {
        "session_id": session_id,
        "dataset_name": session["filename"],
        "target_col": target_col,
        "numeric_features": num_cols,
        "feature_summary": feature_metrics,
        "class_ratio_summary": class_ratio_summary,
        "real_correlation": r_corr,
        "syn_correlation": s_corr,
        "diff_correlation": diff_corr,
        "correlation_frobenius_error": round(frobenius_diff, 4),
        "histograms": histograms,
        "kde_curves": kde_curves,
        "box_plots": box_plots,
        "quantile_preservation": quantile_preservation,
        "clinical_rule_violations": violations,
        "clinical_rule_pass_rate": 100.0 if violations == 0 else round((1.0 - violations/len(df_syn)) * 100, 2),
        "privacy": {
            "target_epsilon": target_eps,
            "eps_used": round(eps_used, 4),
            "status": privacy_status,
            "verified": bool(pipeline_report.get("release_privacy_verified", False)),
            "privacy_training_verified": bool(privacy_training.get("verified", False)),
            "dp_steps": privacy_training.get("steps"),
            "dp_noise_multiplier": privacy_training.get("noise_multiplier"),
            "privacy_composition": pipeline_report.get("privacy_composition"),
            "note": "Verified under the declared public-schema assumption; the exact epsilon/delta composition is included in privacy_composition."
        }
    }



@app.get("/api/download/{session_id}")
def download_synthetic_csv(session_id: str):
    """
    Downloads the generated synthetic dataset as a CSV file.
    """
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    df_syn = session["df_syn"]
    if df_syn is None:
        raise HTTPException(status_code=400, detail="Synthetic data not generated yet.")

    stream = io.StringIO()
    df_syn.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=synthetic_{session['filename']}"
    return response


@app.get("/api/powerbi/export/{session_id}")
def export_powerbi_bundle(session_id: str):
    """
    Returns Power BI Ready DAX measures, data schema, and download endpoints.
    """
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    session = SESSIONS_DB[session_id]
    if session["df_syn"] is None:
        raise HTTPException(status_code=400, detail="Synthetic data not generated yet.")

    dax_measures = [
        "Total Real Patient Count = COUNTROWS('RealDataset')",
        "Total Synthetic Patient Count = COUNTROWS('SyntheticDataset')",
        "Mean Absolute Feature Error % = AVERAGE('MetricsTable'[MeanAbsError])",
        "Clinical Rule Pass Rate % = 100.0 - (SUM('SyntheticDataset'[ViolationFlag]) / COUNTROWS('SyntheticDataset') * 100.0)",
        "Differential Privacy Margin = 1.0 - MAX('PrivacyMetrics'[EpsUsed])"
    ]

    pbix_schema = {
        "model_name": "ACFM_Net_v5_Healthcare_Synthetic_Data_Model",
        "version": "5.1",
        "tables": [
            {"name": "RealDataset", "rows": len(session["df_real"]), "columns": list(session["df_real"].columns)},
            {"name": "SyntheticDataset", "rows": len(session["df_syn"]), "columns": list(session["df_syn"].columns)},
            {"name": "MetricsTable", "measures": len(dax_measures)}
        ],
        "relationships": [
            {"fromTable": "RealDataset", "fromColumn": session["target_col"], "toTable": "SyntheticDataset", "toColumn": session["target_col"]}
        ]
    }

    return {
        "session_id": session_id,
        "pbix_schema": pbix_schema,
        "dax_measures": dax_measures,
        "powerbi_import_instructions": [
            "1. Open Power BI Desktop.",
            "2. Click Get Data -> Web/CSV.",
            "3. Load Real Dataset and Synthetic Dataset CSV tables.",
            "4. Copy and paste the DAX measures into your Power BI report model.",
            "5. Drag and drop the side-by-side distribution visual charts."
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
