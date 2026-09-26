import sqlite3
import json
import os
import time
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "artifacts", "sessions.db")


def _ensure_artifact_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    _ensure_artifact_dir()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            filename TEXT,
            target_col TEXT,
            created_at REAL,
            row_count INTEGER,
            col_count INTEGER,
            dapc_certificate TEXT,
            status TEXT,
            has_synthetic INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_session_db(session_id: str, filename: str, target_col: str, row_count: int, col_count: int, dapc_certificate: Optional[Dict[str, Any]] = None, status: str = "READY", has_synthetic: bool = False, created_at: Optional[float] = None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cert_json = json.dumps(dapc_certificate) if dapc_certificate else None
    now = created_at or time.time()
    cursor.execute("""
        INSERT OR REPLACE INTO sessions 
        (session_id, filename, target_col, created_at, row_count, col_count, dapc_certificate, status, has_synthetic)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, filename, target_col, now, row_count, col_count, cert_json, status, 1 if has_synthetic else 0))
    conn.commit()
    conn.close()

def get_session_db(session_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, filename, target_col, created_at, row_count, col_count, dapc_certificate, status, has_synthetic FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "session_id": row[0],
        "filename": row[1],
        "target_col": row[2],
        "created_at": row[3],
        "row_count": row[4],
        "col_count": row[5],
        "dapc_certificate": json.loads(row[6]) if row[6] else None,
        "status": row[7],
        "has_synthetic": bool(row[8])
    }

def list_runs_db() -> List[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, filename, row_count, col_count, created_at, status, dapc_certificate FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    runs = []
    for r in rows:
        dapc = json.loads(r[6]) if r[6] else {}
        runs.append({
            "run_id": r[0],
            "dataset_name": r[1],
            "row_count": r[2],
            "col_count": r[3],
            "created_at": r[4],
            "status": r[5],
            "target_epsilon": dapc.get("selected_target_epsilon", 0.766),
            "risk_score_R_P": dapc.get("risk_score_R_P", 0.392),
            "dapc_decision": dapc.get("decision", "PROCEED")
        })
    return runs

init_db()
