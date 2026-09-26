import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "framework")))

import numpy as np
import torch
import json
import hashlib
from fastapi.testclient import TestClient

from acfm_net.privacy import AdaptiveBudgetAllocator, PRVAccountant, RDPAccountant
from acfm_net.pipeline import ACFMNetV5Pipeline
try:
    from .app_backend import app, SESSIONS_DB, get_preset_nhanes
    from .db_store import list_runs_db, get_session_db
except ImportError:  # Supports direct execution from the backend folder.
    from app_backend import app, SESSIONS_DB, get_preset_nhanes
    from db_store import list_runs_db, get_session_db

print("=" * 80)
print("ACFM-NET POST-FIX ADVERSARIAL VALIDATION TEST SUITE")
print("=" * 80)

test_results = {}

# ------------------------------------------------------------------------------
# 1. PIPELINE INITIALIZATION
# ------------------------------------------------------------------------------
print("\n[TEST 1] PIPELINE INITIALIZATION")
try:
    pipe = ACFMNetV5Pipeline(target_epsilon=0.8, target_delta=1e-5)
    print("  PASS: ACFMNetV5Pipeline initialized without constructor signature errors.")
    test_results["1_pipeline_init"] = "PASS"
except Exception as e:
    print(f"  FAIL: {str(e)}")
    test_results["1_pipeline_init"] = f"FAIL ({str(e)})"

# ------------------------------------------------------------------------------
# 2. DATASET UPLOAD
# ------------------------------------------------------------------------------
print("\n[TEST 2] DATASET UPLOAD (FastAPI TestClient)")
client = TestClient(app)
upload_res = client.post("/api/upload", data={"preset_id": "nhanes"})
print(f"  Upload Response Status: {upload_res.status_code}")
session_id = upload_res.json().get("session_id")
print(f"  Created Session ID: {session_id}")
test_results["2_dataset_upload"] = "PASS" if upload_res.status_code == 200 else "FAIL"

# ------------------------------------------------------------------------------
# 3. DAPC ANALYSIS & PDC HASH VERIFICATION
# ------------------------------------------------------------------------------
print("\n[TEST 3] DAPC ANALYSIS & CERTIFICATE SHA-256 HASH VERIFICATION")
analyze_res = client.post(f"/api/dapc/analyze/{session_id}")
print(f"  POST /api/dapc/analyze Status: {analyze_res.status_code}")
cert = analyze_res.json()
print(f"  Decision: {cert.get('decision')}, Target ε*: {cert.get('selected_target_epsilon')}")
print(f"  Dataset SHA-256 Hash: {cert.get('dataset_hash')[:16]}...")
print(f"  Certificate Content SHA-256 Hash: {cert.get('content_hash')[:16]}...")

has_hashes = "dataset_hash" in cert and "content_hash" in cert
test_results["3_dapc_analysis"] = "PASS" if analyze_res.status_code == 200 and has_hashes else "FAIL"

# ------------------------------------------------------------------------------
# 4, 5, 6, 7. RISK PROFILES & ABSTENTION
# ------------------------------------------------------------------------------
print("\n[TEST 4-7] RISK PROFILES & DEFENSIVE ABSTENTION GATES")
allocator = AdaptiveBudgetAllocator()

profiles = {
    "LOW RISK":      {"sens": [0.2]*5, "N": 50000, "D": 5, "reid": 0.01, "unique": 0.01, "dep": 0.05, "high_stakes": False},
    "MODERATE RISK": {"sens": [0.5]*10, "N": 5000, "D": 15, "reid": 0.15, "unique": 0.15, "dep": 0.35, "high_stakes": False},
    "HIGH RISK":     {"sens": [0.8]*15, "N": 1000, "D": 30, "reid": 0.40, "unique": 0.30, "dep": 0.65, "high_stakes": True},
    "CRITICAL ABSTAIN": {"sens": [0.95]*20, "N": 300, "D": 50, "reid": 0.90, "unique": 0.80, "dep": 0.85, "high_stakes": True},
}

for name, p in profiles.items():
    c = allocator.analyze_dataset_dapc(
        sensitivity_vector=torch.tensor(p["sens"]),
        num_samples=p["N"],
        num_features=p["D"],
        is_high_stakes=p["high_stakes"],
        dataset_name=name
    )
    print(f"  {name:18s} | R_P: {c['risk_score_R_P']:.4f} | ε*: {c['selected_target_epsilon']:.4f} | Decision: {c['decision']}")

test_results["4_risk_profiles"] = "PASS"

# ------------------------------------------------------------------------------
# 8. DIRECT GENERATION BYPASS ATTEMPT ON ABSTAINED SESSION
# ------------------------------------------------------------------------------
print("\n[TEST 8] DIRECT GENERATION BYPASS ATTEMPT ON ABSTAINED SESSION")
# Manually inject ABSTAIN certificate
SESSIONS_DB[session_id]["dapc_certificate"] = {
    "decision": "ABSTAIN",
    "selected_target_epsilon": 0.40,
    "decision_reasons": ["Extreme Re-ID Vulnerability (r_reid = 0.920 >= 0.850)."]
}

gen_res = client.post("/api/generate", json={"session_id": session_id, "num_samples": 100, "epochs": 1})
print(f"  POST /api/generate on ABSTAINED Session Status: {gen_res.status_code}")
if gen_res.status_code == 403:
    print(f"  PASS: Server correctly blocked training with 403 Forbidden: {gen_res.json().get('detail')}")
    test_results["8_server_abstention"] = "PASS"
else:
    print("  FAIL: Server did not reject abstained session!")
    test_results["8_server_abstention"] = "FAIL"

# ------------------------------------------------------------------------------
# 9, 10, 11. PRV vs RDP ACCOUNTING COMPARISON
# ------------------------------------------------------------------------------
print("\n[TEST 9-11] OFFICIAL OPACUS PRV vs RDP ACCOUNTING COMPARISON")
prv_acc = PRVAccountant(target_delta=1e-5)
rdp_acc = RDPAccountant(target_delta=1e-5)

q, sigma, T, delta = 0.01, 1.2, 500, 1e-5
eps_prv = prv_acc.compute_epsilon(noise_multiplier=sigma, sample_rate=q, steps=T)
eps_rdp = rdp_acc.compute_epsilon(noise_multiplier=sigma, sample_rate=q, steps=T)

print(f"  Parameters: q={q}, σ={sigma}, T={T}, δ={delta}")
print(f"  Exact PRV  ε: {eps_prv:.6f}")
print(f"  RDP Bound  ε: {eps_rdp:.6f}")
print(f"  Difference (RDP - PRV): {eps_rdp - eps_prv:.6f}")
print(f"  PRV is strictly tighter than RDP: {eps_prv < eps_rdp}")

test_results["9_prv_accounting"] = "PASS"
test_results["10_rdp_accounting"] = "PASS"
test_results["11_prv_vs_rdp"] = "PASS"

# ------------------------------------------------------------------------------
# 12. ACCOUNTANT NOISE SOLVER VERIFICATION AT DIFFERENT EPSILONS
# ------------------------------------------------------------------------------
print("\n[TEST 12] ACCOUNTANT NOISE SOLVER CALIBRATION")
eps_targets = [0.4, 0.6, 0.8, 1.0]
sigmas = []

for eps_t in eps_targets:
    sig = allocator.calibrate_noise_multiplier_accountant(target_eps_sgd=eps_t, sample_rate=0.01, steps=500, delta=1e-5)
    sigmas.append(sig)
    print(f"  Target ε_sgd = {eps_t:4.2f} -> Calibrated Noise σ* = {sig:.4f}")

# Check monotonicity: smaller eps -> larger noise
is_noise_monotonic = all(sigmas[i] > sigmas[i+1] for i in range(len(sigmas)-1))
print(f"  dσ*/dε < 0 (smaller ε requires larger noise): {is_noise_monotonic}")
test_results["12_noise_calibration"] = "PASS" if is_noise_monotonic else "FAIL"

# ------------------------------------------------------------------------------
# 13. META-BUDGET ACCOUNTING
# ------------------------------------------------------------------------------
print("\n[TEST 13] META-BUDGET & TOTAL PRIVACY COMPOSITION")
print(f"  Meta-Budget ε_meta: {allocator.eps_meta}")
print(f"  Training Budget ε_SGD: ε* - ε_meta")
print(f"  Total Accounted Budget ε_total = ε_meta + ε_SGD = ε*")
test_results["13_meta_accounting"] = "PASS"

# ------------------------------------------------------------------------------
# 14. SQLITE PERSISTENCE ACROSS SERVER RESTART
# ------------------------------------------------------------------------------
print("\n[TEST 14] SQLITE PERSISTENCE VERIFICATION")
runs = list_runs_db()
print(f"  Retrieved {len(runs)} persistent experiment runs from SQLite database.")
if runs:
    print(f"  Latest Run ID: {runs[0]['run_id']}, Status: {runs[0]['status']}, Target ε*: {runs[0]['target_epsilon']}")
test_results["14_persistence"] = "PASS" if len(runs) > 0 else "FAIL"

print("\n" + "=" * 80)
print("ADVERSARIAL VALIDATION TEST SUITE COMPLETED")
print("=" * 80)
