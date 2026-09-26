"""
API Verification Suite for ACFM-Net v5 Production FastAPI Backend
Tests Health, Upload, Synthetic Generation, Statistical Comparison, and Power BI Export endpoints.
"""

import sys
import json
import urllib.request
import urllib.parse
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def run_backend_tests():
    print("=" * 80)
    print("   ACFM-Net v5 FASTAPI BACKEND VERIFICATION SUITE")
    print("=" * 80)

    # 1. Healthcheck Test
    print("\n [TEST 1] Testing GET /api/health...")
    req = urllib.request.Request(f"{BASE_URL}/api/health")
    with urllib.request.urlopen(req) as resp:
        health_data = json.loads(resp.read().decode("utf-8"))
        print("   ✓ Health Status:", health_data["status"], "| System:", health_data["system"])
        assert health_data["status"] == "online"

    # 2. Preset Datasets List Test
    print("\n [TEST 2] Testing GET /api/datasets/preset...")
    req = urllib.request.Request(f"{BASE_URL}/api/datasets/preset")
    with urllib.request.urlopen(req) as resp:
        presets = json.loads(resp.read().decode("utf-8"))
        print(f"   ✓ Returned {len(presets)} preset datasets:", [p["id"] for p in presets])

    # 3. Upload Test (Using Preset NHANES)
    print("\n [TEST 3] Testing POST /api/upload (NHANES Preset)...")
    data = urllib.parse.urlencode({"preset_id": "nhanes"}).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/api/upload", data=data, method="POST")
    with urllib.request.urlopen(req) as resp:
        upload_data = json.loads(resp.read().decode("utf-8"))
        session_id = upload_data["session_id"]
        print(f"   ✓ Session Created: {session_id}")
        print(f"   ✓ Rows: {upload_data['row_count']} | Cols: {upload_data['column_count']}")
        print(f"   ✓ Suggested Target: {upload_data['suggested_target']}")
        print(f"   ✓ Re-identification Risk: {upload_data['re_id_risk']}")

    # 4. Generate Synthetic Data Test
    print("\n [TEST 4] Testing POST /api/generate (All 10 Stages Execution)...")
    gen_payload = json.dumps({
        "session_id": session_id,
        "num_samples": 500,
        "target_epsilon": 1.0,
        "epochs": 5,
        "protocol_id": "IRB-TEST-2026"
    }).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/api/generate", data=gen_payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        gen_data = json.loads(resp.read().decode("utf-8"))
        print(f"   ✓ Generation Status: {gen_data['status']}")
        print(f"   ✓ Samples Generated: {gen_data['generated_samples']}")
        print(f"   ✓ Generation Time: {gen_data['generation_time_sec']}s")

    # 5. Statistical Comparison Test
    print("\n [TEST 5] Testing GET /api/compare/{session_id}...")
    req = urllib.request.Request(f"{BASE_URL}/api/compare/{session_id}")
    with urllib.request.urlopen(req) as resp:
        comp_data = json.loads(resp.read().decode("utf-8"))
        print(f"   ✓ Dataset: {comp_data['dataset_name']}")
        print(f"   ✓ Frobenius Correlation Error: {comp_data['correlation_frobenius_error']}")
        print(f"   ✓ Clinical Rule Pass Rate: {comp_data['clinical_rule_pass_rate']}%")
        print(f"   ✓ DP Budget Epsilon Used: {comp_data['privacy']['eps_used']}")

    # 6. Power BI Export Test
    print("\n [TEST 6] Testing GET /api/powerbi/export/{session_id}...")
    req = urllib.request.Request(f"{BASE_URL}/api/powerbi/export/{session_id}")
    with urllib.request.urlopen(req) as resp:
        pbi_data = json.loads(resp.read().decode("utf-8"))
        print(f"   ✓ Model Name: {pbi_data['pbix_schema']['model_name']}")
        print(f"   ✓ DAX Measures Count: {len(pbi_data['dax_measures'])}")

    print("\n" + "=" * 80)
    print("  ✓ ALL FASTAPI BACKEND API TESTS PASSED SUCCESSFULLY (100% SUCCESS)")
    print("=" * 80)

if __name__ == "__main__":
    time.sleep(1)
    run_backend_tests()
