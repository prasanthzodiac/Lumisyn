"""
End-to-end smoke test of the demo API against a running backend (not collected by pytest: no test_ prefix).

    python lumisyn/backend/tests/demo_smoke.py [preset_id]      # needs the backend on 127.0.0.1:8000

Checks every endpoint the UI calls, every per-stage PDF, and basic output quality.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(path, method="GET", body=None, form=None):
    data, headers = None, {}
    if body is not None:
        data, headers = json.dumps(body).encode(), {"Content-Type": "application/json"}
    if form is not None:
        boundary = "----lumisyn"
        data = "".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n" for k, v in form.items())
        data = (data + f"--{boundary}--\r\n").encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
        return raw if r.headers.get_content_type() != "application/json" else json.loads(raw)


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "nhanes"
    up = call("/api/upload", "POST", form={"preset_id": preset})
    sid, target = up["session_id"], up["suggested_target"]
    print("upload ok", sid[:8], "schema:", up["schema_source"])
    dapc = call(f"/api/dapc/analyze/{sid}")
    print("dapc", dapc["decision"], "eps*", dapc["selected_target_epsilon"])
    t0 = time.time()
    gen = call("/api/generate", "POST", body={"session_id": sid, "num_samples": 1000, "target_epsilon": 1.0, "epochs": 10, "target_col": target})
    stages = {s["stage"]: s["status"] for s in gen["stage_report"]}
    print(f"generate {gen['pipeline_status']} in {time.time() - t0:.0f}s eps={gen['report_summary']['prv_eps_used']} 7c={gen['report_summary']['stage_7c_marginals']}")
    print("stages", stages)
    assert gen["pipeline_status"] == "SUCCESS"
    assert all(v in {"PASSED", "READY", "SKIPPED"} for v in stages.values()), stages
    cmp = call(f"/api/compare/{sid}")
    ratio = cmp["class_ratio_summary"]
    print("target rate real/syn", ratio["real_positive_ratio"], ratio["syn_positive_ratio"], "| BP rule violations", cmp["clinical_rule_violations"])
    worst = max(cmp["feature_summary"], key=lambda f: f["mean_abs_error"] / max(1e-9, f["real_std"]))
    print("worst mean error (in real SDs):", worst["feature"], round(worst["mean_abs_error"] / worst["real_std"], 3))
    ev = call(f"/api/evaluate/{sid}")
    print("evaluate keys", sorted(ev)[:8], "TSTR", ev.get("utility", {}).get("tstr", {}).get("accuracy"))
    for path in [f"/api/stages/{sid}", f"/api/graph/{sid}", f"/api/dapc/certificate/{sid}", "/api/runs", f"/api/powerbi/export/{sid}"]:
        call(path)
    assert call(f"/api/download/{sid}")[:20]
    assert call(f"/api/report/pdf/{sid}")[:4] == b"%PDF"
    assert call(f"/api/report/stages/{sid}")[:4] == b"%PDF"
    for st in stages:
        assert call(f"/api/report/stage/{sid}/{st}")[:4] == b"%PDF", st
    print(f"all {len(stages)} stage PDFs + full stage PDF + evaluation PDF ok")
    assert cmp["clinical_rule_violations"] == 0
    assert abs(ratio["real_positive_ratio"] - ratio["syn_positive_ratio"]) < 0.08
    print("DEMO SMOKE TEST PASSED", sid)


if __name__ == "__main__":
    main()
