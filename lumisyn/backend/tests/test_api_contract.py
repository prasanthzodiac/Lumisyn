from fastapi.testclient import TestClient

from backend.app_backend import app


def test_lumisyn_health_and_upload_contract():
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["framework_connected"] is True

    upload = client.post("/api/upload", data={"preset_id": "nhanes"})
    assert upload.status_code == 200
    payload = upload.json()
    assert payload["suggested_target"] == "diabetes"
    assert payload["session_id"]

    stages = client.get(f"/api/stages/{payload['session_id']}")
    assert stages.status_code == 200
    assert stages.json()["pipeline_status"] == "READY"
