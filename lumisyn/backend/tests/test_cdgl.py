import pytest
import pandas as pd
from acfm_net.governance import ClinicalDataGovernanceLayer, GovernanceCertificate

def test_cdgl_pass():
    df = pd.DataFrame({
        "age": [45, 60, 32, 58],
        "glucose": [110, 140, 95, 180],
        "systolic_bp": [120, 135, 115, 150]
    })
    cdgl = ClinicalDataGovernanceLayer()
    cert, logs, cleaned = cdgl.process_governance(df, protocol_id="IRB-TEST-123")
    assert isinstance(cert, GovernanceCertificate)
    assert cert.passed_irb is True
    assert cert.passed_deid is True
    assert cert.passed_consent is True
    assert cert.dataset_hash is not None
    assert cleaned.shape == df.shape

def test_cdgl_fail_direct_phi():
    df = pd.DataFrame({
        "ssn": ["123-45-6789", "987-65-4321"],
        "name": ["Alice", "Bob"]
    })
    cdgl = ClinicalDataGovernanceLayer()
    with pytest.raises(PermissionError):
        cdgl.process_governance(df, protocol_id="IRB-TEST-123")
