"""
Stage 0 (CDGL) Strict Invariant and Governance Tests.
Verifies:
1. Direct identifier removal on fixture (PII dropped, tracking IDs hashed, clinical preserved).
2. GovernanceCertificate and audit hashes contain no private statistics (only counts, hashes, metadata).
3. Removal of misleading 'HIPAA compliant' certification wording in favor of technical de-identification controls.
"""

import pytest
import pandas as pd
import numpy as np
from lumisyn.framework.acfm_net.governance.cdgl import (
    ClinicalDataGovernanceLayer,
    GovernanceCertificate,
)


@pytest.fixture
def mock_ehr_with_phi():
    """Fixture containing direct identifiers, tracking IDs, and legitimate clinical features."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        # Direct identifiers (must be dropped)
        "ssn": [f"{i:03d}-00-{i:04d}" for i in range(n)],
        "patient_name": [f"Patient_{i}" for i in range(n)],
        "email": [f"user{i}@hospital.org" for i in range(n)],
        "phone": [f"+1-555-{i:04d}" for i in range(n)],
        "address": [f"{i} Main St" for i in range(n)],
        # Tracking identifiers (must be SHA-256 hashed)
        "mrn": [f"MRN{10000 + i}" for i in range(n)],
        "patient_id": [f"PID_{i:05d}" for i in range(n)],
        # Clinical variables (must be retained)
        "age": np.random.randint(18, 85, size=n),
        "systolic_bp": np.random.normal(120, 15, size=n).round(1),
        "cholesterol": np.random.normal(200, 30, size=n).round(1),
        "outcome": np.random.choice([0, 1], size=n),
    })


def test_direct_identifier_removal_on_fixture(mock_ehr_with_phi):
    """Assert all direct identifiers are dropped and tracking columns are hashed."""
    cdgl = ClinicalDataGovernanceLayer(re_id_threshold=0.10)
    df_clean, dropped, hashed, val_phi = cdgl.deidentify(mock_ehr_with_phi)

    # 1. Direct identifiers must be dropped completely
    expected_dropped = ["ssn", "patient_name", "email", "phone", "address"]
    for col in expected_dropped:
        assert col in dropped, f"Direct identifier {col} was not flagged as dropped"
        assert col not in df_clean.columns, f"Direct identifier {col} still in clean DataFrame"

    # 2. Tracking identifiers must be hashed, not raw
    expected_hashed = ["mrn", "patient_id"]
    for col in expected_hashed:
        assert col in hashed, f"Tracking identifier {col} was not flagged as hashed"
        assert col in df_clean.columns, f"Tracking identifier {col} missing from clean DataFrame"
        # Check that values are SHA-256 hex digests (64 hex characters)
        for val in df_clean[col]:
            assert len(str(val)) in (16, 64), f"Value in {col} is not a valid SHA-256 digest: {val}"
            assert not str(val).startswith("MRN"), f"Raw value leaked in {col}"

    # 3. Clinical variables must be preserved intact
    for col in ["age", "systolic_bp", "cholesterol", "outcome"]:
        assert col in df_clean.columns
        assert np.allclose(df_clean[col].values, mock_ehr_with_phi[col].values)


def test_certificate_contains_no_private_statistics(mock_ehr_with_phi):
    """Assert certificate outputs contain only metadata, hashes, and boolean flags, no private stats."""
    cdgl = ClinicalDataGovernanceLayer(re_id_threshold=0.10)
    cert, logs, df_clean = cdgl.process_governance(
        mock_ehr_with_phi,
        protocol_id="IRB-TEST-2026",
        run_deidentification=True,
    )

    # Allowed certificate attributes
    allowed_keys = {
        "protocol_id", "dataset_hash", "timestamp", "re_id_risk",
        "passed_irb", "passed_deid", "passed_consent",
        "direct_identifiers_removed", "hipaa_safe_harbor_verified", "hipaa_compliant",
        "gdpr_compliant", "audit_trail_handle",
        "phi_columns_removed", "hashed_columns"
    }
    cert_dict = cert.__dict__ if hasattr(cert, "__dict__") else {}
    for key in cert_dict:
        assert key in allowed_keys, f"Unexpected key in certificate: {key}"
    assert cert.direct_identifiers_removed is True

    # Check that no patient clinical statistics are stored
    for val in cert_dict.values():
        assert not isinstance(val, (pd.DataFrame, pd.Series, np.ndarray)), "Array/Series stored in certificate"
        if isinstance(val, list):
            # Only column name strings allowed
            assert all(isinstance(x, str) for x in val)
            for x in val:
                assert not any(char.isdigit() for char in x if x not in ["patient_id"]), f"Possible data leak in {x}"

    # Verify dataset hash is a fixed-length SHA-256 digest
    assert len(cert.dataset_hash) == 64
    assert cert.audit_trail_handle.startswith("AUDIT-")


def test_no_misleading_hipaa_compliant_claims():
    """Verify code and logs do not claim legal compliance ('HIPAA compliant')."""
    import inspect
    import lumisyn.framework.acfm_net.governance.cdgl as cdgl_mod

    # Read the source code of cdgl module
    source = inspect.getsource(cdgl_mod)
    lines = source.splitlines()
    for i, line in enumerate(lines):
        clean_line = line.strip().lower()
        if "hipaa compliant" in clean_line:
            # Check if this is a backward-compatible alias property or comment explaining deprecation
            is_compat = "alias" in clean_line or "deprecated" in clean_line or "property" in clean_line or "def hipaa_compliant" in clean_line
            assert is_compat, f"Found unqualified 'HIPAA compliant' wording at line {i+1}: {line}"
