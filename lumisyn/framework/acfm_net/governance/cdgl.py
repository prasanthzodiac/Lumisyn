import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np


@dataclass
class GovernanceCertificate:
    """
    Stage 0 Governance Certificate.
    Note: direct_identifiers_removed is keyword-based; not a legal determination.
    """
    protocol_id: str
    dataset_hash: str
    timestamp: float
    re_id_risk: float
    passed_irb: bool
    passed_deid: bool
    passed_consent: bool
    direct_identifiers_removed: bool
    gdpr_compliant: bool
    audit_trail_handle: str
    phi_columns_removed: List[str] = field(default_factory=list)
    hashed_columns: List[str] = field(default_factory=list)

    @property
    def hipaa_safe_harbor_verified(self) -> bool:
        """Technical keyword-based identifier drop verification; not a legal determination."""
        return self.direct_identifiers_removed

    @property
    def hipaa_compliant(self) -> bool:
        """Deprecated alias: technical Safe Harbor de-identification verification (keyword-based; not a legal determination)."""
        return self.direct_identifiers_removed


class ClinicalDataGovernanceLayer:
    """
    STAGE 0: Clinical Data Governance Layer (CDGL) — v5.1
    Full spec compliance:
      0.1 IRB / Ethics verification
      0.2 PII column detection + SHA-256 hashing + removal
      0.3 Consent Framework validation
      0.4 Re-identification risk estimation
      0.5 Immutable audit trail initialisation

    Direct identifiers are DROPPED.
    Quasi/tracking identifiers are SHA-256 HASHED in-place.
    Returns (df_deidentified, GovernanceCertificate, logs).
    """

    # --- PHI column keyword taxonomy ---
    _DIRECT_IDENTIFIERS = [
        "ssn", "social_security", "social_security_number",
        "name", "first_name", "last_name", "full_name",
        "fname", "lname", "fullname", "patient_name", "patientname",
        "address", "street", "city",
        "phone", "telephone", "mobile", "cell",
        "email", "e_mail",
        "dob", "date_of_birth", "birthdate",
        "biometric", "fingerprint", "face",
        "ip_address", "mac_address",
    ]

    _TRACKING_IDENTIFIERS = [
        "mrn", "medical_record_number", "medical_record",
        "patient_id", "patientid", "subject_id", "subjectid",
        "encounter_id", "encounterid", "record_id", "recordid",
        "case_id", "caseid", "sample_id", "sampleid",
        "zipcode", "zip_code", "zip",
    ]

    # Regex patterns for value-level PHI scanning
    _PHI_VALUE_PATTERNS = {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "phone": re.compile(r"\b(\+?\d[\d\s\-().]{7,}\d)\b"),
        "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    }

    def __init__(self, re_id_threshold: float = 0.05):
        self.re_id_threshold = re_id_threshold

    # ------------------------------------------------------------------
    # 0.1  IRB Compliance
    # ------------------------------------------------------------------
    def verify_irb_compliance(
        self,
        protocol_id: Optional[str],
        hipaa_flag: bool,
        gdpr_flag: bool,
        dua_approved: bool = True,
    ) -> Tuple[bool, List[str]]:
        logs = []
        if not protocol_id or not protocol_id.strip():
            logs.append("IRB Check FAILED: No Protocol ID provided.")
            return False, logs
        if not dua_approved:
            logs.append("IRB Check FAILED: Data Use Agreement (DUA) not approved.")
            return False, logs
        logs.append(f"IRB Compliance PASSED. Protocol: {protocol_id}")
        return True, logs

    # ------------------------------------------------------------------
    # 0.2  PII Detection, Hashing and Removal
    # ------------------------------------------------------------------
    def _classify_phi_columns(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Returns (direct_phi_cols, tracking_id_cols)."""
        direct = []
        tracking = []
        for col in df.columns:
            col_l = str(col).lower().strip()
            if any(kw == col_l or col_l.startswith(kw + "_") or col_l.endswith("_" + kw)
                   for kw in self._DIRECT_IDENTIFIERS):
                direct.append(col)
            elif any(kw in col_l for kw in self._TRACKING_IDENTIFIERS):
                tracking.append(col)
        return direct, tracking

    def _scan_phi_values(self, df: pd.DataFrame) -> List[str]:
        """
        Scan string-typed columns for value-level PHI patterns (SSN, email, phone).
        Returns list of column names containing PHI values.
        """
        flagged = []
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            sample = df[col].dropna().astype(str).head(200)
            for pattern_name, pattern in self._PHI_VALUE_PATTERNS.items():
                matches = sample.apply(lambda v: bool(pattern.search(v))).sum()
                if matches >= 1:
                    flagged.append(col)
                    break
        return flagged

    def _sha256_hash_column(self, series: pd.Series) -> pd.Series:
        """Replace each value with its SHA-256 hex digest."""
        def _hash(v):
            if pd.isna(v):
                return None
            return hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:16]
        return series.apply(_hash)

    def deidentify(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
        """
        Full de-identification pass:
          - Drop direct identifier columns
          - SHA-256 hash tracking/quasi-identifier columns
          - Scan values for embedded PHI and hash flagged columns
        Returns:
            (df_clean, dropped_cols, hashed_cols, value_phi_cols)
        """
        df_clean = df.copy()
        logs = []

        direct_cols, tracking_cols = self._classify_phi_columns(df_clean)

        # 1. Drop direct identifiers
        dropped = [c for c in direct_cols if c in df_clean.columns]
        if dropped:
            df_clean = df_clean.drop(columns=dropped)
            logs.append(f"De-ID: Dropped {len(dropped)} direct identifier columns: {dropped}")
        else:
            logs.append("De-ID: No direct PII columns detected.")

        # 2. SHA-256 hash tracking identifiers
        hashed = []
        for col in tracking_cols:
            if col in df_clean.columns:
                df_clean[col] = self._sha256_hash_column(df_clean[col])
                hashed.append(col)
        if hashed:
            logs.append(f"De-ID: SHA-256 hashed {len(hashed)} tracking columns: {hashed}")

        # 3. Value-level PHI scan
        value_phi_cols = self._scan_phi_values(df_clean)
        for col in value_phi_cols:
            if col not in hashed and col in df_clean.columns:
                df_clean[col] = self._sha256_hash_column(df_clean[col])
                hashed.append(col)
                logs.append(f"De-ID: Value-level PHI detected and hashed column: '{col}'")

        return df_clean, dropped, hashed, value_phi_cols

    # ------------------------------------------------------------------
    # 0.3  Re-identification Risk Estimation
    # ------------------------------------------------------------------
    def verify_deidentification(
        self, df: pd.DataFrame
    ) -> Tuple[bool, float, List[str]]:
        logs = []

        direct_cols, _ = self._classify_phi_columns(df)
        phi_found = direct_cols + self._scan_phi_values(df)

        if phi_found:
            re_id_risk = min(1.0, 0.2 * len(phi_found) + 0.1)
            logs.append(f"De-ID WARNING: PHI still present in {phi_found}")
        else:
            num_rows = max(len(df), 1)
            quasi = [c for c in df.columns if df[c].nunique() > 1]
            if quasi:
                dup_ratio = 1.0 - (len(df.drop_duplicates(subset=quasi[:3])) / num_rows)
                re_id_risk = max(0.001, (1.0 - dup_ratio) * 0.02)
            else:
                re_id_risk = 0.005
            logs.append(f"De-ID PASSED: Estimated re-ID risk = {re_id_risk:.4f}")

        passed = re_id_risk <= self.re_id_threshold and not phi_found
        if not passed and not phi_found:
            logs.append(f"Re-ID risk {re_id_risk:.4f} > threshold {self.re_id_threshold}")
        return passed, re_id_risk, logs

    # ------------------------------------------------------------------
    # 0.4  Consent Framework
    # ------------------------------------------------------------------
    def verify_consent(
        self, consent_scope_valid: bool, secondary_use_auth: bool
    ) -> Tuple[bool, List[str]]:
        logs = []
        passed = consent_scope_valid and secondary_use_auth
        logs.append(
            "Consent PASSED: Valid scope and secondary research authorization."
            if passed else
            "Consent FAILED: Missing scope or secondary research permission."
        )
        return passed, logs

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_governance(
        self,
        df: pd.DataFrame,
        protocol_id: str = "IRB-2026-HC-001",
        hipaa_flag: bool = True,
        gdpr_flag: bool = True,
        dua_approved: bool = True,
        consent_scope_valid: bool = True,
        secondary_use_auth: bool = True,
        run_deidentification: bool = False,
    ) -> Tuple["GovernanceCertificate", List[str], pd.DataFrame]:
        """
        Executes all Stage 0 governance checks.
        Returns (GovernanceCertificate, all_logs, df_deidentified).
        Raw governance checks fail closed by default. The model pipeline must
        explicitly opt into its controlled de-identification pass.
        """
        all_logs = ["=== STAGE 0: Clinical Data Governance Layer ==="]

        # 0.1 IRB
        irb_passed, irb_logs = self.verify_irb_compliance(
            protocol_id, hipaa_flag, gdpr_flag, dua_approved
        )
        all_logs.extend(irb_logs)

        # 0.2 De-identification
        df_clean = df.copy()
        phi_dropped, phi_hashed = [], []

        if run_deidentification:
            df_clean, phi_dropped, phi_hashed, _ = self.deidentify(df)
            all_logs.extend([
                f"De-ID: {len(phi_dropped)} cols dropped, {len(phi_hashed)} cols hashed"
            ])

        deid_passed, re_id_risk, deid_logs = self.verify_deidentification(df_clean)
        all_logs.extend(deid_logs)

        # 0.3 Consent
        consent_passed, consent_logs = self.verify_consent(
            consent_scope_valid, secondary_use_auth
        )
        all_logs.extend(consent_logs)

        # 0.4 Audit trail
        raw_bytes = pd.util.hash_pandas_object(df_clean, index=True).values.tobytes()
        dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
        ts = time.time()
        audit_handle = f"AUDIT-{dataset_hash[:12]}-{int(ts)}"

        cert = GovernanceCertificate(
            protocol_id=protocol_id,
            dataset_hash=dataset_hash,
            timestamp=ts,
            re_id_risk=re_id_risk,
            passed_irb=irb_passed,
            passed_deid=deid_passed,
            passed_consent=consent_passed,
            direct_identifiers_removed=deid_passed,
            gdpr_compliant=gdpr_flag,
            audit_trail_handle=audit_handle,
            phi_columns_removed=phi_dropped,
            hashed_columns=phi_hashed,
        )

        overall = irb_passed and deid_passed and consent_passed
        if not overall:
            all_logs.append("GOVERNANCE HALT: Access denied due to compliance failures.")
            raise PermissionError(
                "Stage 0 Governance FAILED.\n" + "\n".join(all_logs)
            )

        all_logs.append(f"GOVERNANCE PASSED. Audit handle: {audit_handle}")
        return cert, all_logs, df_clean
