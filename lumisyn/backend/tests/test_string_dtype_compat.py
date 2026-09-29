"""
Regression tests for pandas StringDtype text columns (the default str dtype in
pandas>=3). Text checks written against dtype "object" silently skip these.
"""
import warnings

import pandas as pd

from acfm_net.governance import ClinicalDataGovernanceLayer
from acfm_net.intelligence.sensitive import SensitiveAttributeDetector


def test_cdgl_phi_value_scan_covers_string_dtype():
    df = pd.DataFrame({
        "note": pd.Series(["contact 123-45-6789", "none"], dtype="string"),
        "glucose": [110, 140],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", FutureWarning)
        flagged = ClinicalDataGovernanceLayer()._scan_phi_values(df)
    assert flagged == ["note"]


def test_sensitive_high_cardinality_text_scored_for_string_dtype():
    codes = [f"code-{i}" for i in range(20)]
    df = pd.DataFrame({
        "ref_as_object": pd.Series(codes, dtype=object),
        "ref_as_string": pd.Series(codes, dtype="string"),
    })
    result = SensitiveAttributeDetector().detect(df, use_data_dependent_risk=True)
    assert result["sensitivity_map"]["ref_as_object"] == 0.85
    assert result["sensitivity_map"]["ref_as_string"] == 0.85
