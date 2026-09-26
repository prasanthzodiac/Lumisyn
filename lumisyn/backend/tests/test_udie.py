import pandas as pd
import numpy as np
from acfm_net.intelligence import (
    MissingValueAnalyzer,
    OutlierDetectionEngine,
    SensitiveAttributeDetector,
    DependencyEstimationEngine,
    DatasetIntelligenceGenerator,
)

def test_udie_components():
    df = pd.DataFrame({
        "age": [45, 60, np.nan, 58, 70, 22],
        "glucose": [110, 140, 95, 180, 500, 105],  # 500 is an outlier
        "gender": ["M", "F", "F", "M", "M", "F"],
        "diagnosis": [1, 0, 1, 1, 0, 0]
    })

    # Missing Value Analyzer
    missing_analyzer = MissingValueAnalyzer()
    m_info = missing_analyzer.analyze(df)
    assert "global_missing_pct" in m_info
    assert m_info["global_missing_pct"] > 0

    # Outlier Detection Engine
    outlier_engine = OutlierDetectionEngine()
    o_info = outlier_engine.analyze(df)
    assert len(o_info["outlier_scores"]) == len(df)

    # Sensitive Attribute Detector
    sensitive_detector = SensitiveAttributeDetector()
    s_info = sensitive_detector.detect(df)
    assert "sensitivity_vector" in s_info
    assert len(s_info["sensitivity_vector"]) == df.shape[1]

    # Dependency Estimation Engine
    dependency_engine = DependencyEstimationEngine()
    d_info = dependency_engine.analyze(df)
    assert d_info["dependency_matrix"].shape == (df.shape[1], df.shape[1])

    # Dataset2Vec Meta-Encoder
    dataset2vec = DatasetIntelligenceGenerator(fingerprint_dim=256)
    fp_info = dataset2vec.analyze_and_encode(
        df,
        dependency_matrix=d_info["dependency_matrix"],
        sensitivity_vector=s_info["sensitivity_vector"]
    )
    assert len(fp_info["fingerprint"]) == 256
