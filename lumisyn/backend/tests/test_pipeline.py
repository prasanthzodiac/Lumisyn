import pandas as pd
import numpy as np
from acfm_net.pipeline import ACFMNetV5Pipeline

def test_full_pipeline_mock_ehr():
    np.random.seed(42)
    n = 100
    mock_df = pd.DataFrame({
        "age": np.random.randint(18, 90, size=n),
        "glucose": np.random.uniform(70, 300, size=n),
        "systolic_bp": np.random.uniform(90, 180, size=n),
        "diastolic_bp": np.random.uniform(60, 110, size=n),
        "bmi": np.random.uniform(18.5, 45.0, size=n),
        "gender": np.random.choice(["M", "F"], size=n),
        "diabetes_outcome": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })

    # Add artificial missingness
    mock_df.loc[::10, "glucose"] = np.nan

    pipeline = ACFMNetV5Pipeline(target_epsilon=1.0, target_delta=1e-5)
    df_synthetic, report = pipeline.fit_generate(
        mock_df,
        num_samples=50,
        target_col="diabetes_outcome",
        epochs=3,
        protocol_id="IRB-TEST-2026"
    )

    assert report["pipeline_status"] == "SUCCESS"
    assert df_synthetic.shape == (50, mock_df.shape[1])
    assert list(df_synthetic.columns) == list(mock_df.columns)
    assert not df_synthetic.isna().any().any()
    assert report["privacy_training"]["verified"] is True
    assert report["privacy_training"]["steps"] == report["privacy_training_plan"]["planned_steps"]
    assert len(report["stage_reports"]) >= 10
    assert report["privacy_status"] == "VERIFIED_CONDITIONAL_PUBLIC_SCHEMA"
    assert report["release_privacy_verified"] is True
    assert report["privacy_composition"]["epsilon"] <= report["target_epsilon_adaptive"]
