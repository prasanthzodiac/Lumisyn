import pandas as pd
import numpy as np
from acfm_net.preparation import (
    AdaptiveImputer,
    DistributionPreservingEncoder,
    AdaptiveFeatureScaler,
)

def test_invertible_encoding_decoding():
    df = pd.DataFrame({
        "age": [45.0, 60.0, 32.0, 58.0],
        "glucose": [110.0, 140.0, 95.0, 180.0],
        "gender": ["M", "F", "F", "M"],
        "stage": ["I", "II", "III", "IV"]
    })

    imputer = AdaptiveImputer()
    df_imp, _ = imputer.fit_transform(df)
    assert not df_imp.isna().any().any()

    encoder = DistributionPreservingEncoder()
    encoded_X, info = encoder.fit_transform(df_imp)
    
    # Verify inverse decoding match
    df_decoded = encoder.decode(encoded_X)
    assert list(df_decoded.columns) == list(df.columns)
    assert (df_decoded["gender"].values == df["gender"].values).all()
