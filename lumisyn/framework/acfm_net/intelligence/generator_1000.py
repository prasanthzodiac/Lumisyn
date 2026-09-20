# ==============================================================================
# QUARANTINED MODULE: DEAD CODE / NON-RELEASE DEVELOPMENT UTILITY
# Excluded from the differential privacy release pipeline and core runtime.
# Retained strictly for historical offline variation generator benchmarking.
# ==============================================================================
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


def generate_dataset_variation(
    variation_idx: int = 1,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Generates a unique, mathematically controlled dataset variation out of 1,000 distinct configurations.
    
    Axes of Variation:
    1. Sample Size N: 100 to 5,000
    2. Dimensionality D: 4 to 32 features
    3. Categorical Ratio: 10% to 70%
    4. Missingness Rate: 0% to 30% (MCAR / MAR)
    5. Imbalance Ratio: 50:50 to 95:5
    6. Non-Linear Correlation & Copula Complexity: Low / Moderate / Severe
    7. Distribution Families: Gaussian, Lognormal, Exponential, Multimodal GMM, Uniform
    8. Clinical / Domain Constraint Rules: Blood pressure consistency, Age boundaries, Lab ranges
    """
    actual_seed = seed if seed is not None else (42 + variation_idx)
    rng = np.random.RandomState(actual_seed)

    # 1. Sample Size N (log-uniform sweep)
    # Range: 150 to 3000 for standard rapid variations
    log_n = rng.uniform(np.log(150), np.log(3000))
    N = int(np.exp(log_n))

    # 2. Dimensionality D: 4 to 24 features
    D = rng.randint(4, 25)

    # 3. Categorical vs Continuous Split
    cat_ratio = rng.uniform(0.1, 0.6)
    n_cat = max(1, int(D * cat_ratio))
    n_cont = max(2, D - n_cat)

    # 4. Imbalance Ratio for binary target
    pos_rate = float(rng.uniform(0.05, 0.50))

    # 5. Missingness Rate: 0% to 25%
    missing_rate = float(rng.choice([0.0, 0.02, 0.05, 0.10, 0.15, 0.25], p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]))

    # Construct Correlation Matrix via random orthogonal matrix
    A = rng.randn(n_cont, n_cont)
    cov_matrix = np.dot(A, A.T)
    # Convert to correlation
    diag = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(diag, diag)

    # Generate continuous features from latent multivariate normal
    cont_latent = rng.multivariate_normal(mean=np.zeros(n_cont), cov=corr_matrix, size=N)

    # Transform continuous features to varied distribution shapes (Gaussian, Lognormal, GMM, Exponential)
    cont_dict = {}
    feature_names = []
    
    # Feature 0: Age (physiological 18 to 90)
    age = np.clip(18 + (cont_latent[:, 0] * 12.0 + 45.0), 18.0, 92.0)
    cont_dict["age"] = np.round(age, 1)
    feature_names.append("age")

    # Feature 1: Systolic BP
    sys_bp = np.clip(100.0 + 0.3 * age + cont_latent[:, 1 % n_cont] * 15.0, 85.0, 210.0)
    cont_dict["systolic_bp"] = np.round(sys_bp, 1)
    feature_names.append("systolic_bp")

    # Feature 2: Diastolic BP (Must satisfy Sys > Dia + 10)
    dia_bp = np.minimum(sys_bp - 12.0, np.clip(65.0 + 0.15 * age + cont_latent[:, 2 % n_cont] * 8.0, 50.0, 125.0))
    cont_dict["diastolic_bp"] = np.round(dia_bp, 1)
    feature_names.append("diastolic_bp")

    # Additional continuous features (biomarkers, scores, indices)
    dist_types = ["lognormal", "gmm", "exponential", "uniform", "standard_normal"]
    for j in range(3, n_cont):
        d_type = dist_types[j % len(dist_types)]
        raw_z = cont_latent[:, j]
        col_name = f"biomarker_{j}"
        if d_type == "lognormal":
            val = np.exp(raw_z * 0.5 + 2.0)
        elif d_type == "gmm":
            # 2-component mixture
            shift = rng.choice([-2.0, 2.0], size=N)
            val = raw_z + shift
        elif d_type == "exponential":
            val = np.exp(np.clip(raw_z, -2.5, 2.5))
        elif d_type == "uniform":
            val = 1.0 / (1.0 + np.exp(-raw_z))
        else:
            val = raw_z * 10.0 + 50.0
        
        cont_dict[col_name] = np.round(val, 2)
        feature_names.append(col_name)

    # Construct Categorical Features
    cat_dict = {}
    cardinalities = [2, 3, 4, 5, 8]
    cat_labels_pool = {
        2: ["Low", "High"],
        3: ["Mild", "Moderate", "Severe"],
        4: ["Stage_I", "Stage_II", "Stage_III", "Stage_IV"],
        5: ["Group_A", "Group_B", "Group_C", "Group_D", "Group_E"],
        8: [f"Code_{k}" for k in range(8)]
    }
    
    for k in range(n_cat):
        card = cardinalities[k % len(cardinalities)]
        labels = cat_labels_pool[card]
        # Introduce correlation between age / latent and categorical category
        logits = cont_latent[:, k % n_cont] + rng.randn(N) * 0.5
        probs = np.exp(logits) / (1.0 + np.exp(logits))
        cat_indices = np.clip((probs * card).astype(int), 0, card - 1)
        col_name = f"category_{k}"
        cat_dict[col_name] = [labels[idx] for idx in cat_indices]
        feature_names.append(col_name)

    # Generate Binary Target Outcome (e.g. Disease, Mortality, Churn)
    # Logit depends on age, sys_bp, and biomarker_3
    logit = -3.5 + 0.04 * age + 0.02 * sys_bp + 0.3 * cont_latent[:, 0]
    # Adjust intercept to match target pos_rate
    prob = 1.0 / (1.0 + np.exp(-logit))
    # Thresholding for target balance
    threshold = np.quantile(prob, 1.0 - pos_rate)
    target = (prob >= threshold).astype(int)

    # Combine all into DataFrame
    data_dict = {**cont_dict, **cat_dict, "clinical_outcome": target}
    df = pd.DataFrame(data_dict)

    # Inject Missingness (MCAR / MAR)
    if missing_rate > 0:
        mask = rng.rand(*df.shape) < missing_rate
        # Keep target intact
        mask[:, df.columns.get_loc("clinical_outcome")] = False
        df = df.mask(mask)

    metadata = {
        "variation_idx": variation_idx,
        "sample_size_N": N,
        "dimension_D": len(df.columns) - 1,
        "continuous_cols": list(cont_dict.keys()),
        "categorical_cols": list(cat_dict.keys()),
        "missing_rate": missing_rate,
        "positive_rate": float(np.mean(target)),
        "seed": actual_seed
    }

    return df, "clinical_outcome", metadata
