import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.evaluation.fidelity import ClassifierTwoSampleTestHarness
from acfm_net.evaluation.privacy_eval import _encode_frames

def run_five_repeats():
    print("================================================================================")
    print(" 5-REPEAT EMPIRICAL EVALUATION: REAL-VS-REAL C2ST & NOISE MIA AUC")
    print("================================================================================")
    n = 300
    seeds = [42, 43, 44, 45, 46]

    c2st_scores = []
    noise_mia_scores = []

    c2st_evaluator = ClassifierTwoSampleTestHarness()

    for s in seeds:
        rng = np.random.default_rng(s)
        
        # Generate independent real train and holdout splits (N=300 each, 3 columns)
        age_tr = rng.normal(45, 10, n)
        chol_tr = rng.normal(200, 30, n)
        outcome_tr = (age_tr + 0.1 * chol_tr > 65).astype(int)
        df_train = pd.DataFrame({"age": age_tr, "cholesterol": chol_tr, "outcome": outcome_tr})

        age_ho = rng.normal(45, 10, n)
        chol_ho = rng.normal(200, 30, n)
        outcome_ho = (age_ho + 0.1 * chol_ho > 65).astype(int)
        df_holdout = pd.DataFrame({"age": age_ho, "cholesterol": chol_ho, "outcome": outcome_ho})

        # 1. Real-vs-Real C2ST (5-fold CV on N=300 real vs N=300 holdout)
        res_c2st = c2st_evaluator.evaluate(df_train, df_holdout, seed=s)
        c2st_scores.append(res_c2st["c2st_auc"])

        # 2. Independent Noise MIA AUC (N=300 members vs N=300 holdout against N=300 random noise)
        X_tr, X_ho, _ = _encode_frames(df_train, df_holdout, df_train)
        X_noise = rng.normal(0, 1.0, size=X_tr.shape)

        dist_m = cdist(X_tr, X_noise, metric="euclidean").min(axis=1)
        dist_nm = cdist(X_ho, X_noise, metric="euclidean").min(axis=1)
        scores = np.concatenate([-dist_m, -dist_nm])
        y_true = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
        mia_auc = float(roc_auc_score(y_true, scores))
        noise_mia_scores.append(mia_auc)

        print(f"Seed {s}: Real-vs-Real C2ST AUC = {res_c2st['c2st_auc']:.4f} | Noise MIA AUC = {mia_auc:.4f}")

    mean_c2st = float(np.mean(c2st_scores))
    sd_c2st = float(np.std(c2st_scores, ddof=1))

    mean_mia = float(np.mean(noise_mia_scores))
    sd_mia = float(np.std(noise_mia_scores, ddof=1))

    print("--------------------------------------------------------------------------------")
    print(f"Real-vs-Real C2ST AUC (N=300, 5-fold CV, 5 repeats): {mean_c2st:.4f} +/- {sd_c2st:.4f}")
    print(f"Independent Noise MIA AUC (N=300, 5 repeats):        {mean_mia:.4f} +/- {sd_mia:.4f}")
    print("================================================================================")

if __name__ == "__main__":
    run_five_repeats()
