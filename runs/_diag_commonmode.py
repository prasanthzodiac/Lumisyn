import sys; sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\framework"); sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\backend")
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from schema_registry import KNOWN_DATASETS, known_schema_for
import acfm_net.pipeline as P
from eval_utils import evaluate_tstr
ds = [d for d in KNOWN_DATASETS if "Wisconsin" in d["name"]][0]
df = pd.read_csv(ds["file"]).drop(columns=[c for c in ds["drop_cols"]], errors="ignore")
tr, te = train_test_split(df, test_size=0.2, random_state=42); tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
schema = known_schema_for(ds["name"], tr); t = ds["target"]
Orig = P.ContinuousOTCFM
def NoAdj(*a, **k): k["adj_matrix"] = None; return Orig(*a, **k)
for label, cls, eps in [("adj", Orig, 1.0), ("no_adj", NoAdj, 1.0), ("adj_eps50", Orig, 50.0), ("no_adj_eps50", NoAdj, 50.0)]:
    P.ContinuousOTCFM = cls
    aucs, cms = [], []
    for seed in range(3):
        torch.manual_seed(seed); np.random.seed(seed)
        p = P.ACFMNetV5Pipeline(target_epsilon=eps, latent_mode="identity", public_schema=schema, enable_dp_recalibration=False)
        syn, rep = p.fit_generate(df_raw=tr, num_samples=len(tr), target_col=t, epochs=10, protocol_id="DIAG")
        C = np.corrcoef(p.scaled_X_generated.T); off = C[~np.eye(len(C), dtype=bool)]
        cms.append(np.nanmean(off)); aucs.append(evaluate_tstr(tr, te, syn, t, "classification")["tstr_auc"])
    print(f"RESULT {label}: AUC {np.mean(aucs):.3f} {np.round(aucs,3)} | mean off-diag corr of synthetic {np.mean(cms):.3f}", flush=True)
C = np.corrcoef(p.scaled_X_train.T); print("RESULT real mean off-diag corr", np.nanmean(C[~np.eye(len(C), dtype=bool)]))
