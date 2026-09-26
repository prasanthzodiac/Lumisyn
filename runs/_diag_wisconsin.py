import sys; sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\framework"); sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\backend")
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from schema_registry import KNOWN_DATASETS, known_schema_for
from acfm_net.pipeline import ACFMNetV5Pipeline
from eval_utils import evaluate_tstr
ds = [d for d in KNOWN_DATASETS if "Wisconsin" in d["name"]][0]
df = pd.read_csv(ds["file"]).drop(columns=[c for c in ds["drop_cols"]], errors="ignore")
tr, te = train_test_split(df, test_size=0.2, random_state=42); tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
schema = known_schema_for(ds["name"], tr); t = ds["target"]
print("target spec:", schema[t], "real label counts:", tr[t].value_counts().to_dict())
for recal in (False, True):
    torch.manual_seed(0); np.random.seed(0)
    syn, rep = ACFMNetV5Pipeline(target_epsilon=1.0, latent_mode="identity", public_schema=schema, enable_dp_recalibration=recal).fit_generate(df_raw=tr, num_samples=len(tr), target_col=t, epochs=10, protocol_id="DIAG")
    u = evaluate_tstr(tr, te, syn, t, "classification")
    ylab = lambda d: (d[t].astype(str) == sorted(tr[t].astype(str).unique())[-1]).astype(float)
    num = [c for c in tr.columns if c != t and pd.api.types.is_numeric_dtype(tr[c])]
    cr = np.array([np.corrcoef(tr[c], ylab(tr))[0,1] for c in num]); cs = np.array([np.corrcoef(pd.to_numeric(syn[c]), ylab(syn))[0,1] if ylab(syn).std()>0 else np.nan for c in num])
    print(f"recal={recal} AUC={u['tstr_auc']:.3f} syn label counts={syn[t].value_counts().to_dict()}")
    print("  corr(feature,label) real vs syn, first 8:", np.round(cr[:8],2), np.round(cs[:8],2), " sign agreement:", np.nanmean(np.sign(cr)==np.sign(cs)))
