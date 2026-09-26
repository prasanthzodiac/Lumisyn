import sys; sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\framework"); sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\backend")
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from schema_registry import KNOWN_DATASETS, known_schema_for
from acfm_net.pipeline import ACFMNetV5Pipeline
from eval_utils import evaluate_tstr
for name in ("Pima", "Wisconsin"):
    ds = [d for d in KNOWN_DATASETS if name in d["name"]][0]
    df = pd.read_csv(ds["file"]).drop(columns=[c for c in ds["drop_cols"]], errors="ignore")
    tr, te = train_test_split(df, test_size=0.2, random_state=42); tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    schema = known_schema_for(ds["name"], tr); t = ds["target"]
    for bs, ep, eps in [(None, 10, 1.0), (64, 10, 1.0), (64, 40, 1.0), (128, 40, 1.0), (32, 40, 1.0), (64, 200, 50.0)]:
        aucs = []
        for seed in range(5):
            torch.manual_seed(seed); np.random.seed(seed)
            p = ACFMNetV5Pipeline(target_epsilon=eps, latent_mode="identity", public_schema=schema, batch_size=bs)
            syn, rep = p.fit_generate(df_raw=tr, num_samples=len(tr), target_col=t, epochs=ep, protocol_id="PROBE")
            aucs.append(evaluate_tstr(tr, te, syn, t, "classification")["tstr_auc"])
        print(f"RESULT {name} bs={bs} ep={ep} eps={rep['privacy_summary']['prv_upper_with_laplace']:.3f} sigma={rep['privacy_training']['noise_multiplier']:.2f} steps={rep['privacy_training']['steps']} AUC {np.mean(aucs):.3f}+-{np.std(aucs,ddof=1):.3f}", flush=True)
