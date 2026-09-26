import sys, time; sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\backend")
import verify_icfm_sensitivity as v
import numpy as np, torch
from sklearn.model_selection import train_test_split
from acfm_net.pipeline import ACFMNetV5Pipeline
from eval_utils import evaluate_tstr
df, schema = v.load_adult(v.N_ROWS); tr, te = train_test_split(df, test_size=0.2, random_state=42); tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
torch.manual_seed(0); np.random.seed(0); t=time.time()
p = ACFMNetV5Pipeline(target_epsilon=1.0, latent_mode="identity", public_schema=schema, conditional_marginals="adaptive", noise_multiplicity=4, flow_hidden_dim=64)
syn, rep = p.fit_generate(df_raw=tr, num_samples=len(tr), target_col="class", epochs=5, protocol_id="SMOKE")
u = evaluate_tstr(tr, te, syn, "class", "classification")
print("SMOKE", round(time.time()-t,1), "s AUC", u["tstr_auc"], "eps", rep["privacy_summary"]["prv_upper_with_laplace"], rep["dapc_certificate"].get("conditional_marginals"), "sigma", rep["privacy_training"]["noise_multiplier"], "verified", rep["release_privacy_verified"])
print(syn.head(3).T)
