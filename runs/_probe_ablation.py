import sys, time
sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\framework")
sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\backend")
import numpy as np, pandas as pd, torch, yaml
from sklearn.model_selection import train_test_split
import acfm_net.pipeline as P
from eval_utils import evaluate_tstr

df = pd.read_csv(r"C:\Users\sarat\Documents\Research\lumisyn\data\kaggle_datasets\adult.csv").sample(n=15000, random_state=42).reset_index(drop=True)
schema = yaml.safe_load(open(r"C:\Users\sarat\Documents\Research\lumisyn\data\schemas\adult_schema.yaml"))["columns"]
tr, te = train_test_split(df, test_size=0.2, random_state=42)
tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)

Orig = P.ContinuousOTCFM
class NoAdj(Orig):
    def __init__(self, *a, adj_matrix=None, **k):
        super().__init__(*a, adj_matrix=None, **k)

log = open(r"C:\Users\sarat\Documents\Research\runs\_probe_ablation.log", "w")
for eps_target, adj in [(50.0, True), (1.0, False), (50.0, False)]:
    P.ContinuousOTCFM = Orig if adj else NoAdj
    for seed in (0, 1, 2):
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        p = P.ACFMNetV5Pipeline(target_epsilon=eps_target, target_delta=1e-5, latent_mode="identity",
                                public_schema=schema, enable_dp_recalibration=False)
        syn, rep = p.fit_generate(df_raw=tr, num_samples=len(tr), target_col="class", epochs=5, protocol_id="ABL")
        u = evaluate_tstr(tr, te, syn, "class", "classification")
        line = (f"eps_target={eps_target} adjacency={adj} seed={seed} time={time.time()-t0:.0f}s "
                f"TSTR_AUC={u['tstr_auc']:.3f} eps={rep['privacy_summary']['prv_upper_with_laplace']:.3f} "
                f"sigma={rep['privacy_training']['noise_multiplier']:.3f}")
        print(line, file=log, flush=True)
log.close()
