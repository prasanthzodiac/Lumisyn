import sys, json, time
sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\framework")
sys.path.insert(0, r"C:\Users\sarat\Documents\Research\lumisyn\backend")
import numpy as np, pandas as pd, torch, yaml
from sklearn.model_selection import train_test_split
from acfm_net.pipeline import ACFMNetV5Pipeline
from eval_utils import evaluate_tstr

df = pd.read_csv(r"C:\Users\sarat\Documents\Research\lumisyn\data\kaggle_datasets\adult.csv").sample(n=15000, random_state=42).reset_index(drop=True)
schema = yaml.safe_load(open(r"C:\Users\sarat\Documents\Research\lumisyn\data\schemas\adult_schema.yaml"))["columns"]
tr, te = train_test_split(df, test_size=0.2, random_state=42)
tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)

configs = [(int(a), int(b), c == "1") for a, b, c in (s.split(":") for s in sys.argv[1:])]  # epochs:batch:recal
for epochs, bs, recal in configs:
    for seed in (0,):
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        p = ACFMNetV5Pipeline(target_epsilon=1.0, target_delta=1e-5, latent_mode="identity", public_schema=schema,
                              enable_dp_recalibration=recal, batch_size=bs)
        syn, rep = p.fit_generate(df_raw=tr, num_samples=len(tr), target_col="class", epochs=epochs, protocol_id="PROBE")
        u = evaluate_tstr(tr, te, syn, "class", "classification")
        print(f"RESULT epochs={epochs} bs={bs} recal={recal} seed={seed} time={time.time()-t0:.0f}s "
              f"TSTR_AUC={u['tstr_auc']:.3f} TRTR_AUC={u['trtr_auc']:.3f} eps={rep['privacy_summary']['prv_upper_with_laplace']:.3f} "
              f"final_flow_loss={rep['flow_stability_log'][-1]['loss']:.4f}", flush=True)
