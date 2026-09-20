import re
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "runs" / "gate4d_experiments.log"
JSONL_PATH = REPO_ROOT / "runs" / "gate4d_results.jsonl"

def parse_log():
    if not LOG_PATH.exists():
        print(f"Log not found at {LOG_PATH}")
        return

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Pattern for Part A3 results
    # --- Testing Config: Opt=ADAM, lr=0.0005, C=1.5411 (0.5x) (3 seeds) ---
    #   Result: Clipped Frac=100.00% | Mean Grad Norm=3.1887 | Norm 1wTVD=80.75x | Norm 2wTVD=36.71x | Norm Corr=8.85x | TSTR AUC=0.6094
    pattern = re.compile(
        r"---\s*Testing Config:\s*Opt=([A-Za-z0-9_]+),\s*lr=([0-9eE.-]+),\s*C=([0-9.]+)\s*\(([0-9.]+)x\)\s*\((\d+)\s*seeds\)\s*---\s*\n"
        r"\s*Result:\s*Clipped Frac=([0-9.]+)%\s*\|\s*Mean Grad Norm=([0-9.]+)\s*\|\s*Norm 1wTVD=([0-9.]+)x\s*\|\s*Norm 2wTVD=([0-9.]+)x\s*\|\s*Norm Corr=([0-9.]+)x\s*\|\s*TSTR AUC=([0-9.]+)",
        re.MULTILINE
    )

    results = []
    for match in pattern.finditer(text):
        opt, lr, c_val, c_mult, n_seeds, clip_frac, grad_norm, n_1w, n_2w, n_corr, tstr_auc = match.groups()
        entry = {
            "part": "A3",
            "optimizer": opt.lower(),
            "lr": float(lr),
            "C": float(c_val),
            "C_mult": float(c_mult),
            "n_seeds": int(n_seeds),
            "clipped_frac": float(clip_frac),
            "mean_grad_norm": float(grad_norm),
            "norm_1way_tvd": float(n_1w),
            "norm_2way_tvd": float(n_2w),
            "norm_frob_corr": float(n_corr),
            "tstr_auc": float(tstr_auc),
            "stage": 2, # E=10, 3 seeds
            "epochs": 10,
            "status": "completed"
        }
        results.append(entry)

    print(f"Parsed {len(results)} completed A3 configs from log.")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {JSONL_PATH}")

if __name__ == "__main__":
    parse_log()
