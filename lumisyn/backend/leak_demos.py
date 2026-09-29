"""
Reproductions of the five leak classes reported in the paper (Section 3.3 / Supplementary S3).
Each demo uses the pre-fix behaviour (re-implemented in a few lines where the fixed code no longer
exposes it) on UCI Adult and shows what an adversary recovers.  CPU only.

    python lumisyn/backend/leak_demos.py   ->  runs/leak_demos.json
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_privacy as ap  # noqa: E402  (sets sys.path; provides encode())
from acfm_net.generative.ot_cfm import ContinuousOTCFM  # noqa: E402
from acfm_net.graph.builder import DomainConstraintGraphBuilder  # noqa: E402
from acfm_net.graph.refinement import PostLatentGraphRefiner  # noqa: E402
from acfm_net.postprocessing.dp_recalibration import DPMarginalRecalibrator  # noqa: E402
from acfm_net.preparation import DistributionPreservingEncoder  # noqa: E402

out = {}
df, schema = ap.v.load_adult(5000)

# 1. Fixed seed on a noise path: regenerate the "random" Laplace noise and subtract it.
rec = DPMarginalRecalibrator({"columns": {"race": schema["race"]}})
released, _ = rec.release_dp_histograms(df, eps_m=0.1, seed=42)            # pre-fix default seed=42
noise = np.random.default_rng(42).laplace(0.0, 1 / 0.1, size=len(schema["race"]["categories"]))
recovered = np.round(released["race"]["dp_counts"] - noise)
true = np.array([float((df["race"].astype(str) == c).sum()) for c in schema["race"]["categories"]])
mask = released["race"]["dp_counts"] > 0                                    # counts clipped at 0 are lost
out["fixed_seed"] = {"cells": int(len(true)), "cells_recovered_exactly": int(np.sum(recovered[mask] == true[mask])),
                     "cells_unclipped": int(mask.sum()), "note": "eps_m=0.1 Laplace histogram, seed 42 as in the pre-fix code"}

# 2. Clipping the noise instead of the query: analytic, supports become disjoint.
c, gap = 1.0, 3.0
out["clipped_noise"] = {"noise_clip": c, "query_gap": gap, "outputs_disjoint": bool(gap > 2 * c),
                        "consequence": "output ranges [f(D)-c, f(D)+c] and [f(D')-c, f(D')+c] do not overlap, so epsilon is infinite"}

# 3. Un-noised statistic stored in a model buffer: invert the normalised adjacency buffer.
X, _ = ap.encode(df, schema)
Xn = X.numpy().astype(np.float64)
cols = [f"f{i}" for i in range(Xn.shape[1])]
_, W0, _ = DomainConstraintGraphBuilder(top_k=15).build_graph(columns=cols, dependency_matrix=np.eye(len(cols), dtype=np.float32),
                                                             sensitivity_vector=np.full(len(cols), 0.5, dtype=np.float32), df_encoded=None, target_col=None)
alpha = 0.60
_, W_ref, _ = PostLatentGraphRefiner(alpha=alpha).refine_graph(nx.Graph(), W0, Xn)   # pre-fix: reads private rows
model = ContinuousOTCFM(continuous_dim=len(cols), cond_dim=0, adj_matrix=torch.from_numpy(np.asarray(W_ref)).float())
N = next(v for k, v in model.state_dict().items() if k.endswith("adj_matrix")).double().numpy()   # what a weight release contains
deg = 1.0 / np.diag(N)                                                      # diag(A+I) = 1, so N_ii = 1/deg_i
A = N * np.sqrt(np.outer(deg, deg)) - np.eye(len(cols))
cos_recovered = (A - alpha * W0) / (1 - alpha)
fe = Xn.T / (np.linalg.norm(Xn.T, axis=1, keepdims=True) + 1e-8)
cos_true = np.abs(fe @ fe.T)
off = ~np.eye(len(cols), dtype=bool)
out["model_buffer"] = {"matrix_size": len(cols), "max_abs_error_recovering_private_cosine": float(np.max(np.abs(cos_recovered[off] - cos_true[off]))),
                       "note": "adversary reads the adj_matrix buffer from the released state_dict and inverts D^-1/2 (A+I) D^-1/2"}

# 4. Data-dependent bin edge: the pre-fix lowest log-bin edge equals the smallest positive private value.
pos = pd.to_numeric(df["capital-gain"]).to_numpy(); pos = pos[pos > 0]
old_edge = max(1.0, float(np.min(pos)))                                     # pre-fix formula
out["bin_edge"] = {"column": "capital-gain", "published_edge": old_edge, "smallest_private_positive_value": float(np.min(pos)),
                   "equal": bool(old_edge == float(np.min(pos))), "fixed_edge": max(1.0, float(schema["capital-gain"]["bounds"][0]))}

# 5. Silent schema fallback: an undeclared bound makes the encoder store the private min / max.
enc = DistributionPreservingEncoder()
partial = {k: v for k, v in schema.items() if k != "age"}                  # age declared nowhere
enc.fit_transform(df[["age", "sex"]].copy(), privacy_schema=partial)
out["schema_fallback"] = {"column": "age", "encoder_raw_min": enc.feature_specs["age"]["raw_min"], "private_min": float(df["age"].min()),
                          "encoder_raw_max": enc.feature_specs["age"]["raw_max"], "private_max": float(df["age"].max()),
                          "leaked": bool(enc.feature_specs["age"]["raw_min"] == float(df["age"].min()))}

path = ap.v.REPO_ROOT / "runs" / "leak_demos.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
