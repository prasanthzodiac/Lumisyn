"""
Regressions for the two leaks found while writing report/PRIVACY_PROOF.md (§5):
1. Stage 3b must not read private rows in strict mode (its output is the flow's adj_matrix buffer).
2. Stage 7b histogram bin edges must not depend on private values.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))

from acfm_net.graph.refinement import PostLatentGraphRefiner  # noqa: E402
from acfm_net.postprocessing.dp_recalibration import DPMarginalRecalibrator  # noqa: E402


def test_stage3b_refinement_without_data_is_public():
    rng = np.random.default_rng(0)
    W0 = np.abs(rng.normal(size=(6, 6)))
    W0 = (W0 + W0.T) / 2
    refiner = PostLatentGraphRefiner(alpha=0.6)
    _, W_none, _ = refiner.refine_graph(nx.Graph(), W0, None)
    _, W_data, _ = refiner.refine_graph(nx.Graph(), W0, rng.normal(size=(50, 6)))
    expected = W0.copy()
    np.fill_diagonal(expected, 0.0)
    assert np.allclose(W_none, expected)          # public W0 only
    assert not np.allclose(W_data, W_none)        # the non-strict path does read data


def test_stage7b_bin_edges_do_not_depend_on_private_values():
    schema = {"gain": {"type": "numeric", "bounds": [0.0, 99999.0], "kind": "continuous", "zero_inflated": True}}
    rec = DPMarginalRecalibrator(schema)
    a = pd.DataFrame({"gain": [0.0] * 50 + [3.0, 500.0, 7000.0]})
    b = pd.DataFrame({"gain": [0.0] * 50 + [950.0, 20000.0, 99000.0]})
    ha, _ = rec.release_dp_histograms(a, eps_m=1.0)
    hb, _ = rec.release_dp_histograms(b, eps_m=1.0)
    assert np.array_equal(ha["gain"]["pos_bins"], hb["gain"]["pos_bins"])
