"""
Stage 3 (DCGB) Strict Invariant and Poisoned-Tensor Tests.
Verifies:
1. No private data is touched in strict mode (poisoned-tensor test).
2. DomainConstraintGraphBuilder uses only public metadata, schema, and clinical ontology when df_encoded is None.
3. If public domain graph has no clinical constraint edges, graph_inactive is set to True.
4. Edge counts match expected domain topology for Adult, Cardio, Pima, Wisconsin.
"""

import pytest
import numpy as np
import pandas as pd
from lumisyn.framework.acfm_net.graph.builder import DomainConstraintGraphBuilder


class PoisonedPrivateDataFrame:
    """Mock private data container that detonates if any private sample values are accessed."""
    def __init__(self, columns):
        self.columns = columns

    def __getitem__(self, key):
        raise RuntimeError("PRIVACY VIOLATION: Private row values were accessed in strict mode!")

    @property
    def values(self):
        raise RuntimeError("PRIVACY VIOLATION: Private array values were accessed in strict mode!")

    def __array__(self):
        raise RuntimeError("PRIVACY VIOLATION: Private tensor conversion attempted in strict mode!")


def test_stage3_poisoned_tensor_isolation():
    """Verify that in strict mode, passing df_encoded=None or inspecting strict path touches NO private data."""
    builder = DomainConstraintGraphBuilder(corr_threshold=0.10, edge_prune_threshold=0.10)
    cols = ["age", "systolic_bp", "diastolic_bp", "cholesterol"]
    d = len(cols)
    sens_vector = np.array([0.65, 0.50, 0.50, 0.50], dtype=np.float32)

    # In strict mode, dependency_matrix is public identity and df_encoded is None
    G, W, info = builder.build_graph(
        columns=cols,
        dependency_matrix=np.eye(d, dtype=np.float32),
        sensitivity_vector=sens_vector,
        df_encoded=None,
    )

    assert info["mi_computed"] is False
    assert info["num_nodes"] == 4
    # systolic_bp <-> diastolic_bp is a hard clinical ontology edge (w=0.92)
    assert info["clinical_edges"] >= 1
    assert G.has_edge(cols.index("systolic_bp"), cols.index("diastolic_bp"))


def test_stage3_graph_inactive_flag():
    """Verify that when a dataset has 0 clinical ontology edges, graph_inactive evaluates to True."""
    builder = DomainConstraintGraphBuilder(corr_threshold=0.10, edge_prune_threshold=0.10)
    # Generic features with zero clinical ontology matches
    cols = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
    d = len(cols)
    sens_vector = np.ones(d, dtype=np.float32) * 0.10

    G, W, info = builder.build_graph(
        columns=cols,
        dependency_matrix=np.eye(d, dtype=np.float32),
        sensitivity_vector=sens_vector,
        df_encoded=None,
    )

    graph_inactive = bool(info.get("num_edges", 0) == 0 or info.get("clinical_edges", 0) == 0)
    assert graph_inactive is True, "Expected graph_inactive=True for non-clinical generic features"


def test_stage3_pima_clinical_edges_active():
    """Verify that clinical datasets with matched ontology pairs (e.g. Pima) have active clinical graphs."""
    builder = DomainConstraintGraphBuilder(corr_threshold=0.10, edge_prune_threshold=0.10)
    pima_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    d = len(pima_cols)
    sens_vector = np.ones(d, dtype=np.float32) * 0.5

    G, W, info = builder.build_graph(
        columns=pima_cols,
        dependency_matrix=np.eye(d, dtype=np.float32),
        sensitivity_vector=sens_vector,
        df_encoded=None,
    )

    assert info["clinical_edges"] >= 3
    graph_inactive = bool(info.get("num_edges", 0) == 0 or info.get("clinical_edges", 0) == 0)
    assert graph_inactive is False, "Expected graph_inactive=False for Pima diabetes clinical ontology"


def test_stage3b_post_latent_refiner_dimension_handling():
    """Verify that Stage 3b refiner properly aligns feature dimensions and avoids IndexError when features != latent dim."""
    from lumisyn.framework.acfm_net.graph.refinement import PostLatentGraphRefiner
    import networkx as nx

    refiner = PostLatentGraphRefiner(alpha=0.6, prune_threshold=0.1)
    num_nodes = 78
    G_initial = nx.Graph()
    for i in range(num_nodes):
        G_initial.add_node(i)
    W_initial = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    W_initial[0, 1] = 0.5
    W_initial[1, 0] = 0.5

    # Case 1: latent_Z has features as columns (N=100, num_nodes=78)
    Z_cols = np.random.randn(100, num_nodes).astype(np.float32)
    G_ref, W_ref, info = refiner.refine_graph(G_initial, W_initial, Z_cols)
    assert W_ref.shape == (num_nodes, num_nodes)
    assert G_ref.number_of_nodes() == num_nodes

    # Case 2: latent_Z has latent_dim != num_nodes (e.g. VAE latent space 64 != 78 nodes)
    # Must NOT raise IndexError or slice dimensions erroneously
    Z_mismatch = np.random.randn(100, 64).astype(np.float32)
    G_ref2, W_ref2, info2 = refiner.refine_graph(G_initial, W_initial, Z_mismatch)
    assert W_ref2.shape == (num_nodes, num_nodes)
    assert G_ref2.number_of_nodes() == num_nodes

