from typing import Tuple, Dict, Any
import numpy as np
import networkx as nx

class PostLatentGraphRefiner:
    """
    STAGE 3b: Graph Refinement (POST-LATENT)
    Solves the v4 chicken-and-egg problem by executing AFTER Stage 4 representation learning.
    Refines edge weights: W_ij = α * W₀_ij + (1-α) * cos(z_i, z_j)
    Prunes weak edges (W_ij < τ) and FREEZES the graph topology G for remaining stages.
    """

    def __init__(self, alpha: float = 0.60, prune_threshold: float = 0.10):
        self.alpha = alpha
        self.prune_threshold = prune_threshold

    def refine_graph(
        self,
        G_initial: nx.Graph,
        W_initial: np.ndarray,
        latent_Z: np.ndarray
    ) -> Tuple[nx.Graph, np.ndarray, Dict[str, Any]]:
        num_nodes = W_initial.shape[0]
        
        # Calculate latent feature-feature representation vectors by computing cosine similarity across samples
        # latent_Z shape: (batch_size, num_nodes) or (num_nodes, latent_dim_per_feature)
        # latent_Z=None: no data-dependent refinement (strict DP release path).
        if latent_Z is not None and latent_Z.ndim == 2:
            if latent_Z.shape[1] == num_nodes:
                # Features are columns (N, num_nodes) -> transpose to (num_nodes, N)
                feature_latents = latent_Z.T
                norm = np.linalg.norm(feature_latents, axis=1, keepdims=True) + 1e-8
                normalized_latents = feature_latents / norm
                similarity_matrix = np.abs(np.dot(normalized_latents, normalized_latents.T))
            elif latent_Z.shape[0] == num_nodes:
                # Features are rows (num_nodes, dim)
                norm = np.linalg.norm(latent_Z, axis=1, keepdims=True) + 1e-8
                normalized_latents = latent_Z / norm
                similarity_matrix = np.abs(np.dot(normalized_latents, normalized_latents.T))
            else:
                # Latent space dimension does not match graph node count (e.g. compressed VAE bottle-neck)
                # Fall back to initial graph topology without slicing arbitrary latent channels
                similarity_matrix = W_initial.copy()
        else:
            similarity_matrix = W_initial.copy()

        # Refine weights: W_ij = α * W₀_ij + (1-α) * cos(z_i, z_j)
        W_refined = self.alpha * W_initial + (1.0 - self.alpha) * similarity_matrix
        np.fill_diagonal(W_refined, 0.0)

        # Build Refined Graph G
        G_refined = nx.Graph()
        for i in range(num_nodes):
            G_refined.add_node(i)

        edges_pruned = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                w = float(W_refined[i, j])
                if w >= self.prune_threshold:
                    G_refined.add_edge(i, j, weight=w)
                else:
                    edges_pruned += 1

        refinement_info = {
            "initial_edges": G_initial.number_of_edges(),
            "refined_edges": G_refined.number_of_edges(),
            "pruned_edges": edges_pruned,
            "alpha": self.alpha,
            "topology_status": "FROZEN"
        }

        return G_refined, W_refined, refinement_info
