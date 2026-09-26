import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttention(nn.Module):
    """
    4.2 Gated Cross-Attention (HCARLN Stage 4) - ★ v5 REDESIGNED ★
    Prevents graph embeddings from dominating feature representations when graph is noisy.
    Q = Feature representation
    K, V = Graph node embeddings (from GAT)
    Gate: g = σ(W_g [Q; K_mean])
    Output: Ã = g ⊙ softmax(QKᵀ / √d) V
    """

    def __init__(self, feature_dim: int, graph_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.q_linear = nn.Linear(feature_dim, hidden_dim)
        self.k_linear = nn.Linear(graph_dim, hidden_dim)
        self.v_linear = nn.Linear(graph_dim, hidden_dim)

        # Gating parameter generator W_g
        self.gate_fc = nn.Linear(feature_dim + graph_dim, 1)
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features: torch.Tensor, graph_nodes: torch.Tensor) -> torch.Tensor:
        """
        features: (B, feature_dim) or (B, N, feature_dim)
        graph_nodes: (B, N_g, graph_dim)
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)  # (B, 1, feature_dim)
        if graph_nodes.dim() == 2:
            graph_nodes = graph_nodes.unsqueeze(0).repeat(features.shape[0], 1, 1)

        B, N, D_f = features.shape
        _, N_g, D_g = graph_nodes.shape

        Q = self.q_linear(features)       # (B, N, hidden_dim)
        K = self.k_linear(graph_nodes)    # (B, N_g, hidden_dim)
        V = self.v_linear(graph_nodes)    # (B, N_g, hidden_dim)

        # Compute cross-attention weights
        scores = torch.matmul(Q, K.transpose(1, 2)) / (K.shape[-1] ** 0.5)  # (B, N, N_g)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, N, hidden_dim)

        # Calculate dynamic gate g = σ(W_g [Q; K_mean])
        k_mean = graph_nodes.mean(dim=1, keepdim=True).repeat(1, N, 1)  # (B, N, D_g)
        gate_input = torch.cat([features, k_mean], dim=-1)               # (B, N, D_f + D_g)
        gate = torch.sigmoid(self.gate_fc(gate_input))                   # (B, N, 1)

        # Gated fusion output: Ã = g ⊙ Context + (1 - g) ⊙ Features
        gated_context = self.out_proj(context)
        out = gate * gated_context + (1.0 - gate) * features
        return out.squeeze(1) if out.shape[1] == 1 else out
