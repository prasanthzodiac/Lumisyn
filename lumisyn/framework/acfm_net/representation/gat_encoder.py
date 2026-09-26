import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionEncoder(nn.Module):
    """
    4.1 Graph Attention Encoder (GAT) - HCARLN Stage 4
    Edge-type-aware Multi-head Graph Attention Network.
    4 edge types -> 4 attention heads (K=8 heads, dim 64 each).
    Residual connections between GAT layers.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, num_heads: int = 8, num_edge_types: int = 4):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(in_features, hidden_dim)
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.v_proj = nn.Linear(in_features, hidden_dim)

        # Edge-type specific attention bias
        self.edge_bias = nn.Embedding(num_edge_types + 1, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.residual_proj = nn.Linear(in_features, hidden_dim) if in_features != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, edge_types: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, num_nodes) or (batch_size, num_nodes, in_features)
        adj_matrix: (num_nodes, num_nodes)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, num_nodes, 1)

        B, N, C = x.shape
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)

        # Apply graph adjacency mask
        adj_mask = (adj_matrix == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        scores = scores.masked_fill(adj_mask, -1e9)

        # Apply edge-type bias if provided
        if edge_types is not None:
            eb = self.edge_bias(edge_types.long()).permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
            scores = scores + eb

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        out = self.out_proj(out) + self.residual_proj(x)
        return F.elu(out)
