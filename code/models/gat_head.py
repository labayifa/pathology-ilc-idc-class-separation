"""Graph Attention Network over each slide's tile spatial k-NN graph (per the
plan's expanded classifier scope): tiles are nodes (their foundation-model
features), edges come from a k-NN graph on tile (x,y) level-0 coordinates
(already saved by tile_and_extract.py), multi-head attention follows
Velickovic et al. 2018 (GAT).

Implemented as dense masked attention over an [n_tiles, n_tiles] score matrix
rather than with torch_geometric: n_tiles is capped at 1000 (plan's own cap),
so the dense form is cheap (<=4MB for the mask) and avoids pinning a
torch_geometric wheel to whatever CUDA version the eventual GPU VM has --
this runs identically on CPU/MPS/CUDA with no extra install risk.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.mil_head import GatedAttentionMIL


def knn_adjacency(coords: torch.Tensor, k: int) -> torch.Tensor:
    """coords: [n, 2]. Returns a symmetric boolean adjacency [n, n] (self-edges excluded)."""
    n = coords.shape[0]
    if n <= 1:
        return torch.zeros((n, n), dtype=torch.bool, device=coords.device)
    dists = torch.cdist(coords, coords)
    k_eff = min(k + 1, n)  # +1 since the nearest point to itself (distance 0) is excluded below
    _, idx = torch.topk(dists, k_eff, largest=False)
    adj = torch.zeros((n, n), dtype=torch.bool, device=coords.device)
    rows = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k_eff)
    adj[rows.reshape(-1), idx.reshape(-1)] = True
    adj.fill_diagonal_(False)
    return adj | adj.T


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.proj = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [n, in_dim], adj: [n, n] bool (adj[i,j] True if j is a neighbor of i)."""
        n = x.shape[0]
        h = self.proj(x).view(n, self.n_heads, self.out_dim)   # [n, heads, out_dim]
        src_score = (h * self.attn_src).sum(-1)                 # [n, heads] -- contribution as a neighbor
        dst_score = (h * self.attn_dst).sum(-1)                 # [n, heads] -- contribution as the attending node
        # e[i, j, h] = dst_score[i, h] + src_score[j, h]  (attention of i toward neighbor j)
        e = self.leaky_relu(dst_score.unsqueeze(1) + src_score.unsqueeze(0))
        e = e.masked_fill(~adj.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # isolated node (no neighbors): zero contribution, not NaN
        alpha = self.dropout(alpha)
        out = torch.einsum("ijh,jho->iho", alpha, h)            # [n, heads, out_dim]
        return out.reshape(n, self.n_heads * self.out_dim) if self.concat else out.mean(dim=1)


class GATBackbone(nn.Module):
    """The graph-attention tile-refinement stage on its own, with no pooling or
    classifier -- composable with different final heads (mean-pool+LR/SVM/MLP,
    gated-attention MIL, or the simple attention-pool+linear head below)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2,
                 k_neighbors: int = 8, dropout: float = 0.1):
        super().__init__()
        self.k = k_neighbors
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layers = []
        in_dim = hidden_dim
        for i in range(n_layers):
            is_last = i == n_layers - 1
            layers.append(GATLayer(in_dim, hidden_dim, n_heads=n_heads, dropout=dropout, concat=not is_last))
            in_dim = hidden_dim * n_heads if not is_last else hidden_dim
        self.gat_layers = nn.ModuleList(layers)
        self.elu = nn.ELU()
        self.out_dim = in_dim

    def forward(self, tile_features: torch.Tensor, tile_coords: torch.Tensor) -> torch.Tensor:
        """Returns refined per-tile embeddings [n_tiles, self.out_dim]."""
        adj = knn_adjacency(tile_coords, self.k)
        x = self.elu(self.input_proj(tile_features))
        for layer in self.gat_layers:
            x = self.elu(layer(x, adj))
        return x


class GATClassifier(nn.Module):
    """Standalone GAT: backbone + a simple learned attention-pool + linear head."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2,
                 k_neighbors: int = 8, dropout: float = 0.1, n_classes: int = 2):
        super().__init__()
        self.backbone = GATBackbone(input_dim, hidden_dim, n_heads, n_layers, k_neighbors, dropout)
        out_dim = self.backbone.out_dim
        self.pool_attn = nn.Sequential(nn.Linear(out_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward(self, tile_features: torch.Tensor, tile_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(tile_features, tile_coords)
        attn_scores = self.pool_attn(x).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        slide_embedding = (attn_weights.unsqueeze(-1) * x).sum(dim=0)
        logits = self.classifier(slide_embedding.unsqueeze(0))
        return logits, attn_weights


class GATThenMLP(nn.Module):
    """GAT-refined tiles, mean-pooled, through a small MLP classifier. Also
    used as the differentiable proxy that trains the GAT backbone for the
    gat_lr / gat_svm configs (see train_evaluate.py): once trained, the
    backbone's mean-pooled output is handed to sklearn instead of self.head."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2,
                 k_neighbors: int = 8, dropout: float = 0.1, n_classes: int = 2):
        super().__init__()
        self.backbone = GATBackbone(input_dim, hidden_dim, n_heads, n_layers, k_neighbors, dropout)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def pooled_embedding(self, tile_features: torch.Tensor, tile_coords: torch.Tensor) -> torch.Tensor:
        return self.backbone(tile_features, tile_coords).mean(dim=0)

    def forward(self, tile_features: torch.Tensor, tile_coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled_embedding(tile_features, tile_coords).unsqueeze(0))


class GATThenMIL(nn.Module):
    """GAT-refined tiles through gated-attention MIL pooling (Ilse et al. 2018)
    instead of mean-pooling or the simple attention-pool above -- lets the
    final aggregation learn per-tile importance on top of the graph-refined
    features, rather than treating every refined tile equally."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2,
                 k_neighbors: int = 8, dropout: float = 0.1, attention_dim: int = 128, n_classes: int = 2):
        super().__init__()
        self.backbone = GATBackbone(input_dim, hidden_dim, n_heads, n_layers, k_neighbors, dropout)
        self.mil = GatedAttentionMIL(self.backbone.out_dim, hidden_dim, attention_dim, n_classes, dropout)

    def forward(self, tile_features: torch.Tensor, tile_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        refined = self.backbone(tile_features, tile_coords)
        return self.mil(refined)
