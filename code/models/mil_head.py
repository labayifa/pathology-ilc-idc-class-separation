"""Gated-attention MIL pooling (Ilse, Tomczak & Welling, 2018, "Attention-based
Deep Multiple Instance Learning" -- the ABMIL mechanism CLAM also builds on).
Learns a per-tile attention weight instead of uniform mean-pooling, so the
slide-level embedding is a weighted sum over tiles."""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, attention_dim: int = 128,
                 n_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, tile_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tile_features: [n_tiles, input_dim]. Returns (logits [1, n_classes], attn_weights [n_tiles])."""
        h = self.feature_proj(tile_features)
        gated = self.attention_v(h) * self.attention_u(h)
        scores = self.attention_w(gated).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=0)
        slide_embedding = (attn_weights.unsqueeze(-1) * h).sum(dim=0)
        logits = self.classifier(slide_embedding.unsqueeze(0))
        return logits, attn_weights
