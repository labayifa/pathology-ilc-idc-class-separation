"""Plain (ungated) soft-attention pooling -- deliberately distinct from
mil_head.py's gated-attention MIL (Ilse et al. 2018, which multiplies a Tanh
branch by a Sigmoid gate before scoring). This head uses a single Tanh-scored
attention with no gate, so MIL's gating mechanism can be judged against an
otherwise-identical ungated alternative rather than compared only to
mean-pooling."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, attention_dim: int = 128,
                 n_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, tile_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_proj(tile_features)
        scores = self.attention(h).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=0)
        slide_embedding = (attn_weights.unsqueeze(-1) * h).sum(dim=0)
        logits = self.classifier(slide_embedding.unsqueeze(0))
        return logits, attn_weights
