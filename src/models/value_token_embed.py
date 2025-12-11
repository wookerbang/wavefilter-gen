from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


class ValueAwareEmbedding(nn.Module):
    """
    Wraps a base embedding to add value-aware offsets for <VAL_xxx> tokens.
    Keeps the base weight for tying with LM head.
    """

    def __init__(
        self,
        base_embedding: nn.Embedding,
        token_to_value: Mapping[int, float],
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        vocab_size, d_model = base_embedding.weight.shape
        lookup = torch.zeros(vocab_size)
        for tid, v in token_to_value.items():
            if 0 <= int(tid) < vocab_size:
                lookup[int(tid)] = float(v)
        self.register_buffer("value_lookup", lookup, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_embedding(input_ids)
        vals = self.value_lookup[input_ids]  # (B, T)
        mask = vals != 0
        if mask.any():
            val_vec = self.mlp(torch.log(vals[mask].unsqueeze(-1) + 1e-16))
            offset = torch.zeros_like(base)
            offset[mask] = val_vec
            return base + offset
        return base
