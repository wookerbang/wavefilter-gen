from __future__ import annotations

import torch
import torch.nn as nn


class SpecEncoder(nn.Module):
    """
    Encodes filter spec (type + log10(fc)) into a single token embedding.
    """

    def __init__(
        self,
        d_model: int = 512,
        type_vocab_size: int = 2,
        hidden_dim: int = 128,
        use_learnable_token: bool = True,
    ):
        super().__init__()
        self.type_emb = nn.Embedding(type_vocab_size, d_model)
        self.fc_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.use_learnable_token = use_learnable_token
        self.base_token = nn.Parameter(torch.zeros(d_model)) if use_learnable_token else None

    def forward(self, filter_type_ids: torch.Tensor, log10_fc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            filter_type_ids: (B,) long tensor (0=lowpass,1=bandpass)
            log10_fc: (B,) float tensor of log10 cutoff freq
        Returns:
            (B, d_model) spec token
        """
        t = self.type_emb(filter_type_ids)
        fc = self.fc_mlp(log10_fc.unsqueeze(-1))
        base = self.base_token if self.use_learnable_token else 0.0
        return t + fc + base
