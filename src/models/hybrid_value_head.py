from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HybridValuePred:
    mantissa_logits: torch.Tensor  # (B,L,M)
    decade_logits: torch.Tensor  # (B,L,D)
    residual_log10: torch.Tensor  # (B,L)


class HybridValueHead(nn.Module):
    """
    Three-branch numeric head:
      - mantissa (E-series) classification
      - decade exponent classification
      - bounded residual regression in log10-space
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_mantissa: int = 24,
        num_decade: int = 20,
        residual_log10_scale: float = 0.05,
        hidden_mult: int = 1,
        num_slot_types: int = 0,
    ):
        super().__init__()
        h = int(d_model * max(1, hidden_mult))
        self.mantissa_head = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, num_mantissa))
        self.decade_head = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, num_decade))
        self.residual_head = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, 1))
        self.slot_type_emb = nn.Embedding(num_slot_types, d_model) if num_slot_types > 0 else None
        self.residual_log10_scale = float(residual_log10_scale)
        self.num_slot_types = num_slot_types

    def forward(self, hidden: torch.Tensor, slot_type_ids: torch.Tensor | None = None) -> HybridValuePred:
        if self.slot_type_emb is not None and slot_type_ids is not None:
            hidden = hidden + self.slot_type_emb(slot_type_ids.clamp(0, self.num_slot_types - 1))
        mant = self.mantissa_head(hidden)
        dec = self.decade_head(hidden)
        res = self.residual_head(hidden).squeeze(-1)
        res = torch.tanh(res) * self.residual_log10_scale
        return HybridValuePred(mantissa_logits=mant, decade_logits=dec, residual_log10=res)
