from __future__ import annotations

import math

import torch
import torch.nn as nn


class ParallelMultiScaleWaveformEncoder(nn.Module):
    """
    真正的多尺度并行编码器：小核支路捕捉纹波，大核支路捕捉趋势。
    输入默认 (B, 1, 256)，输出约 (B, 64, d_model)。
    """

    def __init__(self, d_model: int = 512, in_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        # 高频支路：小卷积核
        self.branch_local = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 低频支路：大卷积核
        self.branch_global = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
        )
        # 融合
        self.fusion = nn.Linear(128 + 128, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _positional_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(length, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) waveform tensor
        Returns:
            (B, L_out, d_model) encoded sequence
        """
        local_feat = self.branch_local(x)   # (B, 128, L_out)
        global_feat = self.branch_global(x) # (B, 128, L_out)

        local_feat = local_feat.transpose(1, 2)
        global_feat = global_feat.transpose(1, 2)
        concat = torch.cat([local_feat, global_feat], dim=-1)  # (B, L_out, 256)

        out = self.fusion(concat)
        pe = self._positional_encoding(out.size(1), out.size(2), out.device)
        return self.dropout(out + pe.unsqueeze(0))


# 向后兼容原类名
MultiScaleWaveformEncoder = ParallelMultiScaleWaveformEncoder
