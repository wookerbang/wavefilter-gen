"""
Orthogonal decomposition of positive physical values in log10-space.

We represent a value v (SI units) as:
  v = (mantissa[m_idx] * 10**exp) * 10**residual

where:
  - m_idx is a discrete E-series mantissa index (E24 by default),
  - exp is an integer decade exponent,
  - residual is a small bounded log10 offset around the nearest anchor.

This is designed for ML training stability: two small classification heads
(mantissa/decade) + a bounded residual regression head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

import torch


E24_MANTISSA: Tuple[float, ...] = (
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.7,
    3.0,
    3.3,
    3.6,
    3.9,
    4.3,
    4.7,
    5.1,
    5.6,
    6.2,
    6.8,
    7.5,
    8.2,
    9.1,
)


def _residual_scale_from_mantissas(mant: Iterable[float]) -> float:
    logs = [math.log10(float(x)) for x in mant]
    if len(logs) < 2:
        return 0.05
    gaps = [logs[i + 1] - logs[i] for i in range(len(logs) - 1)]
    # Half of the maximum adjacent gap (covers worst-case midpoint residual).
    return 0.5 * max(gaps)


DEFAULT_DECADE_MIN = -15
DEFAULT_DECADE_MAX = 4
DEFAULT_NUM_DECADES = DEFAULT_DECADE_MAX - DEFAULT_DECADE_MIN + 1
# Default residual bound in log10-space.
# We intentionally keep this slightly conservative (~0.05) to:
#  - cover E24 non-uniform gaps and boundary decades,
#  - keep residual "small" (max factor ~1.12) for manufacturability.
DEFAULT_RESIDUAL_LOG10_SCALE = max(0.05, _residual_scale_from_mantissas(E24_MANTISSA))


@dataclass(frozen=True)
class MDRConfig:
    decade_min: int = DEFAULT_DECADE_MIN
    decade_max: int = DEFAULT_DECADE_MAX
    residual_log10_scale: float = DEFAULT_RESIDUAL_LOG10_SCALE

    @property
    def num_decades(self) -> int:
        return int(self.decade_max - self.decade_min + 1)


def value_to_mdr(
    value_si: float,
    *,
    cfg: MDRConfig = MDRConfig(),
) -> tuple[int, int, float]:
    """
    Convert a positive float value (SI) to (mantissa_idx, decade_idx, residual_log10).
    """
    if not (value_si > 0.0) or not math.isfinite(value_si):
        raise ValueError(f"value_to_mdr expects positive finite value, got {value_si!r}")

    logv = math.log10(value_si)
    exp0 = int(math.floor(logv))
    best = None
    best_dist = float("inf")
    best_m = 0
    best_exp = exp0
    best_anchor_log = 0.0
    for exp in (exp0 - 1, exp0, exp0 + 1):
        if exp < cfg.decade_min or exp > cfg.decade_max:
            continue
        for mi, m in enumerate(E24_MANTISSA):
            anchor_log = math.log10(m) + exp
            dist = abs(anchor_log - logv)
            if dist < best_dist:
                best_dist = dist
                best = (mi, exp, anchor_log)
                best_m = mi
                best_exp = exp
                best_anchor_log = anchor_log
    if best is None:
        raise ValueError(f"Value {value_si} outside decade range [{cfg.decade_min},{cfg.decade_max}]")

    decade_idx = best_exp - cfg.decade_min

    residual = logv - best_anchor_log
    # Safety clamp (true residual should already be within half adjacent gap).
    residual = max(-cfg.residual_log10_scale, min(cfg.residual_log10_scale, residual))
    return int(best_m), int(decade_idx), float(residual)


def mdr_to_value(
    mantissa_idx: int,
    decade_idx: int,
    residual_log10: float = 0.0,
    *,
    cfg: MDRConfig = MDRConfig(),
    mode: Literal["standard", "precision"] = "precision",
) -> float:
    """
    Convert (mantissa_idx, decade_idx, residual_log10) back to a positive SI value.
    """
    if not (0 <= mantissa_idx < len(E24_MANTISSA)):
        raise ValueError(f"mantissa_idx out of range: {mantissa_idx}")
    if not (0 <= decade_idx < cfg.num_decades):
        raise ValueError(f"decade_idx out of range: {decade_idx}")

    exp = cfg.decade_min + int(decade_idx)
    anchor = float(E24_MANTISSA[mantissa_idx]) * (10.0**exp)
    if mode == "standard":
        return anchor
    residual_log10 = float(residual_log10)
    residual_log10 = max(-cfg.residual_log10_scale, min(cfg.residual_log10_scale, residual_log10))
    return anchor * (10.0**residual_log10)


def torch_decompose_mdr(
    values: torch.Tensor,
    *,
    cfg: MDRConfig = MDRConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized torch decomposition for training.

    Args:
      values: (B,L) float tensor in SI units.
    Returns:
      mantissa_idx: (B,L) long
      decade_idx: (B,L) long
      residual_log10: (B,L) float
      valid_mask: (B,L) bool, True where decomposition is valid and within decade range.
    """
    if values.ndim != 2:
        raise ValueError(f"Expected (B,L) values, got shape={tuple(values.shape)}")

    device = values.device
    dtype = values.dtype
    mant = torch.tensor(E24_MANTISSA, device=device, dtype=dtype)  # (24,)
    log_mant = torch.log10(mant)  # (24,)

    valid = torch.isfinite(values) & (values > 0)
    safe = torch.where(valid, values, torch.ones_like(values))
    logv = torch.log10(safe)
    exp0 = torch.floor(logv).to(torch.long)  # (B,L)

    exp_cand = torch.stack([exp0 - 1, exp0, exp0 + 1], dim=-1)  # (B,L,3)
    log_anchor = exp_cand.to(dtype).unsqueeze(-1) + log_mant.view(1, 1, 1, -1)  # (B,L,3,24)
    dist = torch.abs(log_anchor - logv.unsqueeze(-1).unsqueeze(-1))  # (B,L,3,24)

    flat = dist.reshape(values.shape[0], values.shape[1], -1)  # (B,L,72)
    best_flat = torch.argmin(flat, dim=-1)  # (B,L)
    mant_idx = torch.remainder(best_flat, len(E24_MANTISSA)).to(torch.long)
    exp_sel = torch.div(best_flat, len(E24_MANTISSA), rounding_mode="floor").to(torch.long)
    exp = torch.gather(exp_cand, dim=-1, index=exp_sel.unsqueeze(-1)).squeeze(-1)  # (B,L)

    decade_idx = exp - int(cfg.decade_min)
    in_range = (decade_idx >= 0) & (decade_idx < int(cfg.num_decades))
    valid = valid & in_range
    decade_idx = torch.clamp(decade_idx, 0, int(cfg.num_decades) - 1).to(torch.long)

    anchor_log = torch.gather(log_mant.view(1, 1, -1).expand(values.shape[0], values.shape[1], -1), 2, mant_idx.unsqueeze(-1)).squeeze(-1) + exp.to(dtype)
    residual = logv - anchor_log
    residual = torch.clamp(residual, -float(cfg.residual_log10_scale), float(cfg.residual_log10_scale))
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    return mant_idx, decade_idx, residual, valid


def torch_compose_value(
    mantissa_idx: torch.Tensor,
    decade_idx: torch.Tensor,
    residual_log10: torch.Tensor,
    *,
    cfg: MDRConfig = MDRConfig(),
    mode: Literal["standard", "precision"] = "precision",
) -> torch.Tensor:
    """
    Vectorized inverse for inference/post-processing.
    """
    if mantissa_idx.shape != decade_idx.shape or mantissa_idx.shape != residual_log10.shape:
        raise ValueError("mantissa_idx/decade_idx/residual_log10 must have identical shapes")
    device = residual_log10.device
    dtype = residual_log10.dtype
    mant = torch.tensor(E24_MANTISSA, device=device, dtype=dtype)
    m = mant[mantissa_idx.clamp(0, len(E24_MANTISSA) - 1)]
    exp = (decade_idx.to(dtype) + float(cfg.decade_min)).round()
    anchor = m * torch.pow(torch.tensor(10.0, device=device, dtype=dtype), exp)
    if mode == "standard":
        return anchor
    residual = torch.clamp(residual_log10, -float(cfg.residual_log10_scale), float(cfg.residual_log10_scale))
    return anchor * torch.pow(torch.tensor(10.0, device=device, dtype=dtype), residual)
