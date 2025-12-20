from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.data.circuits import (
    abcd_to_sparams,
    abcd_to_sparams_complex,
    components_to_abcd,
    components_to_sparams_nodal,
)
from src.data.dsl_codec import vactdsl_tokens_to_components
from src.data.sfci_net_codec import sfci_net_tokens_to_components
from src.data.vact_codec import vact_tokens_to_components
from src.eval.metrics import PassivityMetrics, WaveformError, passivity_metrics, waveform_error
from src.data.schema import ComponentSpec
from src.data.action_codec import action_tokens_to_components
from src.data.dsl_v2 import dslv2_tokens_to_components


ReprKind = Literal["vact", "vactdsl", "sfci", "action", "dslv2"]
SimKind = Literal["nodal", "abcd"]


def build_label_value_map(tokenizer) -> Dict[str, float]:
    from src.data import quantization

    vocab = tokenizer.get_vocab()
    mp: Dict[str, float] = {}
    for tok in vocab.keys():
        if tok.startswith("<VAL_"):
            label = tok.replace("<VAL_", "").replace(">", "")
            try:
                mp[label] = float(quantization.label_to_value(label))
            except Exception:
                continue
    return mp


def decode_components_from_token_ids(
    token_ids: Sequence[int],
    tokenizer,
    *,
    repr_kind: ReprKind,
    label_to_value: Mapping[str, float] | None = None,
    slot_values: Sequence[float] | None = None,
) -> Tuple[list, List[str]]:
    ids = list(token_ids)
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    if repr_kind == "vact":
        comps = vact_tokens_to_components(tokens, label_to_value=label_to_value, drop_non_component_tokens=True)
        return comps, tokens
    if repr_kind == "vactdsl":
        comps = vactdsl_tokens_to_components(tokens, label_to_value=label_to_value, drop_non_component_tokens=True)
        return comps, tokens
    if repr_kind == "sfci":
        comps = sfci_net_tokens_to_components(tokens, label_to_value=label_to_value)
        return comps, tokens
    if repr_kind == "dslv2":
        # Keep alignment between tokens and slot_values by filtering special tokens on both.
        if slot_values is not None:
            if len(slot_values) != len(ids):
                raise ValueError(f"slot_values must align with token_ids: {len(slot_values)} != {len(ids)}")
            full_tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            filtered_tokens: List[str] = []
            filtered_values: List[float] = []
            for tid, tok, v in zip(ids, full_tokens, slot_values):
                if int(tid) in special_ids:
                    continue
                filtered_tokens.append(tok)
                filtered_values.append(float(v))
            comps = dslv2_tokens_to_components(filtered_tokens, slot_values=filtered_values)
            return comps, filtered_tokens
        comps = dslv2_tokens_to_components(tokens, slot_values=None)
        return comps, tokens
    if repr_kind == "action":
        comps = action_tokens_to_components(tokens, label_to_value=label_to_value)
        return comps, tokens
    raise ValueError(f"Unknown repr_kind: {repr_kind}")


@dataclass(frozen=True)
class SimulationResult:
    s21_db: np.ndarray
    s_matrix: Optional[np.ndarray] = None  # (F,2,2) complex
    passivity: Optional[PassivityMetrics] = None


def simulate_s21(
    components,
    freq_hz: np.ndarray,
    *,
    z0: float = 50.0,
    sim_kind: SimKind = "nodal",
    gmin: float = 1e-12,
) -> SimulationResult:
    """
    Simulate S21 magnitude response for an LC network.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)

    if sim_kind == "nodal":
        S = components_to_sparams_nodal(components, freq_hz, z0=z0, gmin=gmin)
        s21 = S[:, 1, 0]
        s21_db = 20.0 * np.log10(np.abs(s21) + 1e-12)
        pm = passivity_metrics(S)
        return SimulationResult(s21_db=s21_db, s_matrix=S, passivity=pm)

    if sim_kind == "abcd":
        A, B, C, D = components_to_abcd(components, freq_hz, z0=z0)
        s21_db, _ = abcd_to_sparams(A, B, C, D, z0=z0)
        S11, S21, S12, S22 = abcd_to_sparams_complex(A, B, C, D, z0=z0)
        S = np.stack(
            [
                np.stack([S11, S12], axis=-1),
                np.stack([S21, S22], axis=-1),
            ],
            axis=-2,
        )
        pm = passivity_metrics(S)
        return SimulationResult(s21_db=s21_db, s_matrix=S, passivity=pm)

    raise ValueError(f"Unknown sim_kind: {sim_kind}")


@dataclass(frozen=True)
class CandidateScore:
    token_ids: List[int]
    tokens: List[str]
    num_components: int
    error: WaveformError
    passivity: Optional[PassivityMetrics]


def score_candidates(
    candidates_token_ids: Sequence[Sequence[int]],
    tokenizer,
    *,
    repr_kind: ReprKind,
    label_to_value: Mapping[str, float],
    freq_hz: np.ndarray,
    target_s21_db: np.ndarray,
    z0: float = 50.0,
    sim_kind: SimKind = "nodal",
    error_kind: Literal["mae_lin", "rmse_lin", "mae_db", "rmse_db", "maxe_lin", "maxe_db"] = "mae_lin",
    gmin: float = 1e-12,
) -> List[CandidateScore]:
    out: List[CandidateScore] = []
    for seq in candidates_token_ids:
        comps, toks = decode_components_from_token_ids(seq, tokenizer, repr_kind=repr_kind, label_to_value=label_to_value)
        sim = simulate_s21(comps, freq_hz, z0=z0, sim_kind=sim_kind, gmin=gmin)
        err = waveform_error(sim.s21_db, target_s21_db, kind=error_kind)
        out.append(
            CandidateScore(
                token_ids=[int(x) for x in seq],
                tokens=toks,
                num_components=len(comps),
                error=err,
                passivity=sim.passivity,
            )
        )
    return out


def refine_component_values_to_match_s21(
    components: Sequence[ComponentSpec],
    *,
    freq_hz: np.ndarray,
    target_s21_db: np.ndarray,
    z0: float = 50.0,
    steps: int = 30,
    lr: float = 5e-2,
    max_ratio: float = 2.0,
    gmin: float = 1e-12,
    device: str = "cpu",
) -> List[ComponentSpec]:
    """
    Few-step differentiable parameter refinement (topology fixed).

    Optimizes component values in log-space with a multiplicative clamp:
      v in [v0/max_ratio, v0*max_ratio]

    Uses nodal S-parameter simulation (2-port, ports to ground) implemented in torch.
    """
    import torch

    comps = list(components)
    if not comps:
        return []

    freq = torch.tensor(np.asarray(freq_hz, dtype=float), dtype=torch.float64, device=device)
    omega = 2.0 * np.pi * freq
    jw = 1j * omega.to(torch.complex128)
    eps = 1e-30

    # Nodes for Y-matrix (exclude gnd).
    gnd = "gnd"
    port_in = "in"
    port_out = "out"
    nodes = sorted({n for c in comps for n in (c.node1, c.node2) if n != gnd})
    if port_in not in nodes:
        nodes = [port_in] + nodes
    if port_out not in nodes:
        nodes = nodes + [port_out]
    seen = set()
    nodes = [n for n in nodes if not (n in seen or seen.add(n))]
    node_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    p_idx = [node_idx[port_in], node_idx[port_out]]
    i_idx = [k for k in range(n_nodes) if k not in p_idx]

    target_mag = torch.tensor(np.power(10.0, np.asarray(target_s21_db, dtype=float) / 20.0), dtype=torch.float64, device=device)

    init_vals = torch.tensor([float(c.value_si) for c in comps], dtype=torch.float64, device=device).clamp_min(1e-30)
    log_init = torch.log(init_vals)
    log_vals = log_init.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([log_vals], lr=float(lr))
    clamp = float(max_ratio) if max_ratio and max_ratio > 1 else None
    log_lo = log_init - (np.log(clamp) if clamp else 0.0)
    log_hi = log_init + (np.log(clamp) if clamp else 0.0)

    def _simulate_s21_mag(values: torch.Tensor) -> torch.Tensor:
        # Build Y (F,N,N)
        Y = torch.zeros((freq.shape[0], n_nodes, n_nodes), dtype=torch.complex128, device=device)
        for m, c in enumerate(comps):
            n1 = c.node1
            n2 = c.node2
            if n1 == gnd and n2 == gnd:
                continue
            v = values[m]
            if c.ctype == "C":
                y = jw * v
            else:
                y = 1.0 / (jw * v + eps)
            i = node_idx.get(n1) if n1 != gnd else None
            j = node_idx.get(n2) if n2 != gnd else None
            if i is not None:
                Y[:, i, i] = Y[:, i, i] + y
            if j is not None:
                Y[:, j, j] = Y[:, j, j] + y
            if i is not None and j is not None:
                Y[:, i, j] = Y[:, i, j] - y
                Y[:, j, i] = Y[:, j, i] - y
        if gmin and gmin > 0:
            Y[:, range(n_nodes), range(n_nodes)] = Y[:, range(n_nodes), range(n_nodes)] + complex(float(gmin), 0.0)

        Ypp = Y[:, p_idx, :][:, :, p_idx]
        if not i_idx:
            Y_port = Ypp
        else:
            Ypi = Y[:, p_idx, :][:, :, i_idx]
            Yii = Y[:, i_idx, :][:, :, i_idx]
            Yip = Y[:, i_idx, :][:, :, p_idx]
            sol = torch.linalg.solve(Yii, Yip)
            Y_port = Ypp - torch.matmul(Ypi, sol)

        I2 = torch.eye(2, dtype=torch.complex128, device=device).unsqueeze(0).expand(Y_port.shape[0], -1, -1)
        A = I2 - complex(float(z0), 0.0) * Y_port
        Bm = I2 + complex(float(z0), 0.0) * Y_port
        S = torch.linalg.solve(Bm, A)
        s21 = S[:, 1, 0]
        return torch.abs(s21).to(torch.float64)

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        vals = torch.exp(log_vals)
        pred_mag = _simulate_s21_mag(vals)
        loss = torch.mean(torch.square(pred_mag - target_mag))
        loss.backward()
        opt.step()
        if clamp is not None:
            with torch.no_grad():
                log_vals.clamp_(log_lo, log_hi)

    refined = torch.exp(log_vals.detach()).cpu().numpy().astype(float)
    out: List[ComponentSpec] = []
    for c, v in zip(comps, refined):
        out.append(
            ComponentSpec(
                ctype=c.ctype,
                role=c.role,
                value_si=float(v),
                std_label=c.std_label,
                node1=c.node1,
                node2=c.node2,
            )
        )
    return out
