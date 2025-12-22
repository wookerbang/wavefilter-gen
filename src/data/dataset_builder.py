"""
将各步骤串联生成样本并落盘。
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch

from .gen_prototype import (
    sample_filter_spec,
    synthesize_filter,
)
from .quantization import quantize_components
from .schema import ComponentSpec, FilterSample
from .vact_codec import components_to_vact_tokens
from .sfci_net_codec import components_to_sfci_net_tokens
from .spice_runner import simulate_real_waveform
from .dsl_codec import components_to_vactdsl_tokens
from .node_canonicalizer import canonicalize_nodes
from .action_codec import components_to_action_tokens
from .dsl_v2 import components_to_dslv2_tokens
from src.physics import FastTrackEngine


def _serialize_components(comps: List[ComponentSpec]) -> List[dict]:
    return [c.to_dict() for c in comps]


def _serialize_sample(sample: FilterSample) -> dict:
    data = sample.to_metadata_dict()
    data.update(
        {
            "freq_hz": np.asarray(sample.freqs_hz, dtype=float).tolist() if sample.freqs_hz is not None else None,
            "ideal_s21_db": np.asarray(sample.w_ideal_S21_db, dtype=float).tolist() if sample.w_ideal_S21_db is not None else None,
            "ideal_s11_db": np.asarray(sample.w_ideal_S11_db, dtype=float).tolist() if sample.w_ideal_S11_db is not None else None,
            "real_s21_db": np.asarray(sample.w_real_S21_db, dtype=float).tolist() if sample.w_real_S21_db is not None else None,
            "real_s11_db": np.asarray(sample.w_real_S11_db, dtype=float).tolist() if sample.w_real_S11_db is not None else None,
            "mask_min_db": np.asarray(sample.mask_min_db, dtype=float).tolist() if sample.mask_min_db is not None else None,
            "mask_max_db": np.asarray(sample.mask_max_db, dtype=float).tolist() if sample.mask_max_db is not None else None,
            "ideal_components": _serialize_components(sample.ideal_components or []),
            "discrete_components": _serialize_components(sample.discrete_components or []),
            "vact_tokens": sample.vact_tokens or [],
            "vactdsl_tokens": sample.vactdsl_tokens or [],
            "dslv2_tokens": sample.dslv2_tokens or [],
            "dslv2_slot_values": sample.dslv2_slot_values or [],
            "sfci_tokens": sample.sfci_tokens or [],
            "action_tokens": sample.action_tokens or [],
        }
    )
    return data


def build_dataset(
    num_samples: int,
    output_dir: str,
    split: str = "train",
    use_ngspice: bool = False,
    seed: int = 42,
    emit_vact_cells: bool = True,
    emit_vactdsl: bool = True,
    emit_actions: bool = True,
    emit_dslv2: bool = True,
    max_nodes: int = 32,
    q_L: float | None = 50.0,
    q_C: float | None = 50.0,
    tol_frac: float = 0.05,
    q_model: str = "freq_dependent",
) -> str:
    """
    串起采样 → 原型 → 离散化 → 仿真 → 序列化。
    返回写入的 jsonl 路径。
    When emit_vact_cells=True, insert <CELL> section markers into VACT tokens.
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split}.jsonl")

    rng = np.random.default_rng(seed)
    fast_engine: FastTrackEngine | None = None

    def _apply_tolerance(comps: List[ComponentSpec]) -> List[ComponentSpec]:
        t = float(tol_frac or 0.0)
        if t <= 0:
            return list(comps)
        out: List[ComponentSpec] = []
        for c in comps:
            scale = float(rng.uniform(1.0 - t, 1.0 + t))
            out.append(
                ComponentSpec(
                    ctype=c.ctype,
                    role=c.role,
                    value_si=float(c.value_si) * scale,
                    std_label=c.std_label,
                    node1=c.node1,
                    node2=c.node2,
                )
            )
        return out

    def _build_spec_masks(spec: dict, freq_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        f = np.asarray(freq_hz, dtype=float)
        mask_min = np.full_like(f, np.nan, dtype=float)
        mask_max = np.full_like(f, np.nan, dtype=float)
        fc = float(spec.get("fc_hz", 0.0))
        ripple_db = float(spec.get("ripple_db", 0.5))
        passband_min_db = -abs(ripple_db)
        stopband_max_db = float(spec.get("stopband_max_db", -40.0))
        ftype = spec.get("filter_type", "lowpass")
        if ftype == "lowpass":
            pass_mask = f <= fc
            stop_mask = f >= 2.0 * fc
        elif ftype == "highpass":
            pass_mask = f >= fc
            stop_mask = f <= 0.5 * fc
        elif ftype == "bandpass":
            bw = float(spec.get("bw_frac") or 0.2)
            pass_lo = fc * (1.0 - 0.5 * bw)
            pass_hi = fc * (1.0 + 0.5 * bw)
            stop_lo = fc * (1.0 - 1.5 * bw)
            stop_hi = fc * (1.0 + 1.5 * bw)
            pass_mask = (f >= pass_lo) & (f <= pass_hi)
            stop_mask = (f <= stop_lo) | (f >= stop_hi)
        elif ftype == "bandstop":
            bw = float(spec.get("bw_frac") or 0.2)
            stop_lo = fc * (1.0 - 0.5 * bw)
            stop_hi = fc * (1.0 + 0.5 * bw)
            pass_lo = fc * (1.0 - 1.5 * bw)
            pass_hi = fc * (1.0 + 1.5 * bw)
            pass_mask = (f <= pass_lo) | (f >= pass_hi)
            stop_mask = (f >= stop_lo) & (f <= stop_hi)
        else:
            pass_mask = f <= fc
            stop_mask = f >= 2.0 * fc
        mask_min[pass_mask] = passband_min_db
        mask_max[stop_mask] = stopband_max_db
        return mask_min, mask_max, passband_min_db, stopband_max_db

    with open(jsonl_path, "w") as f:
        for i in range(num_samples):
            spec = sample_filter_spec(rng=rng)
            z0 = spec["z0"]
            if fast_engine is None or float(z0) != float(fast_engine.z0):
                fast_engine = FastTrackEngine(z0=float(z0), device="cpu", dtype=torch.float64)
            base_components = synthesize_filter(spec)

            # 频率轴：依据 filter_type 选取覆盖区间
            fc = spec["fc_hz"]
            f_min = fc / 10.0
            f_max = fc * 10.0
            if spec.get("filter_type") == "bandpass":
                fbw = float(spec.get("bw_frac") or 0.2)
                f_min = fc * (1 - fbw) * 0.8
                f_max = fc * (1 + fbw) * 1.2
            freq_hz = np.logspace(np.log10(f_min), np.log10(f_max), 256)
            mask_min_db, mask_max_db, passband_min_db, stopband_max_db = _build_spec_masks(spec, freq_hz)

            # dual-band：并联第二个 bandpass 子滤波器
            if spec.get("scenario") == "dualband" and spec.get("fc2_hz"):
                spec2 = dict(spec)
                spec2["fc_hz"] = spec["fc2_hz"]
                spec2["filter_type"] = "bandpass"
                spec2["bw_frac"] = spec.get("bw_frac")
                comps2 = synthesize_filter(spec2)
                # 重命名内部节点避免冲突，保留 in/out/gnd
                renamed = []
                for c in comps2:
                    def _ren(node: str) -> str:
                        if node in ("in", "out", "gnd"):
                            return node
                        return f"{node}_b2"
                    renamed.append(
                        ComponentSpec(
                            ctype=c.ctype,
                            role=c.role,
                            value_si=c.value_si,
                            std_label=c.std_label,
                            node1=_ren(c.node1),
                            node2=_ren(c.node2),
                        )
                    )
                base_components = base_components + renamed

            # 可选 notch：在某个主节点挂一个串联 LC 到地
            if spec.get("scenario") == "notch":
                # 选一个非 gnd 的节点作挂载
                nodes = []
                for c in base_components:
                    nodes.extend([c.node1, c.node2])
                nodes = [n for n in set(nodes) if n not in ("gnd", "out")]
                anchor = nodes[0] if nodes else "in"
                f_notch = fc * rng.uniform(1.2, 1.8)
                Cn = 1e-12 * rng.uniform(0.5, 5.0)
                Ln = 1.0 / ((2 * np.pi * f_notch) ** 2 * Cn)
                mid_node = f"{anchor}_notch"
                base_components.append(ComponentSpec("L", "series", Ln, None, anchor, mid_node))
                base_components.append(ComponentSpec("C", "series", Cn, None, mid_node, "gnd"))

            # 非纯 ladder（notch/dualband/BP）优先用仿真获取 ideal，以避免 ABCD 近似误差
            need_sim_for_ideal = spec.get("scenario") in ("notch", "dualband") or spec.get("filter_type") != "lowpass"

            # Canonicalize node names so tokenization stays in-vocab.
            base_components = canonicalize_nodes(base_components, max_nodes=max_nodes)

            # Output label: nominal standard parts (no tolerance, no loss).
            discrete_components = quantize_components(base_components, series="E24")
            ref_freq_hz = float(spec.get("fc_hz") or np.sqrt(float(np.min(freq_hz)) * float(np.max(freq_hz))))
            if need_sim_for_ideal and use_ngspice:
                ideal_s21_db, ideal_s11_db = simulate_real_waveform(
                    discrete_components,
                    spec,
                    freq_hz,
                    use_ngspice=True,
                    q_L=None,
                    q_C=None,
                    ref_freq_hz=ref_freq_hz,
                    q_model=q_model,
                )
            else:
                ideal_s21_db, ideal_s11_db = fast_engine.simulate_sparams_db(
                    discrete_components,
                    freq_hz,
                    q_L=None,
                    q_C=None,
                    q_model="freq_dependent",
                )

            # Input waveform: tolerance-perturbed + finite-Q loss model.
            real_components = _apply_tolerance(discrete_components)
            use_spice_real = bool(use_ngspice) and ((q_L is None and q_C is None) or str(q_model) == "fixed_ref")
            if use_spice_real:
                real_s21_db, real_s11_db = simulate_real_waveform(
                    real_components,
                    spec,
                    freq_hz,
                    use_ngspice=True,
                    q_L=q_L,
                    q_C=q_C,
                    ref_freq_hz=ref_freq_hz,
                    q_model=q_model,
                )
            else:
                real_s21_db, real_s11_db = fast_engine.simulate_sparams_db(
                    real_components,
                    freq_hz,
                    q_L=q_L,
                    q_C=q_C,
                    q_model=str(q_model),
                    ref_freq_hz=ref_freq_hz if str(q_model) == "fixed_ref" else None,
                )

            # --- Sanity checks ---
            if real_s21_db is None or np.any(np.isnan(real_s21_db)):
                print(f"Sample {i}: Simulation failed or NaN.")
                continue
            is_broken = False
            ftype = spec.get("filter_type", "lowpass")
            if ftype == "lowpass":
                if np.mean(real_s21_db[:10]) < -10.0:
                    is_broken = True
            elif ftype == "bandpass":
                mid = len(real_s21_db) // 2
                window = real_s21_db[max(0, mid - 5) : mid + 5]
                if np.mean(window) < -10.0:
                    is_broken = True
            if is_broken:
                print(f"Sample {i}: Circuit broken (High insertion loss).")
                continue

            vact_tokens = [f"<ORDER_{spec['order']}>", "<SEP>"] + components_to_vact_tokens(
                discrete_components,
                emit_cell_tokens=emit_vact_cells,
                normalize_node_order=True,
            )
            vactdsl_tokens = None
            if emit_vactdsl:
                vactdsl_tokens = [f"<ORDER_{spec['order']}>", "<SEP>"] + components_to_vactdsl_tokens(
                    discrete_components,
                    z0=float(z0),
                    include_ports=True,
                    emit_cells=True,
                )
            dslv2_tokens = None
            dslv2_slot_values = None
            if emit_dslv2:
                dslv2_tokens, dslv2_slot_values = components_to_dslv2_tokens(discrete_components)
            sfci_tokens = components_to_sfci_net_tokens(discrete_components)
            action_tokens = components_to_action_tokens(discrete_components) if emit_actions else None
            sample = FilterSample(
                spec_id=i,
                circuit_id=i,
                sample_id=f"{split}_{i}",
                filter_type=spec["filter_type"],
                prototype_type=spec["prototype_type"],
                order=spec["order"],
                ripple_db=spec["ripple_db"],
                fc_hz=spec["fc_hz"],
                variant="quantized",
                z0=z0,
                num_L=sum(1 for c in base_components if c.ctype == "L"),
                num_C=sum(1 for c in base_components if c.ctype == "C"),
                scenario=spec.get("scenario"),
                ideal_components=base_components,
                discrete_components=discrete_components,
                json_components=None,
                freqs_hz=freq_hz,
                w_ideal_S21_db=ideal_s21_db,
                w_real_S21_db=real_s21_db,
                w_ideal_S11_db=ideal_s11_db,
                w_real_S11_db=real_s11_db,
                passband_min_db=passband_min_db,
                stopband_max_db=stopband_max_db,
                mask_min_db=mask_min_db,
                mask_max_db=mask_max_db,
                vact_tokens=vact_tokens,
                vactdsl_tokens=vactdsl_tokens,
                dslv2_tokens=dslv2_tokens,
                dslv2_slot_values=dslv2_slot_values,
                sfci_tokens=sfci_tokens,
                action_tokens=action_tokens,
            )

            f.write(json.dumps(_serialize_sample(sample)) + "\n")
    return jsonl_path
