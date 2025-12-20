"""
将各步骤串联生成样本并落盘。
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from .gen_prototype import (
    compute_ideal_waveform,
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
) -> str:
    """
    串起采样 → 原型 → 离散化 → 仿真 → 序列化。
    返回写入的 jsonl 路径。
    When emit_vact_cells=True, insert <CELL> section markers into VACT tokens.
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split}.jsonl")

    rng = np.random.default_rng(seed)

    with open(jsonl_path, "w") as f:
        for i in range(num_samples):
            spec = sample_filter_spec(rng=rng)
            z0 = spec["z0"]
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
            if need_sim_for_ideal and use_ngspice:
                ideal_s21_db, ideal_s11_db = simulate_real_waveform(base_components, spec, freq_hz, use_ngspice=True)
            else:
                ideal_s21_db, ideal_s11_db = compute_ideal_waveform(base_components, spec, freq_hz)

            # Canonicalize node names so tokenization stays in-vocab.
            base_components = canonicalize_nodes(base_components, max_nodes=max_nodes)

            discrete_components = quantize_components(base_components, series="E24")
            real_s21_db, real_s11_db = simulate_real_waveform(discrete_components, spec, freq_hz, use_ngspice=use_ngspice)

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
                vact_tokens=vact_tokens,
                vactdsl_tokens=vactdsl_tokens,
                dslv2_tokens=dslv2_tokens,
                dslv2_slot_values=dslv2_slot_values,
                sfci_tokens=sfci_tokens,
                action_tokens=action_tokens,
            )

            f.write(json.dumps(_serialize_sample(sample)) + "\n")
    return jsonl_path
