"""
调用 ngspice 做 AC 仿真，获取真实波形。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Literal, Tuple

import numpy as np

from .circuits import Circuit
from .schema import ComponentSpec


def run_ac_analysis_with_ngspice(
    circuit: Circuit,
    freq_hz: np.ndarray,
    z0: float = 50.0,
    *,
    q_L: float | None = None,
    q_C: float | None = None,
    ref_freq_hz: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 ngspice 进行 AC 分析，返回 S21/S11 in dB。
    如果 ngspice 不可用，则抛出 RuntimeError。
    """
    f_start = float(np.min(freq_hz))
    f_stop = float(np.max(freq_hz))
    n_points = len(freq_hz)

    with tempfile.TemporaryDirectory() as tmpdir:
        net_path = os.path.join(tmpdir, "circuit.sp")
        log_path = os.path.join(tmpdir, "ngspice.log")
        csv_path = os.path.join(tmpdir, "ac.csv")
        if ref_freq_hz is None:
            ref_freq_hz = float(np.sqrt(f_start * f_stop))
        netlist = circuit.to_spice_netlist(q_L=q_L, q_C=q_C, ref_freq_hz=ref_freq_hz)
        lines = [netlist]
        # 使用对数扫描与上游 logspace 频网更匹配
        decades = max(1, np.ceil(np.log10(f_stop / f_start)))
        points_per_dec = max(1, int(np.ceil(n_points / decades)))
        if points_per_dec <= 0:
            raise RuntimeError("Invalid points_per_dec computed for AC analysis.")
        lines.append(f".ac dec {points_per_dec} {f_start} {f_stop}")
        lines.append(".control")
        lines.append("set filetype=ascii")
        lines.append("run")
        lines.append('wrdata "ac.csv" frequency v(in) v(out)')
        lines.append("quit")
        lines.append(".endc")
        lines.append(".end")
        with open(net_path, "w") as f:
            f.write("\n".join(lines))

        cmd = ["ngspice", "-b", "-o", log_path, net_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir)
        except FileNotFoundError as exc:
            raise RuntimeError("ngspice not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            log_preview = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as lf:
                        log_preview = "".join(lf.readlines()[:50])
                except Exception:
                    log_preview = ""
            raise RuntimeError(f"ngspice failed, see log at {log_path}\n{log_preview}") from exc

        if not os.path.exists(csv_path):
            raise RuntimeError("ngspice did not produce ac.csv")

        try:
            data = np.genfromtxt(csv_path, delimiter=None, comments="*;", dtype=float)
        except Exception as exc:
            log_preview = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as lf:
                        log_preview = "".join(lf.readlines()[:50])
                except Exception:
                    log_preview = ""
            raise RuntimeError(f"Failed to read ac.csv: {exc}\n{log_preview}") from exc

        # 若 genfromtxt 读到空数组，尝试过滤非数字行重读
        if data.size == 0:
            numeric_lines = []
            with open(csv_path, "r") as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    if line[0].isdigit() or line[0] in "+-.":
                        numeric_lines.append(line)
            if numeric_lines:
                data = np.genfromtxt(numeric_lines, delimiter=None, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise RuntimeError("Unexpected ac.csv format from ngspice")

    freq_sim = data[:, 0]
    v_in = data[:, 1] + 1j * data[:, 2]
    v_out = data[:, 3] + 1j * data[:, 4]

    H = v_out / (v_in + 1e-18)
    # 若频率网格与请求不一致，插值到目标频轴
    if len(freq_sim) != len(freq_hz) or not np.allclose(freq_sim, freq_hz):
        # 确保单调递增
        order = np.argsort(freq_sim)
        freq_sim_sorted = freq_sim[order]
        H_sorted = H[order]
        H = np.interp(freq_hz, freq_sim_sorted, H_sorted.real) + 1j * np.interp(freq_hz, freq_sim_sorted, H_sorted.imag)
    s21_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    s11_db = np.zeros_like(s21_db)
    return s21_db, s11_db


def simulate_real_waveform(
    components: list[ComponentSpec],
    spec: dict,
    freq_hz: np.ndarray,
    use_ngspice: bool = True,
    *,
    q_L: float | None = None,
    q_C: float | None = None,
    ref_freq_hz: float | None = None,
    q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 Circuit -> 仿真得到 real S21/S11。
    ngspice 不可用时，退化为 Fast Track 物理引擎计算。
    q_model:
      - "freq_dependent": keep Q constant over band (R/G varies with frequency).
      - "fixed_ref": use fixed R/G at ref_freq_hz (matches SPICE netlist).
    """
    z0 = float(spec["z0"])
    circuit = Circuit(components, z0=z0, in_port=("in", "gnd"), out_port=("out", "gnd"))
    if use_ngspice:
        if (q_L is not None or q_C is not None) and q_model == "freq_dependent":
            use_ngspice = False
        else:
            try:
                if ref_freq_hz is None:
                    ref_freq_hz = float(spec.get("fc_hz") or np.sqrt(float(np.min(freq_hz)) * float(np.max(freq_hz))))
                return run_ac_analysis_with_ngspice(circuit, freq_hz, z0, q_L=q_L, q_C=q_C, ref_freq_hz=ref_freq_hz)
            except RuntimeError:
                pass

    # 回退：用 Fast Track 引擎替代（支持 notch 与可微 Q 模型）
    import torch

    from src.physics import FastTrackEngine

    engine = FastTrackEngine(z0=z0, device="cpu", dtype=torch.float64)
    return engine.simulate_sparams_db(
        components,
        freq_hz,
        q_L=q_L,
        q_C=q_C,
        q_model=str(q_model),
        ref_freq_hz=ref_freq_hz if q_model == "fixed_ref" else None,
    )
