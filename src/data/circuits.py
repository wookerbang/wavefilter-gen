"""
电路中间表示 + ABCD 工具。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .schema import ComponentSpec


def _infer_output_node(components: List[ComponentSpec]) -> str:
    for comp in reversed(components):
        if comp.node2 != "gnd":
            return comp.node2
        if comp.node1 != "gnd":
            return comp.node1
    return "out"


class Circuit:
    def __init__(
        self,
        components: List[ComponentSpec],
        z0: float = 50.0,
        in_port: Tuple[str, str] = ("in", "gnd"),
        out_port: Tuple[str, str] | None = None,
    ):
        self.components = components
        self.z0 = z0
        self.in_port = in_port
        self.out_port = out_port or (_infer_output_node(components), "gnd")

    def to_spice_netlist(
        self,
        title: str = "LC_FILTER",
        *,
        q_L: float | None = None,
        q_C: float | None = None,
        ref_freq_hz: float | None = None,
    ) -> str:
        """
        输出一个简单的 2-port AC 仿真 netlist。
        """
        lines = [f"* {title}"]
        src_node, gnd_node = self.in_port
        out_node, _ = self.out_port
        lines.append(f"V1 {src_node} {gnd_node} AC 1")
        lines.append(f"Rload {out_node} {gnd_node} {self.z0}")

        q_L_eff = float(q_L) if q_L is not None else None
        q_C_eff = float(q_C) if q_C is not None else None
        if q_L_eff is not None and (not np.isfinite(q_L_eff) or q_L_eff <= 0):
            q_L_eff = None
        if q_C_eff is not None and (not np.isfinite(q_C_eff) or q_C_eff <= 0):
            q_C_eff = None

        w_ref = None
        if (q_L_eff is not None or q_C_eff is not None) and ref_freq_hz is not None:
            w_ref = 2.0 * np.pi * float(ref_freq_hz)
        if (q_L_eff is not None or q_C_eff is not None) and w_ref is None:
            raise ValueError("ref_freq_hz is required when q_L/q_C are enabled for SPICE netlist generation.")

        for idx, comp in enumerate(self.components, start=1):
            name = f"{comp.ctype}{idx}"
            value = float(comp.value_si)

            if comp.ctype == "L" and q_L_eff is not None and w_ref is not None:
                r_ser = (w_ref * value) / float(q_L_eff)
                mid = f"q_{name}"
                if r_ser > 0:
                    lines.append(f"R{name}Q {comp.node1} {mid} {r_ser}")
                else:
                    mid = comp.node1
                lines.append(f"{name} {mid} {comp.node2} {value}")
                continue

            lines.append(f"{name} {comp.node1} {comp.node2} {value}")

            if comp.ctype == "C" and q_C_eff is not None and w_ref is not None:
                # Parallel resistor: Rp = Q / (ωC)
                rp = float(q_C_eff) / (w_ref * value)
                if rp > 0 and np.isfinite(rp):
                    lines.append(f"R{name}Q {comp.node1} {comp.node2} {rp}")

        # 控制语句由调用者补充
        return "\n".join(lines)


def components_to_abcd(
    components: List[ComponentSpec],
    freq_hz: np.ndarray,
    z0: float,
    *,
    q_L: float | None = None,
    q_C: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * freq_hz
    n = len(freq_hz)
    A = np.ones(n, dtype=complex)
    B = np.zeros(n, dtype=complex)
    C = np.zeros(n, dtype=complex)
    D = np.ones(n, dtype=complex)

    for comp in components:
        if comp.ctype == "L" and comp.role == "series":
            Z = 1j * omega * comp.value_si
            if q_L is not None and q_L > 0:
                Z = Z + (omega * comp.value_si) / float(q_L)
            B = A * Z + B
            D = C * Z + D
        elif comp.ctype == "C" and comp.role == "series":
            Y = 1j * omega * comp.value_si
            if q_C is not None and q_C > 0:
                Y = Y + (omega * comp.value_si) / float(q_C)
            Z = 1.0 / (Y + 1e-18)
            B = A * Z + B
            D = C * Z + D
        elif comp.ctype == "C" and comp.role == "shunt":
            Y = 1j * omega * comp.value_si
            if q_C is not None and q_C > 0:
                Y = Y + (omega * comp.value_si) / float(q_C)
            A = A + B * Y
            C = C + D * Y
        elif comp.ctype == "L" and comp.role == "shunt":
            Z = 1j * omega * comp.value_si
            if q_L is not None and q_L > 0:
                Z = Z + (omega * comp.value_si) / float(q_L)
            Y = 1.0 / (Z + 1e-18)
            A = A + B * Y
            C = C + D * Y
        else:
            continue
    return A, B, C, D


def abcd_to_sparams(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    z0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    S11, S21, _, _ = abcd_to_sparams_complex(A, B, C, D, z0)
    s21_db = 20.0 * np.log10(np.abs(S21) + 1e-12)
    s11_db = 20.0 * np.log10(np.abs(S11) + 1e-12)
    return s21_db, s11_db


def abcd_to_sparams_complex(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    z0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ABCD parameters to complex S-parameters with equal reference z0.
    Returns (S11, S21, S12, S22), each shape (F,).
    """
    denom = A + B / z0 + C * z0 + D
    denom = denom + 1e-18
    S11 = (A + B / z0 - C * z0 - D) / denom
    S21 = 2.0 / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S22 = (-A + B / z0 - C * z0 + D) / denom
    return S11, S21, S12, S22


def components_to_sparams_nodal(
    components: List[ComponentSpec],
    freq_hz: np.ndarray,
    z0: float = 50.0,
    *,
    port_in: str = "in",
    port_out: str = "out",
    gnd: str = "gnd",
    gmin: float = 1e-12,
) -> np.ndarray:
    """
    General 2-port S-parameter simulation via nodal admittance + Kron reduction.

    Returns:
      S: complex array of shape (F, 2, 2) with port order [in, out].
    """
    omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    n_freq = omega.shape[0]

    # Build node index (exclude ground).
    nodes = sorted({n for c in components for n in (c.node1, c.node2) if n != gnd})
    if port_in not in nodes:
        nodes = [port_in] + nodes
    if port_out not in nodes:
        nodes = nodes + [port_out]
    # Ensure uniqueness while preserving insertion.
    seen = set()
    nodes = [n for n in nodes if not (n in seen or seen.add(n))]

    idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    if n_nodes == 0:
        return np.zeros((n_freq, 2, 2), dtype=complex)

    Y = np.zeros((n_freq, n_nodes, n_nodes), dtype=complex)
    jw = 1j * omega
    eps = 1e-30

    for comp in components:
        n1 = comp.node1
        n2 = comp.node2
        if n1 == gnd and n2 == gnd:
            continue
        if comp.ctype == "C":
            y = jw * float(comp.value_si)
        elif comp.ctype == "L":
            y = 1.0 / (jw * float(comp.value_si) + eps)
        else:
            continue

        i = idx.get(n1) if n1 != gnd else None
        j = idx.get(n2) if n2 != gnd else None
        if i is not None:
            Y[:, i, i] += y
        if j is not None:
            Y[:, j, j] += y
        if i is not None and j is not None:
            Y[:, i, j] -= y
            Y[:, j, i] -= y

    if gmin and gmin > 0:
        Y[:, range(n_nodes), range(n_nodes)] += float(gmin)

    p_idx = [idx[port_in], idx[port_out]]
    i_idx = [k for k in range(n_nodes) if k not in p_idx]

    Ypp = Y[:, p_idx, :][:, :, p_idx]  # (F,2,2)
    if not i_idx:
        Y_port = Ypp
    else:
        Ypi = Y[:, p_idx, :][:, :, i_idx]  # (F,2,Ni)
        Yii = Y[:, i_idx, :][:, :, i_idx]  # (F,Ni,Ni)
        Yip = Y[:, i_idx, :][:, :, p_idx]  # (F,Ni,2)
        # Kron reduction: Y_port = Ypp - Ypi @ inv(Yii) @ Yip
        Y_port = np.empty((n_freq, 2, 2), dtype=complex)
        for k in range(n_freq):
            sol = np.linalg.solve(Yii[k], Yip[k])  # (Ni,2)
            Y_port[k] = Ypp[k] - Ypi[k] @ sol

    # Y -> S with equal reference z0: S = (I - z0*Y) (I + z0*Y)^{-1}
    I2 = np.eye(2, dtype=complex)
    S = np.empty((n_freq, 2, 2), dtype=complex)
    for k in range(n_freq):
        A = I2 - float(z0) * Y_port[k]
        Bm = I2 + float(z0) * Y_port[k]
        # (I+Z0Y)^{-1}(I-Z0Y) == (I-Z0Y)(I+Z0Y)^{-1} since both are polynomials in Y.
        S[k] = np.linalg.solve(Bm, A)
    return S
