"""
Canonicalize node names to a finite set (in/out/gnd + n1..nK).

Why:
- Data generation may introduce ad-hoc node names (e.g., bp_mid0, *_notch, *_b2).
- Tokenizers use a fixed node vocabulary; uncontrolled node strings become <unk>.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .schema import ComponentSpec


RESERVED_NODES = ("in", "out", "gnd")


def canonicalize_nodes(
    components: Sequence[ComponentSpec],
    *,
    reserved_nodes: Tuple[str, str, str] = RESERVED_NODES,
    start_index: int = 1,
    max_nodes: int | None = 32,
) -> List[ComponentSpec]:
    """
    Rename all non-reserved nodes to n{start_index}.. in a deterministic order.
    """
    reserved = set(reserved_nodes)
    internal: List[str] = sorted({n for c in components for n in (c.node1, c.node2) if n not in reserved})
    if max_nodes is not None and len(internal) > int(max_nodes):
        raise ValueError(f"Too many internal nodes ({len(internal)}) > max_nodes ({max_nodes}).")

    mapping: Dict[str, str] = {old: f"n{start_index + i}" for i, old in enumerate(internal)}
    out: List[ComponentSpec] = []
    for c in components:
        out.append(
            ComponentSpec(
                ctype=c.ctype,
                role=c.role,
                value_si=float(c.value_si),
                std_label=c.std_label,
                node1=mapping.get(c.node1, c.node1),
                node2=mapping.get(c.node2, c.node2),
            )
        )
    return out

