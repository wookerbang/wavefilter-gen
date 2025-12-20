"""
Action-oriented construction trace for circuits (ActionVACT).

Goal: describe how a circuit is built, not just the final netlist, to mitigate
node-renaming symmetry and enable procedural generation / RL-style rollouts.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Set

from .schema import ComponentSpec

# Action tokens
ACT_START = "<ACT_START>"
ACT_END = "<ACT_END>"
ACT_ADD_NODE = "<ACT_ADD_NODE>"
ACT_ADD_DEV = "<ACT_ADD_DEV>"
ACT_CONNECT = "<ACT_CONNECT>"
ACT_CLOSE = "<ACT_CLOSE>"

PIN_PREFIX = "<PIN_"


def _pin_token(node: str) -> str:
    return f"{PIN_PREFIX}{node}>"


def components_to_action_tokens(components: Sequence[ComponentSpec]) -> List[str]:
    """
    Encode a simple ladder-like circuit into an action sequence.
    Actions:
      - ADD_NODE <PIN_x>
      - ADD_DEV <TYPE> <ROLE> <VAL_xxx> <PIN_a> <PIN_b>
    """
    tokens: List[str] = [ACT_START]
    # declare reserved nodes first
    for node in ("in", "out", "gnd"):
        tokens.extend([ACT_ADD_NODE, _pin_token(node)])

    seen: Set[str] = {"in", "out", "gnd"}
    # declare remaining nodes deterministically
    for comp in components:
        for node in (comp.node1, comp.node2):
            if node not in seen:
                tokens.extend([ACT_ADD_NODE, _pin_token(node)])
                seen.add(node)

    for comp in components:
        tokens.extend(
            [
                ACT_ADD_DEV,
                f"<{comp.ctype.upper()}>",
                f"<{comp.role.upper()}>",
                f"<VAL_{comp.std_label or 'NA'}>",
                _pin_token(comp.node1),
                _pin_token(comp.node2),
            ]
        )
    tokens.append(ACT_END)
    return tokens


def action_tokens_to_components(tokens: Sequence[str], label_to_value: Mapping[str, float] | None = None) -> List[ComponentSpec]:
    """
    Recover components from a flat action sequence. This assumes deterministic ADD_DEV steps.
    """
    comps: List[ComponentSpec] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == ACT_ADD_DEV and i + 5 < len(tokens):
            ctype = tokens[i + 1].strip("<>").upper()
            role = tokens[i + 2].strip("<>").lower()
            val_tok = tokens[i + 3]
            label = val_tok.replace("<VAL_", "").replace(">", "")
            if label == "NA":
                label = None
            node1 = tokens[i + 4].replace(PIN_PREFIX, "").replace(">", "")
            node2 = tokens[i + 5].replace(PIN_PREFIX, "").replace(">", "")
            value = 0.0
            if label:
                if label_to_value is not None:
                    value = float(label_to_value.get(label, 0.0))
                else:
                    try:
                        from .quantization import label_to_value as _label_to_value
                        value = float(_label_to_value(label))
                    except Exception:
                        value = 0.0
            comps.append(ComponentSpec(ctype=ctype, role=role, value_si=value, std_label=label, node1=node1, node2=node2))
            i += 6
        else:
            i += 1
    return comps


def build_action_vocab(
    value_labels: Sequence[str],
    node_names: Sequence[str],
) -> List[str]:
    vocab: Set[str] = {
        ACT_START,
        ACT_END,
        ACT_ADD_NODE,
        ACT_ADD_DEV,
        ACT_CONNECT,
        ACT_CLOSE,
    }
    vocab.update([f"<L>", f"<C>", f"<SERIES>", f"<SHUNT>"])
    for node in node_names:
        vocab.add(_pin_token(node))
    for label in value_labels:
        vocab.add(f"<VAL_{label}>")
    return sorted(vocab)

