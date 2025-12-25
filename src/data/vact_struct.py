"""
VACT-Struct: a composable, execution-oriented IR for LC circuits.

Design goals (paper-facing):
- composable: explicit <CIRCUIT>/<CELL> blocks (optionally hierarchical tags later)
- recursive-friendly: structure tokens enable future <SUBCKT>/<CALL> expansion
- backward compatible: stripping structure tokens yields the original VACT 5-token
  component sequence (<TYPE> <ROLE> <VAL_xxx> <NODE_a> <NODE_b>).
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Set

from .schema import ComponentSpec
from .vact_codec import (
    CELL_TOKEN,
    components_to_vact_tokens,
    vact_tokens_to_components,
)


# ---- VACT-Struct structural tokens (avoid collision with VACT role tokens) ----

CIRCUIT_START = "<CIRCUIT>"
CIRCUIT_END = "</CIRCUIT>"

CELL_END = "</CELL>"

SERIES_BLOCK_START = "<SERIES_BLOCK>"
SERIES_BLOCK_END = "</SERIES_BLOCK>"
SHUNT_BLOCK_START = "<SHUNT_BLOCK>"
SHUNT_BLOCK_END = "</SHUNT_BLOCK>"

# Generic device block (multi-pin, multi-param friendly).
DEVICE_START = "<DEV>"
DEVICE_END = "</DEV>"
PINS_START = "<PINS>"
PINS_END = "</PINS>"
PARAMS_START = "<PARAMS>"
PARAMS_END = "</PARAMS>"

PORT_IN = "<PORT_IN>"
PORT_OUT = "<PORT_OUT>"

# Keep Z0 explicit but fixed for now (dataset uses 50Ω everywhere).
Z0_50 = "<Z0_50>"

# Reserved (not yet used by generators, but kept for forward compatibility).
SUBCKT_START = "<SUBCKT>"
SUBCKT_END = "</SUBCKT>"
CALL_START = "<CALL>"
CALL_END = "</CALL>"


_STRUCT_TOKENS: Set[str] = {
    CIRCUIT_START,
    CIRCUIT_END,
    CELL_TOKEN,
    CELL_END,
    SERIES_BLOCK_START,
    SERIES_BLOCK_END,
    SHUNT_BLOCK_START,
    SHUNT_BLOCK_END,
    PORT_IN,
    PORT_OUT,
    Z0_50,
    DEVICE_START,
    DEVICE_END,
    PINS_START,
    PINS_END,
    PARAMS_START,
    PARAMS_END,
    SUBCKT_START,
    SUBCKT_END,
    CALL_START,
    CALL_END,
}


def components_to_vact_struct_tokens(
    components: List[ComponentSpec],
    *,
    z0: float = 50.0,
    include_ports: bool = True,
    emit_cells: bool = True,
    emit_device_blocks: bool = False,
) -> List[str]:
    """
    Encode components into a VACT-Struct token sequence.

    Notes:
    - Internally uses VACT 5-token components for maximal backward compatibility.
    - Structural grouping uses <CELL> + <SERIES_BLOCK>/<SHUNT_BLOCK> with explicit end tags.
    """
    tokens: List[str] = [CIRCUIT_START]
    if include_ports:
        tokens.extend([PORT_IN, "<NODE_in>", PORT_OUT, "<NODE_out>"])
    if float(z0) == 50.0:
        tokens.append(Z0_50)

    if not emit_cells:
        tokens.extend(components_to_vact_tokens(components, emit_cell_tokens=False, normalize_node_order=True))
        tokens.append(CIRCUIT_END)
        return tokens

    # Use VACT's own canonicalization + optional <CELL> markers, then lift into blocks.
    flat = components_to_vact_tokens(components, emit_cell_tokens=True, normalize_node_order=True)
    # Split by <CELL> markers.
    cells: List[List[str]] = []
    cur: List[str] = []
    for tok in flat:
        if tok == CELL_TOKEN:
            if cur:
                cells.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        cells.append(cur)

    for cell_tokens in cells:
        # cell_tokens is a flat list of 5-token components (no <CELL> inside).
        series_parts: List[str] = []
        shunt_parts: List[str] = []
        device_parts: List[str] = []
        for i in range(0, len(cell_tokens) // 5):
            chunk = cell_tokens[i * 5 : (i + 1) * 5]
            role_tok = chunk[1]
            if role_tok == "<SERIES>":
                series_parts.extend(chunk)
            else:
                shunt_parts.extend(chunk)

            if emit_device_blocks:
                # Expand to generic device block: <DEV> <TYPE> <PINS> <NODE_a> <NODE_b> </PINS> <PARAMS> <VAL_x> </PARAMS> </DEV>
                device_parts.extend(
                    [
                        DEVICE_START,
                        chunk[0],  # TYPE
                        PINS_START,
                        chunk[3],
                        chunk[4],
                        PINS_END,
                        PARAMS_START,
                        chunk[2],  # VAL
                        PARAMS_END,
                        DEVICE_END,
                    ]
                )
        tokens.append(CELL_TOKEN)
        if emit_device_blocks:
            tokens.extend(device_parts)
        else:
            tokens.append(SERIES_BLOCK_START)
            tokens.extend(series_parts)
            tokens.append(SERIES_BLOCK_END)
            tokens.append(SHUNT_BLOCK_START)
            tokens.extend(shunt_parts)
            tokens.append(SHUNT_BLOCK_END)
        tokens.append(CELL_END)

    tokens.append(CIRCUIT_END)
    return tokens


def _strip_vact_struct_non_component_tokens(tokens: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for tok in tokens:
        if tok in _STRUCT_TOKENS:
            continue
        cleaned.append(tok)
    return cleaned


def vact_struct_tokens_to_components(
    tokens: List[str],
    label_to_value: Mapping[str, float] | None = None,
    *,
    drop_non_component_tokens: bool = True,
) -> List[ComponentSpec]:
    """
    Decode VACT-Struct tokens back to components.
    When drop_non_component_tokens=True, structure tokens are ignored and the sequence
    degrades to the original VACT format.
    """
    if drop_non_component_tokens:
        tokens = _strip_vact_struct_non_component_tokens(tokens)
    return vact_tokens_to_components(tokens, label_to_value=label_to_value, drop_non_component_tokens=True)


def build_vact_struct_vocab(
    value_labels: Sequence[str],
    node_names: Sequence[str] = ("in", "out", "gnd") + tuple(f"n{k}" for k in range(16)),
    order_range: tuple[int, int] | None = (2, 7),
    include_ports: bool = True,
) -> List[str]:
    """
    Build a tokenizer vocab that supports VACT-Struct + the original VACT tokens.
    """
    vocab = set(components_to_vact_tokens([]))  # empty list -> []
    # Reuse VACT's vocab builder via components tokens generation is not enough; import directly.
    from .vact_codec import build_vact_vocab

    vocab.update(build_vact_vocab(value_labels=value_labels, node_names=node_names, order_range=order_range, include_cell_token=True))
    vocab.update(
        {
            CIRCUIT_START,
            CIRCUIT_END,
            CELL_END,
            SERIES_BLOCK_START,
            SERIES_BLOCK_END,
            SHUNT_BLOCK_START,
            SHUNT_BLOCK_END,
            DEVICE_START,
            DEVICE_END,
            PINS_START,
            PINS_END,
            PARAMS_START,
            PARAMS_END,
            Z0_50,
            SUBCKT_START,
            SUBCKT_END,
            CALL_START,
            CALL_END,
        }
    )
    if include_ports:
        vocab.update({PORT_IN, PORT_OUT, "<NODE_in>", "<NODE_out>"})
    return sorted(vocab)


def make_vact_struct_prefix_allowed_tokens_fn(
    tokenizer,
    *,
    require_ports: bool = False,
    require_z0: bool = False,
    allow_device_blocks: bool = False,
) -> "callable":
    """
    Prefix constraint for VACT-Struct decoding.

    Grammar (simplified):
      (<ORDER_k>)* <SEP>* <CIRCUIT>
        [<PORT_IN> <NODE_in> <PORT_OUT> <NODE_out>]
        [<Z0_50>]
        (<CELL> <SERIES_BLOCK> (COMP)* </SERIES_BLOCK> <SHUNT_BLOCK> (COMP)* </SHUNT_BLOCK> </CELL>)*
      </CIRCUIT> </s>

    where COMP is the original VACT 5-token component:
      <L/C> <SERIES/SHUNT> <VAL_xxx> <NODE_a> <NODE_b>

    Additionally:
    - Inside <SERIES_BLOCK>, role token is forced to <SERIES>.
    - Inside <SHUNT_BLOCK>, role token is forced to <SHUNT>.
    """
    vocab = tokenizer.get_vocab()
    all_ids = list(vocab.values())

    def _id(tok: str) -> int | None:
        return vocab.get(tok)

    # token sets
    type_ids = {vocab[tok] for tok in ("<L>", "<C>") if tok in vocab}
    val_ids = {tid for tok, tid in vocab.items() if tok.startswith("<VAL_")}
    node_ids = {tid for tok, tid in vocab.items() if tok.startswith("<NODE_")}
    order_ids = {tid for tok, tid in vocab.items() if tok.startswith("<ORDER_")}

    sep_id = _id("<SEP>")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    # exact tokens
    circuit_start_id = _id(CIRCUIT_START)
    circuit_end_id = _id(CIRCUIT_END)
    cell_start_id = _id(CELL_TOKEN)
    cell_end_id = _id(CELL_END)
    series_blk_start_id = _id(SERIES_BLOCK_START)
    series_blk_end_id = _id(SERIES_BLOCK_END)
    shunt_blk_start_id = _id(SHUNT_BLOCK_START)
    shunt_blk_end_id = _id(SHUNT_BLOCK_END)
    port_in_id = _id(PORT_IN)
    port_out_id = _id(PORT_OUT)
    node_in_id = _id("<NODE_in>")
    node_out_id = _id("<NODE_out>")
    z0_id = _id(Z0_50)
    role_series_id = _id("<SERIES>")
    role_shunt_id = _id("<SHUNT>")

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Fail open if tokenizer missing required symbols.
    required = [
        circuit_start_id,
        circuit_end_id,
        cell_start_id,
        cell_end_id,
    ]
    if not allow_device_blocks:
        required.extend([series_blk_start_id, series_blk_end_id, shunt_blk_start_id, shunt_blk_end_id])
    if any(x is None for x in required):
        return lambda batch_id, input_ids: all_ids

    # State machine (see docstring).
    S_START = 0
    S_AFTER_CIRCUIT = 1
    S_AFTER_PORT_IN = 2
    S_AFTER_NODE_IN = 3
    S_AFTER_PORT_OUT = 4
    S_AFTER_NODE_OUT = 5
    S_BODY = 6
    S_AFTER_CELL = 7
    S_IN_SERIES_EXPECT_TYPE_OR_END = 8
    S_IN_SERIES_EXPECT_ROLE = 9
    S_IN_SERIES_EXPECT_VAL = 10
    S_IN_SERIES_EXPECT_N1 = 11
    S_IN_SERIES_EXPECT_N2 = 12
    S_AFTER_SERIES_END = 13
    S_IN_SHUNT_EXPECT_TYPE_OR_END = 14
    S_IN_SHUNT_EXPECT_ROLE = 15
    S_IN_SHUNT_EXPECT_VAL = 16
    S_IN_SHUNT_EXPECT_N1 = 17
    S_IN_SHUNT_EXPECT_N2 = 18
    S_AFTER_SHUNT_END = 19
    S_DONE = 20

    def _state_from_ids(input_ids: List[int]) -> tuple[int, bool]:
        state = S_START
        valid = True
        ports_seen = False
        z0_seen = False
        for tid in input_ids:
            if tid in special_ids:
                continue
            if state == S_START:
                if tid in order_ids:
                    continue
                if sep_id is not None and tid == sep_id:
                    continue
                if tid == circuit_start_id:
                    state = S_AFTER_CIRCUIT
                    continue
                valid = False
                break

            if state == S_AFTER_CIRCUIT:
                if port_in_id is not None and tid == port_in_id:
                    state = S_AFTER_PORT_IN
                    ports_seen = True
                    continue
                if not require_ports:
                    if z0_id is not None and tid == z0_id:
                        z0_seen = True
                        state = S_BODY
                        continue
                    if tid == cell_start_id:
                        state = S_AFTER_CELL
                        continue
                    if tid == circuit_end_id:
                        state = S_DONE
                        continue
                valid = False
                break

            if state == S_AFTER_PORT_IN:
                if node_in_id is not None and tid == node_in_id:
                    state = S_AFTER_NODE_IN
                    continue
                valid = False
                break

            if state == S_AFTER_NODE_IN:
                if port_out_id is not None and tid == port_out_id:
                    state = S_AFTER_PORT_OUT
                    continue
                valid = False
                break

            if state == S_AFTER_PORT_OUT:
                if node_out_id is not None and tid == node_out_id:
                    state = S_AFTER_NODE_OUT
                    continue
                valid = False
                break

            if state == S_AFTER_NODE_OUT:
                if z0_id is not None and tid == z0_id:
                    z0_seen = True
                    state = S_BODY
                    continue
                if not require_z0:
                    if tid == cell_start_id:
                        state = S_AFTER_CELL
                        continue
                    if tid == circuit_end_id:
                        state = S_DONE
                        continue
                valid = False
                break

            if state == S_BODY:
                if tid == cell_start_id:
                    state = S_AFTER_CELL
                    continue
                if tid == circuit_end_id:
                    state = S_DONE
                    continue
                valid = False
                break

            if state == S_AFTER_CELL:
                if allow_device_blocks and tid == _id(DEVICE_START):
                    state = S_BODY  # treat opaque device blocks as a single unit
                    continue
                if not allow_device_blocks and tid == series_blk_start_id:
                    state = S_IN_SERIES_EXPECT_TYPE_OR_END
                    continue
                valid = False
                break

            if state == S_IN_SERIES_EXPECT_TYPE_OR_END:
                if tid in type_ids:
                    state = S_IN_SERIES_EXPECT_ROLE
                    continue
                if tid == series_blk_end_id:
                    state = S_AFTER_SERIES_END
                    continue
                valid = False
                break

            if state == S_IN_SERIES_EXPECT_ROLE:
                if role_series_id is not None and tid == role_series_id:
                    state = S_IN_SERIES_EXPECT_VAL
                    continue
                valid = False
                break

            if state == S_IN_SERIES_EXPECT_VAL:
                if tid in val_ids:
                    state = S_IN_SERIES_EXPECT_N1
                    continue
                valid = False
                break

            if state == S_IN_SERIES_EXPECT_N1:
                if tid in node_ids:
                    state = S_IN_SERIES_EXPECT_N2
                    continue
                valid = False
                break

            if state == S_IN_SERIES_EXPECT_N2:
                if tid in node_ids:
                    state = S_IN_SERIES_EXPECT_TYPE_OR_END
                    continue
                valid = False
                break

            if state == S_AFTER_SERIES_END:
                if tid == shunt_blk_start_id:
                    state = S_IN_SHUNT_EXPECT_TYPE_OR_END
                    continue
                valid = False
                break

            if state == S_IN_SHUNT_EXPECT_TYPE_OR_END:
                if tid in type_ids:
                    state = S_IN_SHUNT_EXPECT_ROLE
                    continue
                if tid == shunt_blk_end_id:
                    state = S_AFTER_SHUNT_END
                    continue
                valid = False
                break

            if state == S_IN_SHUNT_EXPECT_ROLE:
                if role_shunt_id is not None and tid == role_shunt_id:
                    state = S_IN_SHUNT_EXPECT_VAL
                    continue
                valid = False
                break

            if state == S_IN_SHUNT_EXPECT_VAL:
                if tid in val_ids:
                    state = S_IN_SHUNT_EXPECT_N1
                    continue
                valid = False
                break

            if state == S_IN_SHUNT_EXPECT_N1:
                if tid in node_ids:
                    state = S_IN_SHUNT_EXPECT_N2
                    continue
                valid = False
                break

            if state == S_IN_SHUNT_EXPECT_N2:
                if tid in node_ids:
                    state = S_IN_SHUNT_EXPECT_TYPE_OR_END
                    continue
                valid = False
                break

            if state == S_AFTER_SHUNT_END:
                if tid == cell_end_id:
                    state = S_BODY
                    continue
                valid = False
                break

            if state == S_DONE:
                # ignore any trailing tokens; generation should stop at EOS anyway.
                continue

        if require_ports and state in (S_AFTER_CIRCUIT,) and not ports_seen:
            valid = False
        if require_z0 and state in (S_AFTER_CIRCUIT, S_AFTER_NODE_OUT, S_BODY) and not z0_seen:
            valid = False
        return state, valid

    def _prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        state, valid = _state_from_ids(ids)
        if not valid:
            return all_ids

        allowed: Set[int] = set()
        if state == S_START:
            allowed.update(order_ids)
            if sep_id is not None:
                allowed.add(sep_id)
            if circuit_start_id is not None:
                allowed.add(circuit_start_id)
            if eos_id is not None:
                allowed.add(eos_id)
        elif state == S_AFTER_CIRCUIT:
            if port_in_id is not None:
                allowed.add(port_in_id)
            if not require_ports:
                if z0_id is not None:
                    allowed.add(z0_id)
                allowed.update({cell_start_id, circuit_end_id})
        elif state == S_AFTER_PORT_IN:
            if node_in_id is not None:
                allowed.add(node_in_id)
        elif state == S_AFTER_NODE_IN:
            if port_out_id is not None:
                allowed.add(port_out_id)
        elif state == S_AFTER_PORT_OUT:
            if node_out_id is not None:
                allowed.add(node_out_id)
        elif state == S_AFTER_NODE_OUT:
            if z0_id is not None:
                allowed.add(z0_id)
            if not require_z0:
                allowed.update({cell_start_id, circuit_end_id})
        elif state == S_BODY:
            allowed.update({cell_start_id, circuit_end_id})
            if eos_id is not None:
                allowed.add(eos_id)
        elif state == S_AFTER_CELL:
            if allow_device_blocks and _id(DEVICE_START) is not None:
                allowed.add(_id(DEVICE_START))
            if not allow_device_blocks and series_blk_start_id is not None:
                allowed.add(series_blk_start_id)
        elif state == S_IN_SERIES_EXPECT_TYPE_OR_END:
            allowed.update(type_ids)
            allowed.add(series_blk_end_id)
        elif state == S_IN_SERIES_EXPECT_ROLE:
            if role_series_id is not None:
                allowed.add(role_series_id)
        elif state == S_IN_SERIES_EXPECT_VAL:
            allowed.update(val_ids)
        elif state in (S_IN_SERIES_EXPECT_N1, S_IN_SERIES_EXPECT_N2):
            allowed.update(node_ids)
        elif state == S_AFTER_SERIES_END:
            allowed.add(shunt_blk_start_id)
        elif state == S_IN_SHUNT_EXPECT_TYPE_OR_END:
            allowed.update(type_ids)
            allowed.add(shunt_blk_end_id)
        elif state == S_IN_SHUNT_EXPECT_ROLE:
            if role_shunt_id is not None:
                allowed.add(role_shunt_id)
        elif state == S_IN_SHUNT_EXPECT_VAL:
            allowed.update(val_ids)
        elif state in (S_IN_SHUNT_EXPECT_N1, S_IN_SHUNT_EXPECT_N2):
            allowed.update(node_ids)
        elif state == S_AFTER_SHUNT_END:
            allowed.add(cell_end_id)
        elif state == S_DONE:
            if eos_id is not None:
                allowed.add(eos_id)

        if pad_id is not None:
            allowed.add(pad_id)
        # If we computed an empty allowed set (missing tokens), fail open.
        if not allowed:
            return all_ids
        return sorted(allowed)

    return _prefix_allowed_tokens_fn
