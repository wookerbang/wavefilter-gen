"""
VACT-DSL v2.1: ML-friendly IR with Macro / Repeat / Typed numeric slots.

Key ideas:
- Canonical main program: <MAIN> Ports Body </MAIN>
- Body supports structured cascade repeat blocks with frozen macro library.
- Numeric slots are typed (<VAL_L>/<VAL_C>/<VAL_R>/...), paired with continuous heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

from .schema import ComponentSpec
from .action_codec import components_to_action_tokens

# ---- Tokens ----

BOS = "<BOS>"
# Align EOS with tokenizer eos (T5 uses </s>) to avoid double-eos drift.
EOS = "</s>"

MAIN_START = "<MAIN>"
MAIN_END = "</MAIN>"

REPEAT_START = "<REPEAT>"
REPEAT_END = "</REPEAT>"
CASCADE = "<CASCADE>"
CALL = "<CALL>"

CELL = "<CELL>"
# Optional cell index markers to stabilize long-K decoding.
CELL_INDEX_TOKENS = [f"<CELL_IDX_{i}>" for i in range(32)]

PORT_IN = "<P_IN>"
PORT_OUT = "<P_OUT>"
PORT_GND = "<P_GND>"

VAL_L = "<VAL_L>"
VAL_C = "<VAL_C>"
VAL_R = "<VAL_R>"
VAL_TL_Z0 = "<VAL_TL_Z0>"
VAL_TL_LEN = "<VAL_TL_LEN>"

VALUE_SLOTS = [VAL_L, VAL_C, VAL_R, VAL_TL_Z0, VAL_TL_LEN]

# Repeat factors (frozen finite set + extensible varint).
K_TOKENS = [f"<K_{k}>" for k in range(1, 13)]
K_VAR_START = "<K>"
K_VAR_END = "</K>"
DIGIT_TOKENS = [f"<D_{d}>" for d in range(10)]

# Macro tokens (frozen library)
MACRO_CELL_LS_CS = "<MAC_CELL_LS_CS>"
MACRO_CELL_CS_LS = "<MAC_CELL_CS_LS>"
MACRO_NOTCH_SHUNT_LC_SER = "<MAC_NOTCH_SHUNT_LC_SER>"
MACRO_CELL_LS = "<MAC_CELL_LS>"
MACRO_CELL_CS = "<MAC_CELL_CS>"
MACRO_PI_CLC = "<MAC_PI_CLC>"
MACRO_T_LCL = "<MAC_T_LCL>"
MACRO_DOUBLE_SERIES_LC = "<MAC_DOUBLE_SERIES_LC>"
MACRO_DOUBLE_SHUNT_LC = "<MAC_DOUBLE_SHUNT_LC>"
MACRO_BRIDGE_C = "<MAC_BRIDGE_C>"

MACRO_IDS = [
    MACRO_CELL_LS_CS,
    MACRO_CELL_CS_LS,
    MACRO_NOTCH_SHUNT_LC_SER,
    MACRO_CELL_LS,
    MACRO_CELL_CS,
    MACRO_PI_CLC,
    MACRO_T_LCL,
    MACRO_DOUBLE_SERIES_LC,
    MACRO_DOUBLE_SHUNT_LC,
    MACRO_BRIDGE_C,
]


# ---- Macro definitions ----


@dataclass(frozen=True)
class MacroDef:
    name: str
    slot_types: Tuple[str, ...]  # e.g., ("L", "C")
    # inst_idx is a deterministic macro instance index used for internal node allocation.
    expand_fn: Callable[[str, str, str, List[float], int], List[ComponentSpec]]


def _expand_cell_ls_cs(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    return [
        ComponentSpec("L", "series", L_val, None, a, b),
        ComponentSpec("C", "shunt", C_val, None, b, gnd),
    ]


def _expand_cell_cs_ls(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val, L_val = (vals + [0.0, 0.0])[:2]
    return [
        ComponentSpec("C", "series", C_val, None, a, b),
        ComponentSpec("L", "shunt", L_val, None, b, gnd),
    ]


def _expand_notch_shunt(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # series LC from anchor (b) to ground via internal node x
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    x = f"x{inst_idx}_0"
    return [
        ComponentSpec("L", "series", L_val, None, b, x),
        ComponentSpec("C", "series", C_val, None, x, gnd),
    ]


def _expand_cell_ls(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val = (vals + [0.0])[0]
    return [ComponentSpec("L", "series", L_val, None, a, b)]


def _expand_cell_cs(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val = (vals + [0.0])[0]
    return [ComponentSpec("C", "series", C_val, None, a, b)]


def _expand_pi_clc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C1, L, C2 = (vals + [0.0, 0.0, 0.0])[:3]
    return [
        ComponentSpec("C", "shunt", C1, None, a, gnd),
        ComponentSpec("L", "series", L, None, a, b),
        ComponentSpec("C", "shunt", C2, None, b, gnd),
    ]


def _expand_t_lcl(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L1, C, L2 = (vals + [0.0, 0.0, 0.0])[:3]
    x = f"x{inst_idx}_0"
    return [
        ComponentSpec("L", "series", L1, None, a, x),
        ComponentSpec("C", "shunt", C, None, x, gnd),
        ComponentSpec("L", "series", L2, None, x, b),
    ]


def _expand_double_series_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L, C = (vals + [0.0, 0.0])[:2]
    x = f"x{inst_idx}_0"
    return [
        ComponentSpec("L", "series", L, None, a, x),
        ComponentSpec("C", "series", C, None, x, b),
    ]


def _expand_double_shunt_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L, C = (vals + [0.0, 0.0])[:2]
    return [
        ComponentSpec("L", "shunt", L, None, b, gnd),
        ComponentSpec("C", "shunt", C, None, b, gnd),
    ]


def _expand_bridge_c(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Simple bridge capacitor across anchor nodes.
    C_val = (vals + [0.0])[0]
    return [ComponentSpec("C", "series", C_val, None, a, b)]


MACRO_LIBRARY: Dict[str, MacroDef] = {
    MACRO_CELL_LS_CS: MacroDef(MACRO_CELL_LS_CS, ("L", "C"), _expand_cell_ls_cs),
    MACRO_CELL_CS_LS: MacroDef(MACRO_CELL_CS_LS, ("C", "L"), _expand_cell_cs_ls),
    MACRO_NOTCH_SHUNT_LC_SER: MacroDef(MACRO_NOTCH_SHUNT_LC_SER, ("L", "C"), _expand_notch_shunt),
    MACRO_CELL_LS: MacroDef(MACRO_CELL_LS, ("L",), _expand_cell_ls),
    MACRO_CELL_CS: MacroDef(MACRO_CELL_CS, ("C",), _expand_cell_cs),
    MACRO_PI_CLC: MacroDef(MACRO_PI_CLC, ("C", "L", "C"), _expand_pi_clc),
    MACRO_T_LCL: MacroDef(MACRO_T_LCL, ("L", "C", "L"), _expand_t_lcl),
    MACRO_DOUBLE_SERIES_LC: MacroDef(MACRO_DOUBLE_SERIES_LC, ("L", "C"), _expand_double_series_lc),
    MACRO_DOUBLE_SHUNT_LC: MacroDef(MACRO_DOUBLE_SHUNT_LC, ("L", "C"), _expand_double_shunt_lc),
    MACRO_BRIDGE_C: MacroDef(MACRO_BRIDGE_C, ("C",), _expand_bridge_c),
}

# Slot type -> token mapping (kept centralized to avoid drift across encoder/decoder/mask).
SLOT_TYPE_TO_TOKEN = {
    "L": VAL_L,
    "C": VAL_C,
    "R": VAL_R,
    "TL_Z0": VAL_TL_Z0,
    "TL_LEN": VAL_TL_LEN,
}


# ---- Vocab builder ----


def build_dslv2_vocab(
    *,
    macro_ids: Sequence[str] | None = None,
    include_bos: bool = True,
) -> List[str]:
    vocab: Set[str] = set()
    vocab.update([MAIN_START, MAIN_END, REPEAT_START, REPEAT_END, CASCADE, CALL])
    vocab.update([CELL])
    vocab.update(CELL_INDEX_TOKENS)
    vocab.update([PORT_IN, PORT_OUT, PORT_GND])
    vocab.update(K_TOKENS)
    vocab.update([K_VAR_START, K_VAR_END])
    vocab.update(DIGIT_TOKENS)
    vocab.update(VALUE_SLOTS)
    vocab.update(macro_ids or MACRO_IDS)
    # EOS is always included because encoder unconditionally appends it.
    vocab.update([EOS])
    if include_bos:
        vocab.update([BOS])
    return sorted(vocab)


# ---- Encoding (components -> tokens) ----


def components_to_dslv2_tokens(
    components: Sequence[ComponentSpec],
    *,
    macro_name: str = MACRO_CELL_LS_CS,
    segments: Sequence[Tuple[str, Sequence[Sequence[float]]]] | None = None,
    use_varint_k: bool = False,
    use_cell_indices: bool = False,
    include_bos: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Convert components (or explicit macro segments) into DSL v2 tokens.
    Returns (tokens, slot_values) where slot_values aligns to tokens (nan when not a slot).
    - segments: optional explicit [(macro_name, [cell_vals...]), ...]; if provided, components are ignored.
    - use_varint_k: encode K with <K> D_* </K> to allow large/unbounded repeat counts.
    - use_cell_indices: emit <CELL_IDX_i> after <CELL> to stabilize long-K decoding.
    """
    def _encode_k(k: int, tokens_out: List[str], slot_out: List[float]) -> None:
        # Prefer varint when requested or k exceeds frozen set.
        if use_varint_k or k > len(K_TOKENS):
            tokens_out.append(K_VAR_START)
            slot_out.append(float("nan"))
            for d in str(max(0, int(k))):
                tok = f"<D_{d}>"
                tokens_out.append(tok)
                slot_out.append(float("nan"))
            tokens_out.append(K_VAR_END)
            slot_out.append(float("nan"))
        else:
            tokens_out.append(f"<K_{k}>")
            slot_out.append(float("nan"))

    tokens: List[str] = []
    if include_bos:
        tokens.append(BOS)
    tokens.extend([MAIN_START, PORT_IN, PORT_OUT, PORT_GND])
    slot_values: List[float] = [float("nan")] * len(tokens)

    def _emit_repeat(macro_id: str, cell_vals: Sequence[Sequence[float]]):
        macro = MACRO_LIBRARY[macro_id]
        k = max(1, len(cell_vals))
        tokens.extend([REPEAT_START])
        slot_values.append(float("nan"))
        _encode_k(k, tokens, slot_values)
        tokens.extend([CASCADE, CALL, macro_id])
        slot_values.extend([float("nan")] * 3)
        for idx_cell in range(k):
            vals = list(cell_vals[idx_cell]) if idx_cell < len(cell_vals) else [0.0] * len(macro.slot_types)
            vals = (vals + [0.0] * len(macro.slot_types))[: len(macro.slot_types)]
            tokens.append(CELL)
            slot_values.append(float("nan"))
            if use_cell_indices and idx_cell < len(CELL_INDEX_TOKENS):
                tokens.append(CELL_INDEX_TOKENS[idx_cell])
                slot_values.append(float("nan"))
            for slot_idx, slot_type in enumerate(macro.slot_types):
                if slot_type not in SLOT_TYPE_TO_TOKEN:
                    raise ValueError(f"Unknown slot type '{slot_type}' in macro '{macro_id}'")
                t = SLOT_TYPE_TO_TOKEN[slot_type]
                tokens.append(t)
                slot_values.append(vals[slot_idx])
        tokens.extend([REPEAT_END])
        slot_values.append(float("nan"))

    def _emit_call(macro_id: str, vals: Sequence[float]):
        macro = MACRO_LIBRARY[macro_id]
        tokens.extend([CALL, macro_id])
        slot_values.extend([float("nan")] * 2)
        vs = list(vals) + [0.0] * len(macro.slot_types)
        vs = vs[: len(macro.slot_types)]
        for slot_idx, slot_type in enumerate(macro.slot_types):
            if slot_type not in SLOT_TYPE_TO_TOKEN:
                raise ValueError(f"Unknown slot type '{slot_type}' in macro '{macro_id}'")
            t = SLOT_TYPE_TO_TOKEN[slot_type]
            tokens.append(t)
            slot_values.append(vs[slot_idx])

    if segments is not None:
        for macro_id, cell_vals in segments:
            cell_vals = list(cell_vals)
            if len(cell_vals) <= 1:
                _emit_call(macro_id, cell_vals[0] if cell_vals else [])
            else:
                _emit_repeat(macro_id, cell_vals)
    else:
        macro = MACRO_LIBRARY[macro_name]
        # Heuristic: derive K from number of series components along main path (non-gnd).
        series = [c for c in components if c.role == "series" and c.node1 != "gnd" and c.node2 != "gnd"]
        K = max(1, len(series))
        # Collect slot values per cell from components: greedy by series/shunt ordering.
        vals_per_cell: List[List[float]] = []
        shunt_map: Dict[str, ComponentSpec] = {c.node1: c for c in components if c.role == "shunt"}
        for c in series:
            vals = []
            for t in macro.slot_types:
                if t == "L":
                    if c.ctype == "L":
                        vals.append(float(c.value_si))
                    else:
                        sh = shunt_map.get(c.node2)
                        vals.append(float(sh.value_si) if sh and sh.ctype == "L" else 0.0)
                elif t == "C":
                    if c.ctype == "C":
                        vals.append(float(c.value_si))
                    else:
                        sh = shunt_map.get(c.node2)
                        vals.append(float(sh.value_si) if sh and sh.ctype == "C" else 0.0)
                else:
                    vals.append(0.0)
            vals_per_cell.append(vals)

        while len(vals_per_cell) < K:
            vals_per_cell.append([0.0] * len(macro.slot_types))
        vals_per_cell = vals_per_cell[:K]

        _emit_repeat(macro_name, vals_per_cell)

    tokens.append(MAIN_END)
    slot_values.append(float("nan"))
    tokens.append(EOS)
    slot_values.append(float("nan"))
    return tokens, slot_values


# ---- Decoding (tokens -> components) ----


def dslv2_tokens_to_components(
    tokens: Sequence[str],
    *,
    slot_values: Sequence[float] | None = None,
) -> List[ComponentSpec]:
    """
    Parse DSL v2 tokens into ComponentSpec list. slot_values should align with tokens;
    if absent, slots are filled with 0.0.
    """
    toks = list(tokens)
    vals = list(slot_values) if slot_values is not None else [float("nan")] * len(toks)
    idx = 0

    def _next():
        nonlocal idx
        tok = toks[idx] if idx < len(toks) else None
        val = vals[idx] if idx < len(vals) else float("nan")
        idx += 1
        return tok, val

    tok, _ = _next()
    if tok == BOS:
        tok, _ = _next()
    if tok != MAIN_START:
        return []
    for expected in (PORT_IN, PORT_OUT, PORT_GND):
        tok, _ = _next()
        if tok != expected:
            return []

    segments: List[Tuple[str, List[float]]] = []
    while True:
        tok, _ = _next()
        if tok is None or tok == MAIN_END:
            break
        if tok == REPEAT_START:
            tok_k, _ = _next()
            if tok_k == K_VAR_START:
                digits: List[str] = []
                while True:
                    tok_digit, _ = _next()
                    if tok_digit is None:
                        return []
                    if tok_digit == K_VAR_END:
                        break
                    if tok_digit not in DIGIT_TOKENS:
                        return []
                    digits.append(tok_digit.removeprefix("<D_").removesuffix(">"))
                try:
                    k_val = max(1, int("".join(digits))) if digits else 1
                except Exception:
                    return []
            elif tok_k in K_TOKENS:
                try:
                    k_val = int(tok_k.removeprefix("<K_").removesuffix(">"))
                except Exception:
                    return []
            else:
                return []
            tok_c, _ = _next()
            if tok_c != CASCADE:
                return []
            tok_call, _ = _next()
            if tok_call != CALL:
                return []
            tok_macro, _ = _next()
            if tok_macro not in MACRO_LIBRARY:
                return []
            macro = MACRO_LIBRARY[tok_macro]
            # New canonical form: each cell starts with <CELL>, then the macro's slot tokens.
            # Backward compatible: allow the older form without <CELL> boundaries.
            use_cell_tokens = (idx < len(toks) and toks[idx] == CELL)
            for _ in range(k_val):
                if use_cell_tokens:
                    tok_cell, _ = _next()
                    if tok_cell != CELL:
                        return []
                    # optional cell index markers
                    while idx < len(toks) and toks[idx] in CELL_INDEX_TOKENS:
                        _next()
                vals_for_macro: List[float] = []
                for slot_type in macro.slot_types:
                    tok_slot, v_slot = _next()
                    while tok_slot in CELL_INDEX_TOKENS:
                        tok_slot, v_slot = _next()
                    expected_tok = SLOT_TYPE_TO_TOKEN.get(slot_type)
                    if expected_tok is None or tok_slot != expected_tok:
                        return []
                    vals_for_macro.append(float(v_slot) if v_slot == v_slot else 0.0)
                segments.append((tok_macro, vals_for_macro))
            tok_end, _ = _next()
            if tok_end != REPEAT_END:
                return []
        elif tok == CALL:
            tok_macro, _ = _next()
            if tok_macro not in MACRO_LIBRARY:
                return []
            macro = MACRO_LIBRARY[tok_macro]
            vals_for_macro: List[float] = []
            for slot_type in macro.slot_types:
                tok_slot, v_slot = _next()
                while tok_slot in CELL_INDEX_TOKENS:
                    tok_slot, v_slot = _next()
                expected_tok = SLOT_TYPE_TO_TOKEN.get(slot_type)
                if expected_tok is None or tok_slot != expected_tok:
                    return []
                vals_for_macro.append(float(v_slot) if v_slot == v_slot else 0.0)
            segments.append((tok_macro, vals_for_macro))
        else:
            return []

    if not segments:
        return []

    comps: List[ComponentSpec] = []
    nodes: List[str] = ["in"]
    last_end = "in"
    for i, (macro_name, vals_for_macro) in enumerate(segments):
        a = nodes[-1]
        b = "out" if i == len(segments) - 1 else f"n{i+1}"
        if b != "out":
            nodes.append(b)
        macro = MACRO_LIBRARY[macro_name]
        comps.extend(macro.expand_fn(a, b, "gnd", vals_for_macro, i))
        last_end = b
    # ensure last node renamed to out if not already
    if last_end != "out":
        for idx_c, c in enumerate(comps):
            n1 = "out" if c.node1 == last_end else c.node1
            n2 = "out" if c.node2 == last_end else c.node2
            comps[idx_c] = ComponentSpec(c.ctype, c.role, c.value_si, c.std_label, n1, n2)
    return comps


# ---- Grammar mask ----


def make_dslv2_prefix_allowed_tokens_fn(tokenizer) -> Callable[[int, List[int]], List[int]]:
    vocab = tokenizer.get_vocab()
    all_ids = list(vocab.values())

    def _id(tok: str) -> int | None:
        return vocab.get(tok)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # precompute sets
    macro_ids = {_id(tok) for tok in MACRO_IDS if _id(tok) is not None}
    k_ids = {_id(tok) for tok in K_TOKENS if _id(tok) is not None}
    k_var_start_id = _id(K_VAR_START)
    k_var_end_id = _id(K_VAR_END)
    digit_ids = {_id(tok) for tok in DIGIT_TOKENS if _id(tok) is not None}
    cell_idx_ids = {_id(tok) for tok in CELL_INDEX_TOKENS if _id(tok) is not None}
    val_slot_ids = {_id(tok) for tok in VALUE_SLOTS if _id(tok) is not None}
    k_id_to_val = {_id(tok): int(tok.removeprefix("<K_").removesuffix(">")) for tok in K_TOKENS if _id(tok) is not None}
    macro_slots_len: Dict[int, int] = {}
    macro_slots_order: Dict[int, List[int]] = {}
    slot_type_to_id: Dict[str, int] = {}
    for slot_type, slot_tok in SLOT_TYPE_TO_TOKEN.items():
        tid = _id(slot_tok)
        if tid is not None:
            slot_type_to_id[slot_type] = tid
    for tok in MACRO_IDS:
        tid = _id(tok)
        if tid is None:
            continue
        macro_slots_len[tid] = len(MACRO_LIBRARY[tok].slot_types)
        ordered: List[int] = []
        for st in MACRO_LIBRARY[tok].slot_types:
            tok_id = slot_type_to_id.get(st)
            if tok_id is None:
                # Tokenizer vocab drift: fail open rather than silently making everything invalid.
                return lambda batch_id, input_ids: all_ids
            ordered.append(tok_id)
        macro_slots_order[tid] = ordered

    main_start = _id(MAIN_START)
    main_end = _id(MAIN_END)
    repeat_start = _id(REPEAT_START)
    repeat_end = _id(REPEAT_END)
    cascade_id = _id(CASCADE)
    call_id = _id(CALL)
    cell_id = _id(CELL)
    port_order = [_id(PORT_IN), _id(PORT_OUT), _id(PORT_GND)]
    bos_id = _id(BOS)
    explicit_eos_id = _id(EOS)

    required = [main_start, main_end, repeat_start, repeat_end, cascade_id, call_id, cell_id]
    if any(r is None for r in required) or not macro_ids or not k_ids:
        return lambda batch_id, input_ids: all_ids

    S_START = 0
    S_AFTER_BOS = 1
    S_PORTS = 2
    S_BODY = 3
    S_IN_REPEAT = 4
    S_AFTER_K = 5
    S_AFTER_CASCADE = 6
    S_AFTER_CALL = 7
    S_IN_SLOTS = 8
    S_EXPECT_REPEAT_END = 9
    S_AFTER_MAIN_END = 10
    S_AFTER_EXPLICIT_EOS = 11
    S_IN_K_VAR = 13
    S_DONE = 14

    def _state_from_ids(input_ids: List[int]) -> Tuple[int, int, int, int | None, bool, bool, int, int, bool, bool, bool]:
        state = S_START
        slot_needed = 0
        slot_pos = 0
        current_macro_tid: int | None = None
        valid = True
        port_idx = 0
        in_repeat = False
        current_k = 1
        has_segment = False
        in_k_var = False
        k_digits_seen = False
        for tid in input_ids:
            if tid in special_ids:
                continue
            if tid in cell_idx_ids:
                continue
            if state == S_START:
                if tid == bos_id:
                    state = S_AFTER_BOS
                    continue
                if tid == main_start:
                    state = S_PORTS
                    continue
                valid = False
                break
            if state == S_AFTER_BOS:
                if tid == main_start:
                    state = S_PORTS
                    continue
                valid = False
                break
            if state == S_PORTS:
                expected = port_order[port_idx] if 0 <= port_idx < len(port_order) else None
                if expected is not None and tid == expected:
                    port_idx += 1
                    if port_idx >= len(port_order):
                        state = S_BODY
                    continue
                valid = False
                break
            if state == S_BODY:
                if tid == repeat_start:
                    state = S_IN_REPEAT
                    in_repeat = True
                    has_segment = True
                    continue
                if tid == call_id:
                    state = S_AFTER_CALL
                    has_segment = True
                    continue
                if tid == main_end:
                    state = S_AFTER_MAIN_END
                    continue
                valid = False
                break
            if state == S_IN_REPEAT:
                if tid in k_ids:
                    current_k = k_id_to_val.get(tid, 1)
                    state = S_AFTER_K
                    continue
                if tid == k_var_start_id:
                    in_k_var = True
                    k_digits_seen = False
                    current_k = 0
                    state = S_IN_K_VAR
                    continue
                valid = False
                break
            if state == S_IN_K_VAR:
                if tid in digit_ids:
                    k_digits_seen = True
                    digit = 0
                    for k, v in vocab.items():
                        if v == tid and k.startswith("<D_"):
                            try:
                                digit = int(k.removeprefix("<D_").removesuffix(">"))
                            except Exception:
                                digit = 0
                            break
                    current_k = current_k * 10 + digit
                    continue
                if tid == k_var_end_id and k_digits_seen:
                    state = S_AFTER_K
                    in_k_var = False
                    continue
                valid = False
                break
            if state == S_AFTER_K:
                if tid == cascade_id:
                    state = S_AFTER_CASCADE
                    continue
                valid = False
                break
            if state == S_AFTER_CASCADE:
                if tid == call_id:
                    state = S_AFTER_CALL
                    continue
                valid = False
                break
            if state == S_AFTER_CALL:
                if tid in macro_ids:
                    macro_len = macro_slots_len.get(tid, 0)
                    rep = current_k if in_repeat else 1
                    # Repeat slots are grouped as: (<CELL> + slot_types) * K
                    slot_needed = (1 + macro_len) * rep if in_repeat else macro_len
                    slot_pos = 0
                    current_macro_tid = tid
                    state = S_IN_SLOTS if slot_needed > 0 else (S_EXPECT_REPEAT_END if in_repeat else S_BODY)
                    continue
                valid = False
                break
            if state == S_IN_SLOTS:
                expected_slots = macro_slots_order.get(current_macro_tid, [])
                macro_len = len(expected_slots)
                if in_repeat:
                    group_len = 1 + macro_len
                    pos_in_group = slot_pos % group_len
                    expected_tok = cell_id if pos_in_group == 0 else (expected_slots[pos_in_group - 1] if expected_slots else None)
                else:
                    expected_tok = expected_slots[slot_pos] if 0 <= slot_pos < macro_len else None
                if expected_tok is not None and tid == expected_tok:
                    slot_needed -= 1
                    slot_pos += 1
                    if slot_needed <= 0:
                        state = S_EXPECT_REPEAT_END if in_repeat else S_BODY
                    continue
                valid = False
                break
            if state == S_EXPECT_REPEAT_END:
                if tid == repeat_end:
                    if slot_needed <= 0:
                        state = S_BODY
                        in_repeat = False
                        current_k = 1
                    continue
                valid = False
                break
            if state == S_AFTER_MAIN_END:
                if explicit_eos_id is not None and tid == explicit_eos_id:
                    state = S_AFTER_EXPLICIT_EOS
                    continue
                if eos_id is not None and tid == eos_id:
                    state = S_DONE
                    continue
                valid = False
                break
            if state == S_AFTER_EXPLICIT_EOS:
                if eos_id is not None and tid == eos_id:
                    state = S_DONE
                    continue
                valid = False
                break
            if state == S_DONE:
                continue
        return state, slot_needed, port_idx, current_macro_tid, valid, in_repeat, current_k, slot_pos, has_segment, in_k_var, k_digits_seen

    def _prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        (
            state,
            slot_needed,
            port_idx,
            current_macro_tid,
            valid,
            in_repeat,
            current_k,
            slot_pos,
            has_segment,
            in_k_var,
            k_digits_seen,
        ) = _state_from_ids(ids)
        if not valid:
            return all_ids
        allowed: Set[int] = set()
        if state == S_START:
            if main_start is not None:
                allowed.add(main_start)
            if bos_id is not None:
                allowed.add(bos_id)
        elif state == S_AFTER_BOS:
            if main_start is not None:
                allowed.add(main_start)
        elif state == S_PORTS:
            expected = port_order[port_idx] if 0 <= port_idx < len(port_order) else None
            if expected is not None:
                allowed.add(expected)
        elif state == S_BODY:
            if repeat_start is not None:
                allowed.add(repeat_start)
            if call_id is not None:
                allowed.add(call_id)
            if has_segment and main_end is not None:
                allowed.add(main_end)
        elif state == S_IN_REPEAT:
            allowed.update(k_ids)
            if k_var_start_id is not None:
                allowed.add(k_var_start_id)
        elif state == S_IN_K_VAR:
            if digit_ids:
                allowed.update(digit_ids)
            if k_digits_seen and k_var_end_id is not None:
                allowed.add(k_var_end_id)
        elif state == S_AFTER_K:
            if cascade_id is not None:
                allowed.add(cascade_id)
        elif state == S_AFTER_CASCADE:
            if call_id is not None:
                allowed.add(call_id)
        elif state == S_AFTER_CALL:
            allowed.update(macro_ids)
        elif state == S_IN_SLOTS:
            expected_slots = macro_slots_order.get(current_macro_tid, [])
            macro_len = len(expected_slots)
            if slot_needed <= 0:
                expected_tok = None
            elif in_repeat:
                group_len = 1 + macro_len
                pos_in_group = slot_pos % group_len
                expected_tok = cell_id if pos_in_group == 0 else (expected_slots[pos_in_group - 1] if expected_slots else None)
            else:
                expected_tok = expected_slots[slot_pos] if 0 <= slot_pos < macro_len else None
            if expected_tok is not None:
                allowed.add(expected_tok)
            if in_repeat and cell_idx_ids:
                allowed.update(cell_idx_ids)
        elif state == S_EXPECT_REPEAT_END:
            if repeat_end is not None:
                allowed.add(repeat_end)
        elif state == S_AFTER_MAIN_END:
            if explicit_eos_id is not None:
                allowed.add(explicit_eos_id)
            if eos_id is not None:
                allowed.add(eos_id)
        elif state == S_AFTER_EXPLICIT_EOS:
            if eos_id is not None:
                allowed.add(eos_id)
        elif state == S_DONE:
            if eos_id is not None:
                allowed.add(eos_id)
        if pad_id is not None:
            allowed.add(pad_id)
        if not allowed:
            return all_ids
        return sorted(allowed)

    return _prefix_allowed_tokens_fn


def dslv2_tokens_to_action_tokens(
    tokens: Sequence[str],
    *,
    slot_values: Sequence[float] | None = None,
) -> List[str]:
    """
    Convenience: DSL v2 tokens (+slots) -> ComponentSpec -> Action tokens.
    Useful for dual-IR supervision or post-hoc compilation to ActionVACT.
    """
    comps = dslv2_tokens_to_components(tokens, slot_values=slot_values)
    return components_to_action_tokens(
        [
            ComponentSpec(c.ctype, c.role, c.value_si, c.std_label, c.node1, c.node2)
            for c in comps
        ]
    )
