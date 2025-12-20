"""
电路 <-> 序列化 token（VACT，component-centric）。

设计目标：
- component-centric，顺序沿主路 in→...→out（生成阶段已是 canonical）。
- 每个元件用独立的 token 片段，便于 tokenizer 直接消费。
- 每个元件 5 个 token：<TYPE> <ROLE> <VAL_xxx> <NODE_n1> <NODE_n2>
- 类型/角色大小写与 ComponentSpec 一致：ctype in {"L","C"}, role in {"series","shunt"}。
- optional <CELL> markers to segment the sequence into sections.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Mapping, Sequence, Set, Tuple

from .schema import ComponentSpec

CELL_TOKEN = "<CELL>"
SEP_TOKEN = "<SEP>"
ORDER_PREFIX = "<ORDER_"
_NODE_PREFIX_RE = re.compile(r"^n(\d+)")
_X_NODE_PREFIX_RE = re.compile(r"^x(\d+)(?:_(\d+))?$")


def _node_order(node: str) -> int:
    """粗略的节点顺序，仅用于确保序列唯一。"""
    if node == "in":
        return 0
    if node == "out":
        return 10_000
    if node == "gnd":
        return 20_000
    if node.startswith("x"):
        match = _X_NODE_PREFIX_RE.match(node)
        if match:
            # Internal nodes allocated by the DSL compiler: x{i}_{j}.
            # Keep them near their owning stage index i so canonicalization stays local.
            return 1 + int(match.group(1))
        return 1_500
    if node.startswith("n"):
        match = _NODE_PREFIX_RE.match(node)
        if match:
            return 1 + int(match.group(1))
        return 1_000
    return 30_000  # others


def _node_sort_key(node: str) -> Tuple[int, str]:
    return (_node_order(node), node)


def _normalize_node_pair(node1: str, node2: str) -> Tuple[str, str]:
    return (node1, node2) if _node_sort_key(node1) <= _node_sort_key(node2) else (node2, node1)


def _canonicalize(components: Iterable[ComponentSpec]) -> List[ComponentSpec]:
    def _key(comp: ComponentSpec) -> Tuple[int, int, str, Tuple[int, str], Tuple[int, str]]:
        n1, n2 = _normalize_node_pair(comp.node1, comp.node2)
        return (
            min(_node_order(n1), _node_order(n2)),
            0 if comp.role == "series" else 1,  # series before shunt on same anchor
            comp.ctype,
            _node_sort_key(n1),
            _node_sort_key(n2),
        )

    return sorted(components, key=_key)


def _type_token(ctype: str) -> str:
    return f"<{ctype.upper()}>"


def _role_token(role: str) -> str:
    return f"<{role.upper()}>"


def _label_token(label: str | None) -> str:
    # 保留原始 label，避免破坏单位前缀大小写
    return f"<VAL_{label or 'NA'}>"


def _node_token(node: str) -> str:
    return f"<NODE_{node}>"


def components_to_vact_tokens(
    components: List[ComponentSpec],
    *,
    emit_cell_tokens: bool = False,
    normalize_node_order: bool = True,
) -> List[str]:
    """
    将离散化元件编码为 token 序列。
    每个元件展开为 5 个 token：
      <L/C> <SERIES/SHUNT> <VAL_xxx> <NODE_n1> <NODE_n2>
    """
    tokens: List[str] = []
    prev_anchor = None
    for comp in _canonicalize(components):
        node1, node2 = (comp.node1, comp.node2)
        if normalize_node_order:
            node1, node2 = _normalize_node_pair(comp.node1, comp.node2)
        anchor = min(_node_order(node1), _node_order(node2))
        if emit_cell_tokens and (prev_anchor is None or anchor != prev_anchor):
            tokens.append(CELL_TOKEN)
            prev_anchor = anchor
        tokens.extend(
            [
                _type_token(comp.ctype),
                _role_token(comp.role),
                _label_token(comp.std_label),
                _node_token(node1),
                _node_token(node2),
            ]
        )
    return tokens


def _strip_vact_non_component_tokens(tokens: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for tok in tokens:
        if tok == CELL_TOKEN:
            continue
        if tok == SEP_TOKEN or tok.startswith(ORDER_PREFIX):
            continue
        if tok in {"<pad>", "</s>", "<unk>"}:
            continue
        cleaned.append(tok)
    return cleaned


def vact_tokens_to_components(
    tokens: List[str],
    label_to_value: Mapping[str, float] | None = None,
    *,
    drop_non_component_tokens: bool = True,
) -> List[ComponentSpec]:
    """
    反向解析 token 序列，按 5-token 一组还原 ComponentSpec。
    如果提供 label_to_value，会将 std_label 映射为 value_si。
    """
    comps: List[ComponentSpec] = []
    if drop_non_component_tokens:
        tokens = _strip_vact_non_component_tokens(tokens)
    if len(tokens) % 5 != 0:
        tokens = tokens[: len(tokens) // 5 * 5]

    for i in range(0, len(tokens), 5):
        t_type, t_role, t_val, t_n1, t_n2 = tokens[i : i + 5]
        ctype = t_type.strip("<>").upper()
        role = t_role.strip("<>").lower()
        label = t_val.strip("<>")
        if label == "VAL_NA":
            label = None
        else:
            label = label.replace("VAL_", "", 1)
        node1 = t_n1.replace("<NODE_", "").replace(">", "")
        node2 = t_n2.replace("<NODE_", "").replace(">", "")
        value = 0.0
        if label is not None:
            if label_to_value is not None:
                value = float(label_to_value.get(label, 0.0))
            else:
                try:
                    from .quantization import label_to_value as _label_to_value
                    value = float(_label_to_value(label))
                except Exception:
                    value = 0.0
        comps.append(ComponentSpec(ctype=ctype, role=role, value_si=value, std_label=label, node1=node1, node2=node2))
    return comps


def make_vact_syntax_prefix_allowed_tokens_fn(
    tokenizer,
    *,
    allow_cell_token: bool = True,
    allow_order_prefix: bool = True,
) -> Callable[[int, List[int]], List[int]]:
    """
    Build a prefix_allowed_tokens_fn that enforces low-level VACT grammar only:
    TYPE -> ROLE -> VAL -> NODE -> NODE, with optional prefix <ORDER_k> <SEP> and <CELL>.
    """
    vocab = tokenizer.get_vocab()
    all_ids = list(vocab.values())

    type_ids = {vocab[tok] for tok in ("<L>", "<C>") if tok in vocab}
    role_ids = {vocab[tok] for tok in ("<SERIES>", "<SHUNT>") if tok in vocab}
    val_ids = {tid for tok, tid in vocab.items() if tok.startswith("<VAL_")}
    node_ids = {tid for tok, tid in vocab.items() if tok.startswith("<NODE_")}
    order_ids = {tid for tok, tid in vocab.items() if tok.startswith(ORDER_PREFIX)}
    sep_id = vocab.get(SEP_TOKEN)
    cell_id = vocab.get(CELL_TOKEN) if allow_cell_token else None

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    def _state_from_ids(input_ids: List[int]) -> Tuple[int, bool, bool]:
        state = 0  # 0:type, 1:role, 2:val, 3:node1, 4:node2
        started = False
        valid = True
        for tid in input_ids:
            if tid in special_ids:
                continue
            if state == 0:
                if allow_order_prefix and not started and tid in order_ids:
                    continue
                if allow_order_prefix and not started and sep_id is not None and tid == sep_id:
                    continue
                if allow_cell_token and cell_id is not None and tid == cell_id:
                    continue
                if tid in type_ids:
                    state = 1
                    started = True
                    continue
                valid = False
                break
            elif state == 1:
                if tid in role_ids:
                    state = 2
                    continue
                valid = False
                break
            elif state == 2:
                if tid in val_ids:
                    state = 3
                    continue
                valid = False
                break
            elif state == 3:
                if tid in node_ids:
                    state = 4
                    continue
                valid = False
                break
            elif state == 4:
                if tid in node_ids:
                    state = 0
                    continue
                valid = False
                break
        return state, started, valid

    def _prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        state, started, valid = _state_from_ids(ids)
        if not valid:
            return all_ids

        allowed: Set[int] = set()
        if state == 0:
            allowed.update(type_ids)
            if allow_cell_token and cell_id is not None:
                allowed.add(cell_id)
            if allow_order_prefix and not started:
                allowed.update(order_ids)
                if sep_id is not None:
                    allowed.add(sep_id)
            if eos_id is not None:
                allowed.add(eos_id)
        elif state == 1:
            allowed.update(role_ids)
        elif state == 2:
            allowed.update(val_ids)
        elif state in (3, 4):
            allowed.update(node_ids)

        if pad_id is not None:
            allowed.add(pad_id)
        return sorted(allowed)

    return _prefix_allowed_tokens_fn


def build_vact_vocab(
    value_labels: Sequence[str],
    node_names: Sequence[str] = ("in", "out", "gnd") + tuple(f"n{k}" for k in range(16)),
    order_range: Tuple[int, int] | None = (2, 7),
    include_cell_token: bool = True,
) -> List[str]:
    """
    构建用于 tokenizer 的 VACT 词表，覆盖类型/角色/数值/节点 token（无 ID）。
    - value_labels: 例如 ['L_3.3nH', 'C_4.7pF', ...]
    - node_names: 允许的节点名称集合
    """
    vocab: Set[str] = set()
    vocab.update([_type_token("L"), _type_token("C"), _role_token("series"), _role_token("shunt")])
    vocab.add(SEP_TOKEN)
    if order_range:
        lo, hi = order_range
        for k in range(lo, hi + 1):
            vocab.add(f"{ORDER_PREFIX}{k}>")
    if include_cell_token:
        vocab.add(CELL_TOKEN)
    for label in value_labels:
        vocab.add(_label_token(label))
    for node in node_names:
        vocab.add(_node_token(node))
    return sorted(vocab)
