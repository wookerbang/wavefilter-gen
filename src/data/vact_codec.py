"""
电路 <-> 序列化 token（VACT，component-centric）。

设计目标：
- component-centric，顺序沿主路 in→...→out（生成阶段已是 canonical）。
- 每个元件用独立的 token 片段，便于 tokenizer 直接消费。
- 每个元件 5 个 token：<TYPE> <ROLE> <VAL_xxx> <NODE_n1> <NODE_n2>
- 类型/角色大小写与 ComponentSpec 一致：ctype in {"L","C"}, role in {"series","shunt"}。
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Set, Tuple

from .schema import ComponentSpec


def _node_order(node: str) -> int:
    """粗略的节点顺序，仅用于确保序列唯一。"""
    if node == "in":
        return 0
    if node.startswith("n"):
        try:
            return 1 + int(node[1:])
        except ValueError:
            return 1_000
    if node == "out":
        return 10_000
    if node == "gnd":
        return 20_000
    return 30_000  # others


def _canonicalize(components: Iterable[ComponentSpec]) -> List[ComponentSpec]:
    return sorted(
        components,
        key=lambda c: (
            _node_order(c.node1),
            0 if c.role == "series" else 1,  # series before shunt on same node
            c.ctype,
        ),
    )


def _type_token(ctype: str) -> str:
    return f"<{ctype.upper()}>"


def _role_token(role: str) -> str:
    return f"<{role.upper()}>"


def _label_token(label: str | None) -> str:
    # 保留原始 label，避免破坏单位前缀大小写
    return f"<VAL_{label or 'NA'}>"


def _node_token(node: str) -> str:
    return f"<NODE_{node}>"


def components_to_vact_tokens(components: List[ComponentSpec]) -> List[str]:
    """
    将离散化元件编码为 token 序列。
    每个元件展开为 5 个 token：
      <L/C> <SERIES/SHUNT> <VAL_xxx> <NODE_n1> <NODE_n2>
    """
    tokens: List[str] = []
    for comp in _canonicalize(components):
        tokens.extend(
            [
                _type_token(comp.ctype),
                _role_token(comp.role),
                _label_token(comp.std_label),
                _node_token(comp.node1),
                _node_token(comp.node2),
            ]
        )
    return tokens


def vact_tokens_to_components(tokens: List[str], label_to_value: Mapping[str, float] | None = None) -> List[ComponentSpec]:
    """
    反向解析 token 序列，按 5-token 一组还原 ComponentSpec。
    如果提供 label_to_value，会将 std_label 映射为 value_si。
    """
    comps: List[ComponentSpec] = []
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


def build_vact_vocab(
    value_labels: Sequence[str],
    node_names: Sequence[str] = ("in", "out", "gnd") + tuple(f"n{k}" for k in range(16)),
    order_range: Tuple[int, int] | None = (2, 7),
) -> List[str]:
    """
    构建用于 tokenizer 的 VACT 词表，覆盖类型/角色/数值/节点 token（无 ID）。
    - value_labels: 例如 ['L_3.3nH', 'C_4.7pF', ...]
    - node_names: 允许的节点名称集合
    """
    vocab: Set[str] = set()
    vocab.update([_type_token("L"), _type_token("C"), _role_token("series"), _role_token("shunt")])
    vocab.add("<SEP>")
    if order_range:
        lo, hi = order_range
        for k in range(lo, hi + 1):
            vocab.add(f"<ORDER_{k}>")
    for label in value_labels:
        vocab.add(_label_token(label))
    for node in node_names:
        vocab.add(_node_token(node))
    return sorted(vocab)
