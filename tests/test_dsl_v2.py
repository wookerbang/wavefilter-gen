import math
import random
import unittest

from src.data.dsl_v2 import (
    BOS,
    CASCADE,
    CALL,
    CALL,
    CELL,
    CELL_INDEX_TOKENS,
    DIGIT_TOKENS,
    EOS,
    K_VAR_START,
    K_VAR_END,
    K_TOKENS,
    MACRO_CELL_CS,
    MACRO_CELL_CS_LS,
    MACRO_CELL_LS,
    MACRO_CELL_LS_CS,
    MACRO_PI_CLC,
    MACRO_LIBRARY,
    MAIN_END,
    MAIN_START,
    PORT_GND,
    PORT_IN,
    PORT_OUT,
    REPEAT_END,
    REPEAT_START,
    SLOT_TYPE_TO_TOKEN,
    VAL_C,
    VAL_L,
    components_to_dslv2_tokens,
    dslv2_tokens_to_components,
    dslv2_tokens_to_action_tokens,
    make_dslv2_prefix_allowed_tokens_fn,
    MacroDef,
)
from src.data.schema import ComponentSpec


class DummyTokenizer:
    def __init__(self, tokens):
        self._vocab = {tok: i for i, tok in enumerate(tokens)}
        self.all_special_ids = []
        self.pad_token_id = self._vocab.get("<pad>")
        self.eos_token_id = self._vocab.get(EOS)

    def get_vocab(self):
        return self._vocab


class DSLv2Tests(unittest.TestCase):
    def test_unknown_slot_type_raises(self):
        macro_name = "<MAC_BOGUS_X>"
        bogus_macro = MacroDef(macro_name, ("X",), lambda a, b, g, vals, idx=0: [])
        components = [ComponentSpec("L", "series", 1.0, None, "in", "out")]
        # Insert bogus macro temporarily.
        MACRO_LIBRARY[macro_name] = bogus_macro
        try:
            with self.assertRaises(ValueError):
                components_to_dslv2_tokens(components, macro_name=macro_name)
        finally:
            MACRO_LIBRARY.pop(macro_name, None)

    def test_repeat_round_trip_with_and_without_cell_markers(self):
        rng = random.Random(0)
        macros = [MACRO_CELL_LS_CS, MACRO_CELL_CS_LS, MACRO_CELL_LS, MACRO_CELL_CS]
        num_trials = 1000
        for with_cells in (True, False):
            for _ in range(num_trials):
                macro_name = rng.choice(macros)
                macro = MACRO_LIBRARY[macro_name]
                k = rng.randint(1, 12)
                comps = []
                nodes = ["in"]
                for i in range(k):
                    a = nodes[-1]
                    b = "out" if i == k - 1 else f"n{i+1}"
                    if b != "out":
                        nodes.append(b)
                    vals = []
                    for slot_type in macro.slot_types:
                        if slot_type == "L":
                            vals.append(rng.uniform(1e-9, 1e-6))
                        elif slot_type == "C":
                            vals.append(rng.uniform(1e-13, 1e-9))
                        else:
                            vals.append(rng.uniform(1e-6, 1e-3))
                    comps.extend(macro.expand_fn(a, b, "gnd", vals, i))

                tokens, slot_values = components_to_dslv2_tokens(comps, macro_name=macro_name)
                if not with_cells:
                    new_tokens = []
                    new_slots = []
                    for t, v in zip(tokens, slot_values):
                        if t == CELL:
                            continue
                        new_tokens.append(t)
                        new_slots.append(v)
                    tokens, slot_values = new_tokens, new_slots

                decoded = dslv2_tokens_to_components(tokens, slot_values=slot_values)
                self.assertEqual(len(decoded), len(comps))
                for c1, c2 in zip(comps, decoded):
                    self.assertEqual(c1.ctype, c2.ctype)
                    self.assertEqual(c1.role, c2.role)
                    self.assertEqual(c1.node1, c2.node1)
                    self.assertEqual(c1.node2, c2.node2)
                    self.assertTrue(math.isclose(c1.value_si, c2.value_si, rel_tol=0, abs_tol=1e-18))

    def test_grammar_mask_blocks_empty_main_end(self):
        tokens = [
            BOS,
            MAIN_START,
            PORT_IN,
            PORT_OUT,
            PORT_GND,
            REPEAT_START,
            K_TOKENS[0],
            CASCADE,
            CALL,
            MACRO_CELL_LS,
            CELL,
            VAL_L,
            REPEAT_END,
        ]
        vocab_tokens = list(
            {
                *tokens,
                MAIN_END,
                VAL_C,
                "</s>",
                "<pad>",
            }
        )
        tok = DummyTokenizer(vocab_tokens)
        prefix_fn = make_dslv2_prefix_allowed_tokens_fn(tok)

        prefix_after_ports = [tok.get_vocab()[t] for t in (MAIN_START, PORT_IN, PORT_OUT, PORT_GND)]
        allowed = prefix_fn(0, prefix_after_ports)
        self.assertNotIn(tok.get_vocab()[MAIN_END], allowed)

        prefix_after_segment = [tok.get_vocab()[t] for t in tokens]
        allowed_after_segment = prefix_fn(0, prefix_after_segment)
        self.assertIn(tok.get_vocab()[MAIN_END], allowed_after_segment)

    def test_varint_k_and_cell_indices_round_trip(self):
        macro = MACRO_CELL_LS_CS
        vals = [[1e-9, 2e-12] for _ in range(15)]
        tokens, slots = components_to_dslv2_tokens(
            [],
            macro_name=macro,
            segments=[(macro, vals)],
            use_varint_k=True,
            use_cell_indices=True,
            include_bos=True,
        )
        # Ensure K is encoded with <K> D D </K>
        self.assertIn(K_VAR_START, tokens)
        self.assertIn(K_VAR_END, tokens)
        self.assertTrue(any(t in DIGIT_TOKENS for t in tokens))
        # Ensure cell index tokens appear
        self.assertTrue(any(t in CELL_INDEX_TOKENS for t in tokens))
        comps = dslv2_tokens_to_components(tokens, slot_values=slots)
        self.assertEqual(len(comps), 15 * 2)
        self.assertTrue(all(c.value_si > 0 for c in comps))

    def test_multi_segment_expansion_and_action_bridge(self):
        segments = [
            (MACRO_CELL_LS, [[5e-9], [6e-9]]),  # repeat
            (MACRO_PI_CLC, [[1e-12, 2e-9, 1.5e-12]]),  # single CALL
        ]
        tokens, slots = components_to_dslv2_tokens([], segments=segments, use_varint_k=True)
        comps = dslv2_tokens_to_components(tokens, slot_values=slots)
        # Expect 2 series L + pi section (C, L, C)
        self.assertEqual(len(comps), 5)
        action_tokens = dslv2_tokens_to_action_tokens(tokens, slot_values=slots)
        self.assertIn("<ACT_START>", action_tokens[0])


if __name__ == "__main__":
    unittest.main()
