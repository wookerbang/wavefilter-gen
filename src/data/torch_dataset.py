"""
PyTorch Dataset 包装 processed jsonl。
"""

from __future__ import annotations

import json
from typing import Literal, Sequence

import torch
from torch.utils.data import Dataset


class FilterDesignDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        use_wave: Literal["ideal", "real", "both", "ideal_s21", "real_s21", "mix"] = "real",
        mix_real_prob: float = 0.3,
        use_repr: Literal["vact", "vactdsl", "sfci", "action"] = "vact",
        normalize_wave: bool = False,
    ):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.use_wave = use_wave
        self.mix_real_prob = mix_real_prob
        self.use_repr = use_repr
        self.normalize_wave = normalize_wave

    def _tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        """
        Convert pre-split tokens (already without special tokens) to ids.

        Note: the tokenizer here is WordLevel with a whitespace pre-tokenizer,
        so feeding the tokens back into ``tokenizer(...)`` will split the
        angle-bracket tokens and turn everything into <unk>. We therefore
        map the tokens directly via ``convert_tokens_to_ids`` when available.
        """
        if not tokens:
            return []

        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
        elif callable(getattr(self.tokenizer, "__call__", None)):
            out = self.tokenizer(tokens, is_split_into_words=True, add_special_tokens=False)
            ids = out.get("input_ids") or out.get("ids") or []
            if ids and isinstance(ids[0], list):
                # flatten batch-style output
                ids = sum(ids, [])
        elif hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(tokens, add_special_tokens=False)
        else:
            ids = [self.tokenizer.get(t, 0) for t in tokens]

        return [int(i) for i in ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        freq = torch.tensor(s["freq_hz"], dtype=torch.float32)
        ideal_s21 = torch.tensor(s["ideal_s21_db"], dtype=torch.float32)
        ideal_s11 = torch.tensor(s["ideal_s11_db"], dtype=torch.float32)
        real_s21 = torch.tensor(s["real_s21_db"], dtype=torch.float32)
        real_s11 = torch.tensor(s["real_s11_db"], dtype=torch.float32)

        mode = self.use_wave
        if mode == "mix":
            mode = "real" if torch.rand(1).item() < self.mix_real_prob else "ideal"

        if mode == "ideal":
            wave = torch.stack([ideal_s21, ideal_s11], dim=0)
        elif mode == "real":
            wave = torch.stack([real_s21, real_s11], dim=0)
        elif mode == "ideal_s21":
            wave = ideal_s21.unsqueeze(0)
        elif mode == "real_s21":
            wave = real_s21.unsqueeze(0)
        else:
            wave = torch.stack([ideal_s21, ideal_s11, real_s21, real_s11], dim=0)

        ftype = s.get("filter_type", "lowpass")
        type_map = {"lowpass": 0, "highpass": 1, "bandpass": 2, "bandstop": 3}
        type_id = type_map.get(ftype, 0)
        scalar = torch.tensor([type_id, s["fc_hz"]], dtype=torch.float32)

        value_targets = None
        if self.use_repr == "vact":
            tokens_raw = s.get("vact_tokens")
        elif self.use_repr == "vactdsl":
            tokens_raw = s.get("vactdsl_tokens")
        elif self.use_repr == "dslv2":
            tokens_raw = s.get("dslv2_tokens")
            value_targets = s.get("dslv2_slot_values")
        elif self.use_repr == "sfci":
            tokens_raw = s.get("sfci_tokens")
        else:
            tokens_raw = s.get("action_tokens")
        tokens_raw = tokens_raw or []
        token_ids = self._tokens_to_ids(tokens_raw)

        if self.normalize_wave:
            # channel-wise standardization to stabilize optimization
            wave = wave - wave.mean(dim=-1, keepdim=True)
            wave_std = wave.std(dim=-1, keepdim=True).clamp_min(1e-4)
            wave = wave / wave_std

        return {
            "freq": freq,
            "wave": wave,
            "scalar": scalar,
            # keep both keys so collate_fn can pick by repr
            "input_ids": token_ids,
            "vact_tokens": token_ids if self.use_repr == "vact" else None,
            "vactdsl_tokens": token_ids if self.use_repr == "vactdsl" else None,
            "dslv2_tokens": token_ids if self.use_repr == "dslv2" else None,
            "sfci_tokens": token_ids if self.use_repr == "sfci" else None,
            "action_tokens": token_ids if self.use_repr == "action" else None,
            "value_targets": value_targets,
        }
