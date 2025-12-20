from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from src.data.value_decomp import MDRConfig, torch_compose_value, torch_decompose_mdr

from .hybrid_value_head import HybridValueHead
from .scalar_encoder import SpecEncoder
from .waveform_encoder import MultiScaleWaveformEncoder


class VACTT5(nn.Module):
    """
    Waveform/spec encoder + T5 decoder for VACT generation.
    Defaults to t5-small for快速迭代，可切换 t5-base.
    """

    def __init__(
        self,
        t5_name: str = "t5-small",
        waveform_in_channels: int = 1,
        d_model_override: Optional[int] = None,
        vocab_size: Optional[int] = None,
        value_loss_weight: float = 1.0,
        mdr_cfg: MDRConfig = MDRConfig(),
        value_token_ids: Optional[Sequence[int]] = None,
        slot_type_token_to_idx: Optional[Mapping[int, int]] = None,
    ):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        if vocab_size is not None:
            self.t5.resize_token_embeddings(vocab_size)
        d_model = d_model_override or self.t5.config.d_model

        self.wave_encoder = MultiScaleWaveformEncoder(d_model=d_model, in_channels=waveform_in_channels)
        self.spec_encoder = SpecEncoder(d_model=d_model, type_vocab_size=4)

        self.mdr_cfg = mdr_cfg
        self.value_loss_weight = value_loss_weight
        self.register_buffer("value_token_ids", torch.tensor(value_token_ids, dtype=torch.long) if value_token_ids else None, persistent=False)
        vocab_size = self.t5.get_input_embeddings().weight.shape[0]
        slot_lookup = torch.full((vocab_size,), -1, dtype=torch.long)
        if slot_type_token_to_idx:
            for tok_id, slot_idx in slot_type_token_to_idx.items():
                if 0 <= int(tok_id) < vocab_size:
                    slot_lookup[int(tok_id)] = int(slot_idx)
        self.register_buffer("slot_type_lookup", slot_lookup, persistent=False)

        self.value_head = HybridValueHead(
            d_model,
            num_mantissa=24,
            num_decade=int(mdr_cfg.num_decades),
            residual_log10_scale=float(mdr_cfg.residual_log10_scale),
            num_slot_types=int(torch.max(slot_lookup).item() + 1) if slot_type_token_to_idx else 0,
        )

    def encode(self, wave: torch.Tensor, filter_type: torch.Tensor, fc_hz: torch.Tensor) -> BaseModelOutput:
        """
        Args:
            wave: (B, C, L) waveform (S21 or S21+S11)
            filter_type: (B,) long tensor 0/1
            fc_hz: (B,) float tensor of cutoff freq in Hz
        """
        log_fc = torch.log10(fc_hz.clamp_min(1e-6))
        spec_tok = self.spec_encoder(filter_type, log_fc).unsqueeze(1)  # (B,1,d)
        wave_feat = self.wave_encoder(wave)  # (B,Lw,d)
        enc = torch.cat([spec_tok, wave_feat], dim=1)
        return BaseModelOutput(last_hidden_state=enc)

    def forward(
        self,
        wave: torch.Tensor,
        filter_type: torch.Tensor,
        fc_hz: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        value_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        # mixed discrete-continuous loss: token CE + (mantissa CE + decade CE + residual regression)
        if value_targets is None:
            return outputs

        dec_hidden = outputs.decoder_hidden_states[-1]  # (B,L,d)
        mant_tgt, dec_tgt, res_tgt, valid_vals = torch_decompose_mdr(value_targets, cfg=self.mdr_cfg)
        valid_mask = valid_vals
        if labels is not None and self.value_token_ids is not None:
            val_ids = self.value_token_ids.to(labels.device)
            is_val_tok = torch.isin(labels, val_ids)
            valid_mask = valid_mask & is_val_tok
        if valid_mask.any():
            slot_type_ids = None
            if self.slot_type_lookup is not None:
                vocab_size = self.slot_type_lookup.shape[0]
                safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
                safe_labels = torch.clamp(safe_labels, 0, vocab_size - 1)
                slot_type_ids = torch.gather(self.slot_type_lookup, 0, safe_labels)
                slot_type_ids = torch.where(slot_type_ids >= 0, slot_type_ids, torch.zeros_like(slot_type_ids))
            pred = self.value_head(dec_hidden, slot_type_ids=slot_type_ids)
            mant_loss = F.cross_entropy(pred.mantissa_logits[valid], mant_tgt[valid])
            dec_loss = F.cross_entropy(pred.decade_logits[valid], dec_tgt[valid])
            res_loss = F.mse_loss(pred.residual_log10[valid], res_tgt[valid])
            value_loss = mant_loss + dec_loss + res_loss
            loss = outputs.loss + self.value_loss_weight * value_loss if outputs.loss is not None else value_loss
            outputs.loss = loss
            outputs.value_loss = value_loss
            outputs.value_loss_mantissa = mant_loss
            outputs.value_loss_decade = dec_loss
            outputs.value_loss_residual = res_loss
        return outputs

    @torch.no_grad()
    def predict_values(
        self,
        wave: torch.Tensor,
        filter_type: torch.Tensor,
        fc_hz: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        mode: str = "precision",
    ) -> torch.Tensor:
        """
        Predict per-position physical values from decoder hidden states.

        The caller should mask out non-<VAL_*> positions (e.g., set them to NaN) if desired.
        """
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        out = self.t5(
            encoder_outputs=encoder_outputs,
            labels=token_ids,
            output_hidden_states=True,
        )
        dec_hidden = out.decoder_hidden_states[-1]
        slot_type_ids = None
        if self.slot_type_lookup is not None:
            vocab_size = self.slot_type_lookup.shape[0]
            safe_ids = torch.clamp(token_ids, 0, vocab_size - 1)
            slot_type_ids = torch.gather(self.slot_type_lookup, 0, safe_ids)
            slot_type_ids = torch.where(slot_type_ids >= 0, slot_type_ids, torch.zeros_like(slot_type_ids))
        pred = self.value_head(dec_hidden, slot_type_ids=slot_type_ids)
        mant_idx = torch.argmax(pred.mantissa_logits, dim=-1)
        dec_idx = torch.argmax(pred.decade_logits, dim=-1)
        values = torch_compose_value(mant_idx, dec_idx, pred.residual_log10, cfg=self.mdr_cfg, mode=mode)  # (B,L)
        return values

    @torch.no_grad()
    def generate(
        self,
        wave: torch.Tensor,
        filter_type: torch.Tensor,
        fc_hz: torch.Tensor,
        **kwargs,
    ):
        encoder_outputs = self.encode(wave, filter_type, fc_hz)
        return self.t5.generate(encoder_outputs=encoder_outputs, **kwargs)
