from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.torch_dataset import FilterDesignDataset
from src.data import quantization
from src.models import VACTT5


def build_value_map(tokenizer) -> Dict[int, float]:
    """Map value tokens to numeric SI values for value-aware embedding."""
    vocab = tokenizer.get_vocab()
    mapping: Dict[int, float] = {}
    for tok, tid in vocab.items():
        if tok.startswith("<VAL_"):
            label = tok.replace("<VAL_", "").replace(">", "")
            try:
                mapping[tid] = float(quantization.label_to_value(label))
            except Exception:
                # skip tokens that cannot be parsed
                continue
    return mapping


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        waves = torch.stack([b["wave"] for b in batch])  # (B, C, L)
        scalars = torch.stack([b["scalar"] for b in batch])
        filter_type = scalars[:, 0].long()
        fc_hz = scalars[:, 1]

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            l = len(b["input_ids"])
            input_ids[i, :l] = b["input_ids"]
        labels = input_ids.clone()
        return {
            "wave": waves,
            "filter_type": filter_type,
            "fc_hz": fc_hz,
            "labels": labels,
        }

    return collate


class VACTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            wave=inputs["wave"],
            filter_type=inputs["filter_type"],
            fc_hz=inputs["fc_hz"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VACT-T5 on LC filter dataset.")
    p.add_argument("--data", type=Path, required=True, help="Path to train jsonl.")
    p.add_argument("--tokenizer", type=str, required=True, help="Path or name of tokenizer.")
    p.add_argument("--output", type=Path, default=Path("checkpoints/vact_t5"), help="Checkpoint dir.")
    p.add_argument("--t5-name", type=str, default="t5-small", help="HF model name, e.g., t5-small or t5-base.")
    p.add_argument("--batch-size", type=int, default=8, help="Per-device batch size.")
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--log-steps", type=int, default=50, help="Logging steps.")
    p.add_argument("--save-steps", type=int, default=500, help="Checkpoint save steps.")
    p.add_argument("--use-wave", choices=["ideal", "real", "both"], default="real", help="Which waveform to use.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    value_map = build_value_map(tokenizer)

    train_ds = FilterDesignDataset(str(args.data), tokenizer, use_wave=args.use_wave)
    collate_fn = make_collate_fn(tokenizer)

    model = VACTT5(
        t5_name=args.t5_name,
        value_token_to_value=value_map,
        waveform_in_channels=2 if args.use_wave in {"ideal", "real"} else 4,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = VACTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(str(args.output))


if __name__ == "__main__":
    main()
