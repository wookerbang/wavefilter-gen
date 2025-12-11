"""Plot ideal vs real S-parameters (S21/S11) for a selected sample in a jsonl dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def load_sample(path: Path, index: Optional[int], sample_id: Optional[str]) -> Dict[str, Any]:
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if sample_id is not None:
                if obj.get("sample_id") == sample_id:
                    return obj
            elif index is not None and i == index:
                return obj
    raise ValueError("Sample not found (check index/sample-id)")


def plot_sample(sample: Dict[str, Any], output: Optional[Path] = None) -> None:
    freq = sample["freq_hz"]
    ideal_s21 = sample["ideal_s21_db"]
    real_s21 = sample["real_s21_db"]
    ideal_s11 = sample["ideal_s11_db"]
    real_s11 = sample["real_s11_db"]

    order = sample.get("order")
    ftype = sample.get("filter_type")
    title = f"{ftype or 'filter'}, order {order}"

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(freq, ideal_s21, label="Ideal S21", color="tab:blue")
    axes[0].plot(freq, real_s21, label="Real S21", color="tab:orange", linestyle="--")
    axes[0].set_ylabel("S21 (dB)")
    axes[0].grid(True, which="both", ls=":")
    axes[0].legend()

    axes[1].plot(freq, ideal_s11, label="Ideal S11", color="tab:blue")
    axes[1].plot(freq, real_s11, label="Real S11", color="tab:orange", linestyle="--")
    axes[1].set_ylabel("S11 (dB)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, which="both", ls=":")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output:
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ideal vs real S21/S11 for one sample.")
    p.add_argument(
        "-f",
        "--jsonl",
        type=Path,
        default=Path("data/processed/demo/train.jsonl"),
        help="Path to dataset jsonl file (default: data/processed/demo/train.jsonl).",
    )
    p.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Zero-based index of the sample to plot (default: 0).",
    )
    p.add_argument(
        "-s",
        "--sample-id",
        help="Sample ID to select (e.g., train_0). If set, overrides --index.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to save the figure instead of showing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_index = None if args.sample_id else args.index
    sample = load_sample(args.jsonl, index=target_index, sample_id=args.sample_id)
    plot_sample(sample, output=args.output)


if __name__ == "__main__":
    main()
