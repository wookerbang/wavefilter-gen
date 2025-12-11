"""Simple CLI to run the data generation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_builder import build_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LC filter dataset jsonl.")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate.")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/demo"), help="Directory to write jsonl.")
    p.add_argument("--split", type=str, default="train", help="Split name, used in <split>.jsonl.")
    p.add_argument("--use-ngspice", action="store_true", help="Use ngspice for real-wave simulation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for spec sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = build_dataset(
        num_samples=args.num_samples,
        output_dir=str(args.output_dir),
        split=args.split,
        use_ngspice=bool(args.use_ngspice),
        seed=args.seed,
    )
    print(f"Dataset written to {path}")


if __name__ == "__main__":
    main()
