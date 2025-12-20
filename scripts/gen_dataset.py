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
    p.add_argument("--vact-cell", dest="vact_cell", action="store_true", help="Insert <CELL> markers in VACT.")
    p.add_argument("--no-vact-cell", dest="vact_cell", action="store_false", help="Disable <CELL> markers in VACT.")
    p.set_defaults(vact_cell=True)
    p.add_argument("--vactdsl", dest="vactdsl", action="store_true", help="Emit VACT-DSL tokens (default: on).")
    p.add_argument("--no-vactdsl", dest="vactdsl", action="store_false", help="Disable VACT-DSL token emission.")
    p.set_defaults(vactdsl=True)
    p.add_argument("--actions", dest="actions", action="store_true", help="Emit action-construction tokens (default: on).")
    p.add_argument("--no-actions", dest="actions", action="store_false", help="Disable action-construction tokens.")
    p.set_defaults(actions=True)
    p.add_argument("--dslv2", dest="dslv2", action="store_true", help="Emit VACT-DSL v2 tokens (macro/repeat).")
    p.add_argument("--no-dslv2", dest="dslv2", action="store_false", help="Disable VACT-DSL v2 tokens.")
    p.set_defaults(dslv2=True)
    p.add_argument("--max-nodes", type=int, default=32, help="Max internal nodes after canonicalization (n1..nK).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = build_dataset(
        num_samples=args.num_samples,
        output_dir=str(args.output_dir),
        split=args.split,
        use_ngspice=bool(args.use_ngspice),
        seed=args.seed,
        emit_vact_cells=bool(args.vact_cell),
        emit_vactdsl=bool(args.vactdsl),
        emit_actions=bool(args.actions),
        emit_dslv2=bool(args.dslv2),
        max_nodes=int(args.max_nodes),
    )
    print(f"Dataset written to {path}")


if __name__ == "__main__":
    main()
