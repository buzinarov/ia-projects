"""Reproduces the recommender's image signal end to end: trains and
evaluates the image classifier across all seeds, then aggregates the
metrics. One command -- the "it's automatic" entry point.

Usage:
    python -m src.run_all --seeds 0 1 2 --epochs 10
"""
import argparse
import subprocess
import sys

from . import aggregate
from .evaluate import MODEL_NAME


def _run(args_list):
    print(f"\n$ {' '.join(args_list)}")
    subprocess.run(args_list, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-frac", type=float, default=1.0)
    args = parser.parse_args()

    for seed in args.seeds:
        _run([
            sys.executable, "-m", "src.train",
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--subset-frac", str(args.subset_frac),
        ])
        _run([sys.executable, "-m", "src.evaluate", "--seed", str(seed)])

    print("\nAggregating across seeds...")
    summary = aggregate.aggregate_model(MODEL_NAME, args.seeds)
    print(
        f"{MODEL_NAME}: accuracy={summary['accuracy']['mean']:.4f}+/-{summary['accuracy']['std']:.4f} "
        f"macro_f1={summary['macro_f1']['mean']:.4f}+/-{summary['macro_f1']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
