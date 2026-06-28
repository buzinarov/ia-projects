"""Reproduces the recommender's image signal end to end: trains and
evaluates both vision variants across all seeds, then aggregates the
classification-appendix metrics. One command -- the "it's automatic"
entry point.

Usage:
    python -m src.run_all --seeds 0 1 2 --epochs 10
"""
import argparse
import subprocess
import sys

from . import aggregate


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

    for model in ("baseline", "proposed"):
        for seed in args.seeds:
            _run([
                sys.executable, "-m", "src.train",
                "--model", model, "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--subset-frac", str(args.subset_frac),
            ])
            _run([sys.executable, "-m", "src.evaluate", "--model", model, "--seed", str(seed)])

    print("\nAggregating across seeds...")
    for model in ("baseline", "proposed"):
        summary = aggregate.aggregate_model(model, args.seeds)
        print(
            f"{model}: accuracy={summary['accuracy']['mean']:.4f}+/-{summary['accuracy']['std']:.4f} "
            f"macro_f1={summary['macro_f1']['mean']:.4f}+/-{summary['macro_f1']['std']:.4f}"
        )


if __name__ == "__main__":
    main()
