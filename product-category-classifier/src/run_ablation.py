"""Attribute ablation: the proposed model lost to the baseline when
fed all 4 attributes (gender, baseColour, season, usage). Before
concluding anything, test whether a smaller, less noisy attribute set
actually helps -- this is the legitimate way to chase a real win
instead of tuning until something sticks.

Trains/evaluates the proposed model for each variant across the same
3 seeds, aggregates each, then prints all variants next to the
baseline for a single comparison.

Usage:
    python -m src.run_ablation --seeds 0 1 2 --epochs 10
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import aggregate
from .data import variant_tag

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

VARIANTS = [
    ["gender"],
    ["gender", "season", "usage"],
]


def _run(args_list):
    print(f"\n$ {' '.join(args_list)}")
    subprocess.run(args_list, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    for attrs in VARIANTS:
        for seed in args.seeds:
            _run([
                sys.executable, "-m", "src.train",
                "--model", "proposed", "--seed", str(seed),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--attributes", *attrs,
            ])
            _run([
                sys.executable, "-m", "src.evaluate",
                "--model", "proposed", "--seed", str(seed),
                "--attributes", *attrs,
            ])

    print("\nAggregating each variant across seeds...")
    summaries = {}
    baseline_summary = json.loads((ARTIFACTS_DIR / "metrics_baseline_summary.json").read_text())
    summaries["baseline"] = baseline_summary
    summaries["proposed (all 4 attrs)"] = json.loads((ARTIFACTS_DIR / "metrics_proposed_summary.json").read_text())

    for attrs in VARIANTS:
        tag = variant_tag(attrs)
        summary = aggregate.aggregate_model("proposed", args.seeds, tag=tag)
        summaries[f"proposed ({'+'.join(attrs)})"] = summary

    print("\n=== Ablation comparison ===")
    print(f"{'variant':<32} {'accuracy':<20} {'macro_f1':<20}")
    for name, s in summaries.items():
        acc = f"{s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}"
        f1 = f"{s['macro_f1']['mean']:.4f}+/-{s['macro_f1']['std']:.4f}"
        print(f"{name:<32} {acc:<20} {f1:<20}")


if __name__ == "__main__":
    main()
