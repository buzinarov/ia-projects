"""Aggregates per-seed image-classifier metrics into a mean +/- std
summary, so the reported accuracy reflects more than one lucky (or
unlucky) training run. Writes a slim summary (headline metrics +
per-class F1 + summed confusion matrix).

Usage:
    python -m src.aggregate --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import MODEL_NAME, plot_confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def load_seed_metrics(model_name, seeds):
    return [json.loads((ARTIFACTS_DIR / f"metrics_{model_name}_seed{s}.json").read_text()) for s in seeds]


def _agg(values):
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": [float(v) for v in values]}


def aggregate_model(model_name, seeds):
    runs = load_seed_metrics(model_name, seeds)
    class_names = runs[0]["class_names"]

    summary = {
        "model": model_name,
        "seeds": seeds,
        "accuracy": _agg([r["accuracy"] for r in runs]),
        "macro_f1": _agg([r["macro_f1"] for r in runs]),
        "weighted_f1": _agg([r["weighted_f1"] for r in runs]),
        "per_class_f1": {
            c: _agg([r["per_class"][c]["f1-score"] for r in runs]) for c in class_names
        },
        "per_class_support": {
            c: int(runs[0]["per_class"][c]["support"]) for c in class_names
        },
        "class_names": class_names,
    }

    confusion_matrix_summed = np.sum([r["confusion_matrix"] for r in runs], axis=0)
    summary["confusion_matrix_summed"] = confusion_matrix_summed.tolist()

    (ARTIFACTS_DIR / f"metrics_{model_name}_summary.json").write_text(json.dumps(summary, indent=2))
    plot_confusion_matrix(
        confusion_matrix_summed, class_names,
        ARTIFACTS_DIR / f"confusion_matrix_{model_name}_summary.png",
        f"{model_name} - confusion matrix (summed across seeds {seeds})",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    summary = aggregate_model(MODEL_NAME, args.seeds)
    print(
        f"{MODEL_NAME}: accuracy={summary['accuracy']['mean']:.4f}+/-{summary['accuracy']['std']:.4f} "
        f"macro_f1={summary['macro_f1']['mean']:.4f}+/-{summary['macro_f1']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
