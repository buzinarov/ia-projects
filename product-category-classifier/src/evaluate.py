"""Evaluates a trained model on the held-out test split, writes metrics +
a confusion matrix plot to artifacts/ for one (model, seed) pair.

Usage:
    python -m src.evaluate --model baseline --seed 0
    python -m src.evaluate --model proposed --seed 0

Aggregating across seeds and exporting the Streamlit demo assets are
separate steps (src/aggregate.py, src/run_all.py) -- this module's job
is strictly "evaluate one checkpoint."
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .data import IMG_SIZE, get_dataloaders, variant_tag
from .models import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

CONFIDENCE_THRESHOLD = 0.85
MAX_ANNOTATED_CLASSES = 12  # above this, per-cell text on the confusion matrix is unreadable noise


def load_checkpoint(name, seed, num_classes, attr_dim, device, tag=""):
    model = build_model(name, num_classes=num_classes, attr_dim=attr_dim, img_size=IMG_SIZE)
    state = torch.load(CHECKPOINT_DIR / f"{name}{tag}_seed{seed}.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_labels, all_preds, all_confidences = [], [], []
    for (images, attrs), labels in loader:
        images, attrs = images.to(device), attrs.to(device)
        probs = torch.softmax(model(images, attrs), dim=1)
        confs, preds = probs.max(dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_confidences.extend(confs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_confidences)


def plot_confusion_matrix(cm, class_names, out_path, title):
    n = len(class_names)
    annotate = n <= MAX_ANNOTATED_CLASSES
    figsize = (6, 5) if annotate else (max(10, n * 0.45), max(9, n * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8 if annotate else 7)
    ax.set_yticklabels(class_names, fontsize=8 if annotate else 7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_model(name, seed, test_loader, class_names, device, num_classes, attr_dim, tag=""):
    model = load_checkpoint(name, seed, num_classes, attr_dim, device, tag=tag)
    labels, preds, confidences = collect_predictions(model, test_loader, device)

    report = classification_report(
        labels, preds, labels=range(num_classes), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=range(num_classes))

    history_path = ARTIFACTS_DIR / f"history_{name}{tag}_seed{seed}.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}

    high_conf_mask = confidences >= CONFIDENCE_THRESHOLD
    per_class_high_conf = {}
    for idx, cname in enumerate(class_names):
        class_mask = labels == idx
        per_class_high_conf[cname] = (
            float((confidences[class_mask] >= CONFIDENCE_THRESHOLD).mean()) if class_mask.sum() > 0 else None
        )

    metrics = {
        "model": name,
        "seed": seed,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {c: report[c] for c in class_names},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "high_confidence_rate": float(high_conf_mask.mean()),
        "high_confidence_threshold": CONFIDENCE_THRESHOLD,
        "per_class_high_confidence_rate": per_class_high_conf,
        "training_history": history,
        "test_labels": labels.tolist(),
        "test_predictions": preds.tolist(),
        "test_confidences": confidences.tolist(),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"metrics_{name}{tag}_seed{seed}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(
        cm, class_names, ARTIFACTS_DIR / f"confusion_matrix_{name}{tag}_seed{seed}.png",
        f"{name}{tag} (seed {seed}) - confusion matrix",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "proposed"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attributes", nargs="+", default=None,
        help="Subset of attribute columns the checkpoint was trained with (proposed model only).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, maps = get_dataloaders(seed=args.seed, attribute_columns=args.attributes)
    class_names = maps["target_classes"]
    num_classes = len(class_names)
    attr_dim = (
        maps["attribute_dim"] if not args.attributes
        else sum(len(maps["attribute_classes"][c]) for c in args.attributes)
    )
    tag = variant_tag(args.attributes)

    metrics = evaluate_model(args.model, args.seed, test_loader, class_names, device, num_classes, attr_dim, tag=tag)
    print(
        f"{args.model}{tag} (seed {args.seed}): accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} high_confidence_rate={metrics['high_confidence_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
