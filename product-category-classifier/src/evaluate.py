"""Evaluates the trained image-signal classifier on the held-out test
split, writing metrics + a confusion matrix plot to artifacts/ for one
seed.

Usage:
    python -m src.evaluate --seed 0

Aggregating across seeds is a separate step (src/aggregate.py).
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

from .data import IMG_SIZE, get_dataloaders
from .models import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
MODEL_NAME = "image_classifier"

MAX_ANNOTATED_CLASSES = 12  # above this, per-cell text on the confusion matrix is unreadable noise


def load_checkpoint(name, seed, num_classes, device):
    model = build_model(num_classes=num_classes, img_size=IMG_SIZE)
    state = torch.load(CHECKPOINT_DIR / f"{name}_seed{seed}.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_labels, all_preds = [], []
    for (images, attrs), labels in loader:
        images, attrs = images.to(device), attrs.to(device)
        preds = model(images, attrs).argmax(dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


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


def evaluate_model(seed, test_loader, class_names, device, num_classes):
    model = load_checkpoint(MODEL_NAME, seed, num_classes, device)
    labels, preds = collect_predictions(model, test_loader, device)

    report = classification_report(
        labels, preds, labels=range(num_classes), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=range(num_classes))

    metrics = {
        "model": MODEL_NAME,
        "seed": seed,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {c: report[c] for c in class_names},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"metrics_{MODEL_NAME}_seed{seed}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(
        cm, class_names, ARTIFACTS_DIR / f"confusion_matrix_{MODEL_NAME}_seed{seed}.png",
        f"{MODEL_NAME} (seed {seed}) - confusion matrix",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, maps = get_dataloaders(seed=args.seed)
    class_names = maps["target_classes"]
    num_classes = len(class_names)

    metrics = evaluate_model(args.seed, test_loader, class_names, device, num_classes)
    print(
        f"{MODEL_NAME} (seed {args.seed}): accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
