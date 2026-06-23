"""Evaluates a trained model on the held-out test split, writes metrics +
a confusion matrix plot to artifacts/. Once both baseline and proposed
checkpoints exist, also exports a small bundle of sample images + both
models' predictions, so the Streamlit demo works without the full dataset.

Usage:
    python -m src.evaluate --model baseline
    python -m src.evaluate --model proposed
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from .data import IMG_SIZE, get_dataloaders
from .models import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
APP_SAMPLES_DIR = ROOT_DIR / "app" / "sample_images"

CONFIDENCE_THRESHOLD = 0.85


def load_checkpoint(name, num_classes, num_genders, device):
    model = build_model(name, num_classes=num_classes, num_genders=num_genders, img_size=IMG_SIZE)
    state = torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_labels, all_preds, all_confidences = [], [], []
    for (images, genders), labels in loader:
        images, genders = images.to(device), genders.to(device)
        probs = torch.softmax(model(images, genders), dim=1)
        confs, preds = probs.max(dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_confidences.extend(confs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_confidences)


def plot_confusion_matrix(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_model(name, test_loader, class_names, device, num_classes, num_genders):
    model = load_checkpoint(name, num_classes, num_genders, device)
    labels, preds, confidences = collect_predictions(model, test_loader, device)

    report = classification_report(
        labels, preds, labels=range(num_classes), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=range(num_classes))

    history_path = ARTIFACTS_DIR / f"history_{name}.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    high_conf_mask = confidences >= CONFIDENCE_THRESHOLD
    per_class_high_conf = {}
    for idx, cname in enumerate(class_names):
        class_mask = labels == idx
        per_class_high_conf[cname] = (
            float((confidences[class_mask] >= CONFIDENCE_THRESHOLD).mean()) if class_mask.sum() > 0 else None
        )

    metrics = {
        "model": name,
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
        # Per-item test results, so the frontend can recompute the
        # auto-tag rate live at any confidence threshold (not just 0.85).
        "test_labels": labels.tolist(),
        "test_predictions": preds.tolist(),
        "test_confidences": confidences.tolist(),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(
        cm, class_names, ARTIFACTS_DIR / f"confusion_matrix_{name}.png", f"{name} - confusion matrix"
    )
    return metrics


def export_demo_assets(maps, device, num_classes, num_genders, n_samples=24):
    baseline_ckpt = CHECKPOINT_DIR / "baseline.pt"
    proposed_ckpt = CHECKPOINT_DIR / "proposed.pt"
    if not (baseline_ckpt.exists() and proposed_ckpt.exists()):
        return

    baseline = load_checkpoint("baseline", num_classes, num_genders, device)
    proposed = load_checkpoint("proposed", num_classes, num_genders, device)

    class_names = maps["masterCategory_classes"]
    gender_names = maps["gender_classes"]

    images = np.load(PROCESSED_DIR / f"images_{IMG_SIZE}.npy")
    gender_idx_arr = np.load(PROCESSED_DIR / "gender.npy")
    label_idx_arr = np.load(PROCESSED_DIR / "labels.npy")

    rng = np.random.default_rng(42)
    per_class = max(1, n_samples // num_classes)
    chosen = []
    for c in range(num_classes):
        class_indices = np.where(label_idx_arr == c)[0]
        if len(class_indices) == 0:
            continue
        chosen.extend(
            rng.choice(class_indices, size=min(per_class, len(class_indices)), replace=False).tolist()
        )

    APP_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.no_grad():
        for i in chosen:
            img_arr = images[i]
            filename = f"sample_{i}.png"
            Image.fromarray(img_arr).save(APP_SAMPLES_DIR / filename)

            img_tensor = torch.from_numpy(img_arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            gender_onehot = torch.zeros(1, num_genders, dtype=torch.float32)
            gender_onehot[0, gender_idx_arr[i]] = 1.0
            gender_onehot = gender_onehot.to(device)

            entry = {
                "id": int(i),
                "filename": filename,
                "true_gender": gender_names[gender_idx_arr[i]],
                "true_category": class_names[label_idx_arr[i]],
            }
            for model_name, model in (("baseline", baseline), ("proposed", proposed)):
                probs = torch.softmax(model(img_tensor, gender_onehot), dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                entry[f"{model_name}_prediction"] = class_names[pred_idx]
                entry[f"{model_name}_confidence"] = float(probs[pred_idx])
            manifest.append(entry)

    (ARTIFACTS_DIR / "predictions_sample.json").write_text(json.dumps(manifest, indent=2))
    (APP_SAMPLES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(manifest)} demo samples to {APP_SAMPLES_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "proposed"], required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, maps = get_dataloaders()
    class_names = maps["masterCategory_classes"]
    num_classes = len(class_names)
    num_genders = len(maps["gender_classes"])

    metrics = evaluate_model(args.model, test_loader, class_names, device, num_classes, num_genders)
    print(
        f"{args.model}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"high_confidence_rate={metrics['high_confidence_rate']:.4f}"
    )

    export_demo_assets(maps, device, num_classes, num_genders)


if __name__ == "__main__":
    main()
