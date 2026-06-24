"""Runs the full experiment end to end: train + evaluate both models
across all seeds, aggregate, then export the Streamlit demo assets.
One command, reproducible -- the "it's automatic" entry point.

Usage:
    python -m src.run_all --seeds 0 1 2 --epochs 10
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import aggregate
from .data import ATTRIBUTE_COLUMNS, IMG_SIZE, PROCESSED_DIR, load_filtered_dataset, load_label_maps
from .evaluate import CHECKPOINT_DIR, load_checkpoint

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
APP_SAMPLES_DIR = ROOT_DIR / "app" / "assets" / "sample_images"


def _run(args_list):
    print(f"\n$ {' '.join(args_list)}")
    subprocess.run(args_list, check=True)


def export_demo_assets(maps, device, num_classes, attr_dim, demo_seed, n_samples=24):
    """Bundles a small stratified sample of test images + both models'
    predictions (at one designated seed), so the Streamlit Live Demo
    works without the full dataset or a training run."""
    baseline_ckpt = CHECKPOINT_DIR / f"baseline_seed{demo_seed}.pt"
    proposed_ckpt = CHECKPOINT_DIR / f"proposed_seed{demo_seed}.pt"
    if not (baseline_ckpt.exists() and proposed_ckpt.exists()):
        print(f"Skipping demo asset export -- checkpoints for seed {demo_seed} not found.")
        return

    baseline = load_checkpoint("baseline", demo_seed, num_classes, attr_dim, device)
    proposed = load_checkpoint("proposed", demo_seed, num_classes, attr_dim, device)

    class_names = maps["target_classes"]
    attribute_classes = maps["attribute_classes"]

    images = np.load(PROCESSED_DIR / f"images_{IMG_SIZE}.npy")
    label_idx_arr = np.load(PROCESSED_DIR / f"{maps['target_column'].lower()}.npy")
    attr_idx_arrs = {col: np.load(PROCESSED_DIR / f"{col.lower()}.npy") for col in ATTRIBUTE_COLUMNS}

    print("Looking up product names for the sampled rows...")
    product_names = load_filtered_dataset(extra_columns=("productDisplayName",))["productDisplayName"]

    rng = np.random.default_rng(42)
    per_class = max(1, n_samples // num_classes)
    chosen = []
    for c in range(num_classes):
        class_indices = np.where(label_idx_arr == c)[0]
        if len(class_indices) == 0:
            continue
        chosen.extend(rng.choice(class_indices, size=min(per_class, len(class_indices)), replace=False).tolist())

    for f in APP_SAMPLES_DIR.glob("sample_*.png"):
        f.unlink()

    APP_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.no_grad():
        for i in chosen:
            img_arr = images[i]
            filename = f"sample_{i}.png"
            Image.fromarray(img_arr).save(APP_SAMPLES_DIR / filename)

            img_tensor = torch.from_numpy(img_arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            onehots = []
            true_attrs = {}
            for col in ATTRIBUTE_COLUMNS:
                classes = attribute_classes[col]
                idx = int(attr_idx_arrs[col][i])
                true_attrs[col] = classes[idx]
                oh = torch.zeros(len(classes), dtype=torch.float32)
                oh[idx] = 1.0
                onehots.append(oh)
            attr_tensor = torch.cat(onehots).unsqueeze(0).to(device)

            entry = {
                "id": int(i),
                "filename": filename,
                "product_name": product_names[i] or f"Product #{i}",
                "true_category": class_names[label_idx_arr[i]],
                **{f"true_{col}": true_attrs[col] for col in ATTRIBUTE_COLUMNS},
            }
            for model_name, model in (("baseline", baseline), ("proposed", proposed)):
                probs = torch.softmax(model(img_tensor, attr_tensor), dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                entry[f"{model_name}_prediction"] = class_names[pred_idx]
                entry[f"{model_name}_confidence"] = float(probs[pred_idx])
            manifest.append(entry)

    (ARTIFACTS_DIR / "predictions_sample.json").write_text(json.dumps(manifest, indent=2))
    (APP_SAMPLES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(manifest)} demo samples to {APP_SAMPLES_DIR}")


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

    print("\nExporting Streamlit demo assets...")
    maps = load_label_maps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_demo_assets(
        maps, device,
        num_classes=len(maps["target_classes"]),
        attr_dim=maps["attribute_dim"],
        demo_seed=args.seeds[0],
    )


if __name__ == "__main__":
    main()
