"""Trains the recommender's image signal -- the CNN whose predicted
subcategory the recommender uses to keep suggestions on-category.

Run across multiple seeds (via src.run_all, or manually) to get a
mean +/- std read instead of trusting a single training run.

Usage:
    python -m src.train --seed 0 --epochs 10
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import IMG_SIZE, get_dataloaders
from .models import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
MODEL_NAME = "image_classifier"


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for (images, attrs), labels in loader:
            images, attrs, labels = images.to(device), attrs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images, attrs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def compute_class_weights(loader, num_classes, device):
    counts = np.zeros(num_classes)
    for (_, _), labels in loader:
        for l in labels.numpy():
            counts[l] += 1
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-frac", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, seed: {args.seed}")

    train_loader, val_loader, _, maps = get_dataloaders(
        batch_size=args.batch_size, subset_frac=args.subset_frac, seed=args.seed,
    )
    num_classes = len(maps["target_classes"])

    model = build_model(num_classes=num_classes, img_size=IMG_SIZE)
    model.to(device)

    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} - "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
        )
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc}
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "seed": args.seed,
            "num_classes": num_classes,
            "img_size": IMG_SIZE,
        },
        CHECKPOINT_DIR / f"{MODEL_NAME}_seed{args.seed}.pt",
    )
    history_payload = {
        "model": MODEL_NAME,
        "seed": args.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs": history,
    }
    (ARTIFACTS_DIR / f"history_{MODEL_NAME}_seed{args.seed}.json").write_text(json.dumps(history_payload, indent=2))
    print(f"Saved checkpoint and history for '{MODEL_NAME}' (seed {args.seed})")


if __name__ == "__main__":
    main()
