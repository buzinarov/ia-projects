"""Data loading and preprocessing for the product category classifier.

Source dataset: ashraq/fashion-product-images-small (Hugging Face Hub),
a public mirror of the Kaggle "Fashion Product Images (Small)" dataset.
Public, no auth required.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_maps.json"

HF_DATASET = "ashraq/fashion-product-images-small"
IMG_SIZE = 80
SEED = 42
MIN_CLASS_COUNT = 100

IMAGES_CACHE = PROCESSED_DIR / f"images_{IMG_SIZE}.npy"
GENDER_CACHE = PROCESSED_DIR / "gender.npy"
LABEL_CACHE = PROCESSED_DIR / "labels.npy"


def _build_label_maps(genders, categories):
    gender_classes = sorted(set(genders))
    category_classes = sorted(set(categories))
    maps = {
        "gender_classes": gender_classes,
        "masterCategory_classes": category_classes,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_PATH.write_text(json.dumps(maps, indent=2))
    return maps


def load_label_maps():
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"{LABEL_MAP_PATH} not found. Run prepare_dataset() first."
        )
    return json.loads(LABEL_MAP_PATH.read_text())


def prepare_dataset(force_rebuild=False):
    """Download (if needed), filter, resize, and cache the dataset to disk.

    Caches a uint8 image array + integer-encoded gender/label arrays under
    data/processed/, so repeated training runs skip re-decoding 44k images.
    """
    if not force_rebuild and IMAGES_CACHE.exists() and LABEL_MAP_PATH.exists():
        return

    from datasets import load_dataset

    print(f"Loading {HF_DATASET} from the Hugging Face Hub (cached after first run)...")
    ds = load_dataset(HF_DATASET, split="train")

    keep_cols = {"image", "gender", "masterCategory"}
    ds = ds.select_columns(list(keep_cols))

    def is_valid(row):
        return row["gender"] is not None and row["masterCategory"] is not None

    ds = ds.filter(is_valid)

    # Drop categories with too few samples to survive a stratified
    # train/val/test split (e.g. "Home" has a single example in this
    # dataset). MIN_CLASS_COUNT keeps the comparison honest: every
    # remaining class has enough examples in every split to evaluate.
    counts = Counter(ds["masterCategory"])
    keep_categories = {c for c, n in counts.items() if n >= MIN_CLASS_COUNT}
    dropped = {c: n for c, n in counts.items() if n < MIN_CLASS_COUNT}
    if dropped:
        print(f"Dropping rare masterCategory classes (< {MIN_CLASS_COUNT} samples): {dropped}")
    ds = ds.filter(lambda row: row["masterCategory"] in keep_categories)

    genders = ds["gender"]
    categories = ds["masterCategory"]
    maps = _build_label_maps(genders, categories)
    gender_to_idx = {g: i for i, g in enumerate(maps["gender_classes"])}
    category_to_idx = {c: i for i, c in enumerate(maps["masterCategory_classes"])}

    n = len(ds)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    gender_idx = np.zeros(n, dtype=np.int64)
    label_idx = np.zeros(n, dtype=np.int64)

    print(f"Resizing {n} images to {IMG_SIZE}x{IMG_SIZE} (one-time cost)...")
    for i, row in enumerate(ds):
        img = row["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images[i] = np.array(img, dtype=np.uint8)
        gender_idx[i] = gender_to_idx[row["gender"]]
        label_idx[i] = category_to_idx[row["masterCategory"]]
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(IMAGES_CACHE, images)
    np.save(GENDER_CACHE, gender_idx)
    np.save(LABEL_CACHE, label_idx)
    print(f"Cached processed dataset to {PROCESSED_DIR}")


class FashionProductDataset(Dataset):
    """Returns ((image_tensor, gender_onehot), label) to match the
    multi-input model's forward signature: model(x_image, x_gender)."""

    def __init__(self, images, gender_idx, label_idx, num_genders, augment=False):
        self.images = images
        self.gender_idx = gender_idx
        self.label_idx = label_idx
        self.num_genders = num_genders
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()  # horizontal flip only
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        gender_onehot = torch.zeros(self.num_genders, dtype=torch.float32)
        gender_onehot[self.gender_idx[idx]] = 1.0

        label = torch.tensor(self.label_idx[idx], dtype=torch.long)
        return (img_tensor, gender_onehot), label


def get_dataloaders(batch_size=64, subset_frac=1.0, num_workers=0):
    """Loads the cached dataset, splits 70/15/15 (stratified by label),
    and returns (train_loader, val_loader, test_loader, label_maps)."""
    prepare_dataset()
    maps = load_label_maps()

    images = np.load(IMAGES_CACHE)
    gender_idx = np.load(GENDER_CACHE)
    label_idx = np.load(LABEL_CACHE)

    n = len(images)
    indices = np.arange(n)
    if subset_frac < 1.0:
        indices, _ = train_test_split(
            indices, train_size=subset_frac, stratify=label_idx, random_state=SEED
        )

    train_idx, rest_idx = train_test_split(
        indices, test_size=0.30, stratify=label_idx[indices], random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=label_idx[rest_idx], random_state=SEED
    )

    num_genders = len(maps["gender_classes"])

    def make_dataset(idx, augment):
        return FashionProductDataset(
            images[idx], gender_idx[idx], label_idx[idx], num_genders, augment=augment
        )

    train_ds = make_dataset(train_idx, augment=True)
    val_ds = make_dataset(val_idx, augment=False)
    test_ds = make_dataset(test_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, maps
