"""Data loading and preprocessing for the product category classifier.

Source dataset: ashraq/fashion-product-images-small (Hugging Face Hub),
a public mirror of the Kaggle "Fashion Product Images (Small)" dataset.
Public, no auth required.

Target and attribute columns are generic by design: TARGET_COLUMN is
whatever the model predicts, ATTRIBUTE_COLUMNS are whatever structured
fields feed the model's second modality. Both are persisted into
label_maps.json so downstream code (models.py, inference.py, the app,
the agent) never hardcodes a column name.
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
SPLIT_SEED = 42  # fixed across all training seeds -- isolates "is the
                 # architecture better" from "did we get a lucky split"
MIN_CLASS_COUNT = 100

TARGET_COLUMN = "subCategory"
ATTRIBUTE_COLUMNS = ["gender", "baseColour", "season", "usage"]

IMAGES_CACHE = PROCESSED_DIR / f"images_{IMG_SIZE}.npy"
LABEL_CACHE = PROCESSED_DIR / f"{TARGET_COLUMN.lower()}.npy"


def _attr_cache_path(column):
    return PROCESSED_DIR / f"{column.lower()}.npy"


def variant_tag(attributes):
    """Filename suffix identifying which attribute subset a 'proposed'
    run used -- empty for the default (all attributes), so existing
    artifacts from before ablation testing keep their original names."""
    if not attributes:
        return ""
    return "_" + "-".join(attributes)


def _filter_rare_classes(ds, column, min_count):
    counts = Counter(ds[column])
    keep = {c for c, n in counts.items() if n >= min_count}
    dropped = {c: n for c, n in counts.items() if n < min_count}
    if dropped:
        print(f"Dropping rare {column} classes (< {min_count} samples): {dropped}")
    return ds.filter(lambda row: row[column] in keep)


def _build_label_maps(target_values, attr_value_lists):
    target_classes = sorted(set(target_values))
    attribute_classes = {col: sorted(set(values)) for col, values in attr_value_lists.items()}
    maps = {
        "target_column": TARGET_COLUMN,
        "target_classes": target_classes,
        "attribute_columns": ATTRIBUTE_COLUMNS,
        "attribute_classes": attribute_classes,
        "attribute_dim": sum(len(v) for v in attribute_classes.values()),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_PATH.write_text(json.dumps(maps, indent=2))
    return maps


def load_label_maps():
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"{LABEL_MAP_PATH} not found. Run prepare_dataset() first.")
    return json.loads(LABEL_MAP_PATH.read_text())


def load_filtered_dataset(extra_columns=()):
    """Loads the HF dataset and applies the exact same null/rare-class
    filtering as prepare_dataset(), so row position `i` here lines up
    with position `i` in the cached arrays. Used to look up auxiliary
    fields (e.g. productDisplayName) for already-cached samples without
    needing to bake them into the image cache."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    keep_cols = {"image", TARGET_COLUMN, *ATTRIBUTE_COLUMNS, *extra_columns}
    ds = ds.select_columns(list(keep_cols))

    required_cols = [TARGET_COLUMN, *ATTRIBUTE_COLUMNS]
    ds = ds.filter(lambda row: all(row[c] is not None for c in required_cols))
    ds = _filter_rare_classes(ds, TARGET_COLUMN, MIN_CLASS_COUNT)
    return ds


def prepare_dataset(force_rebuild=False):
    """Download (if needed), filter, resize, and cache the dataset to disk.

    Caches a uint8 image array plus one small int array per
    target/attribute column under data/processed/, so repeated training
    runs skip re-decoding 44k images.
    """
    if not force_rebuild and IMAGES_CACHE.exists() and LABEL_MAP_PATH.exists():
        return

    print(f"Loading {HF_DATASET} from the Hugging Face Hub (cached after first run)...")
    ds = load_filtered_dataset()

    target_values = ds[TARGET_COLUMN]
    attr_value_lists = {col: ds[col] for col in ATTRIBUTE_COLUMNS}
    maps = _build_label_maps(target_values, attr_value_lists)

    target_to_idx = {c: i for i, c in enumerate(maps["target_classes"])}
    attr_to_idx = {
        col: {c: i for i, c in enumerate(maps["attribute_classes"][col])} for col in ATTRIBUTE_COLUMNS
    }

    n = len(ds)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    label_idx = np.zeros(n, dtype=np.int64)
    attr_idx = {col: np.zeros(n, dtype=np.int64) for col in ATTRIBUTE_COLUMNS}

    print(f"Resizing {n} images to {IMG_SIZE}x{IMG_SIZE} (one-time cost)...")
    for i, row in enumerate(ds):
        img = row["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images[i] = np.array(img, dtype=np.uint8)
        label_idx[i] = target_to_idx[row[TARGET_COLUMN]]
        for col in ATTRIBUTE_COLUMNS:
            attr_idx[col][i] = attr_to_idx[col][row[col]]
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(IMAGES_CACHE, images)
    np.save(LABEL_CACHE, label_idx)
    for col in ATTRIBUTE_COLUMNS:
        np.save(_attr_cache_path(col), attr_idx[col])
    print(f"Cached processed dataset to {PROCESSED_DIR}")


class FashionProductDataset(Dataset):
    """Returns ((image_tensor, attr_onehot), label) to match the
    multi-input model's forward signature: model(x_image, x_attr).

    attr_onehot is the concatenation of one-hot vectors for each column
    in `attribute_columns`, in that exact order -- inference.py and
    agent.py must reconstruct the vector the same way at predict time.
    """

    def __init__(self, images, attr_idx, label_idx, attribute_classes, attribute_columns, augment=False, rng=None):
        self.images = images
        self.attr_idx = attr_idx  # dict[col] -> int array
        self.label_idx = label_idx
        self.attribute_classes = attribute_classes  # dict[col] -> list of class names
        self.attribute_columns = attribute_columns
        self.augment = augment
        self.rng = rng or np.random.default_rng()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        if self.augment and self.rng.random() < 0.5:
            img = img[:, ::-1, :].copy()  # horizontal flip only
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        onehots = []
        for col in self.attribute_columns:
            n_classes = len(self.attribute_classes[col])
            oh = torch.zeros(n_classes, dtype=torch.float32)
            oh[self.attr_idx[col][idx]] = 1.0
            onehots.append(oh)
        attr_onehot = torch.cat(onehots)

        label = torch.tensor(self.label_idx[idx], dtype=torch.long)
        return (img_tensor, attr_onehot), label


def get_dataloaders(batch_size=64, subset_frac=1.0, num_workers=0, seed=0, attribute_columns=None):
    """Loads the cached dataset, splits 70/15/15 (stratified by label,
    using the fixed SPLIT_SEED), and returns
    (train_loader, val_loader, test_loader, label_maps).

    `seed` only affects shuffling/augmentation randomness, not the split.
    `attribute_columns` lets a caller use a subset of the cached
    attribute columns (e.g. for an ablation sweep) without needing a
    separate cache -- every attribute is cached independently, so any
    subset of ATTRIBUTE_COLUMNS can be selected at load time.
    """
    prepare_dataset()
    maps = load_label_maps()
    attribute_columns = attribute_columns or ATTRIBUTE_COLUMNS

    images = np.load(IMAGES_CACHE)
    label_idx = np.load(LABEL_CACHE)
    attr_idx = {col: np.load(_attr_cache_path(col)) for col in attribute_columns}

    n = len(images)
    indices = np.arange(n)
    if subset_frac < 1.0:
        indices, _ = train_test_split(
            indices, train_size=subset_frac, stratify=label_idx, random_state=SPLIT_SEED
        )

    train_idx, rest_idx = train_test_split(
        indices, test_size=0.30, stratify=label_idx[indices], random_state=SPLIT_SEED
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=label_idx[rest_idx], random_state=SPLIT_SEED
    )

    def make_dataset(idx, augment, rng=None):
        return FashionProductDataset(
            images[idx],
            {col: attr_idx[col][idx] for col in attribute_columns},
            label_idx[idx],
            maps["attribute_classes"],
            attribute_columns,
            augment=augment,
            rng=rng,
        )

    train_ds = make_dataset(train_idx, augment=True, rng=np.random.default_rng(seed))
    val_ds = make_dataset(val_idx, augment=False)
    test_ds = make_dataset(test_idx, augment=False)

    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=train_generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, maps
