"""Data loading for the review-intelligence assistant.

Source dataset: florentgbelidji/edmunds-car-ratings (Hugging Face Hub),
Edmunds consumer car reviews. Public, no auth required. Columns include
`Review` (free text) and `Rating` (1-5 float).

The sentiment labels used by the Triage skill's *evaluation* are a PROXY
derived from the star rating, not human annotations:

    Rating >= 4 -> positive
    Rating <= 2 -> negative
    Rating == 3 -> dropped (genuinely ambiguous)

This measures agreement with rating-implied sentiment. That caveat is the
project's honesty boundary and is repeated wherever these labels are used
(see docs/requirement.md).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

HF_DATASET = "florentgbelidji/edmunds-car-ratings"
SPLIT_SEED = 42  # fixed so "is the model better" never depends on a lucky split

POSITIVE_THRESHOLD = 4.0  # Rating >= -> positive
NEGATIVE_THRESHOLD = 2.0  # Rating <= -> negative
# Ratings strictly between the two thresholds (i.e. the 3-star middle) are
# dropped from the labeled evaluation set as ambiguous.

REVIEW_COLUMN = "Review"
RATING_COLUMN = "Rating"

CACHE_PATH = PROCESSED_DIR / "edmunds_labeled.parquet"

# The original DataCamp exercise's five hand-picked reviews, retained as a
# fixed demo set for the QA / translation examples (positions matter: the
# 2nd review emphasizes the brand, the last is summarized).
DEMO_CSV = DATA_DIR / "car_reviews.csv"


def rating_to_label(rating):
    """Map a star rating to a binary sentiment label, or None if ambiguous.

    Returns "POSITIVE" | "NEGATIVE" | None. None means the row should be
    excluded from the labeled evaluation set, not silently bucketed.
    """
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return None
    if rating >= POSITIVE_THRESHOLD:
        return "POSITIVE"
    if rating <= NEGATIVE_THRESHOLD:
        return "NEGATIVE"
    return None


def _load_raw():
    """Load the Edmunds dataset from the Hub as a pandas DataFrame."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    return ds.to_pandas()


def load_labeled_reviews(force_rebuild=False, max_chars=2000):
    """Return a DataFrame of reviews with a rating-derived sentiment label.

    Columns: review (str), rating (float), label (str: POSITIVE/NEGATIVE).
    Ambiguous (3-star) and empty rows are dropped. Long reviews are
    truncated to `max_chars` so a transformer's token budget isn't blown by
    a handful of essay-length outliers. Cached to parquet after first build.
    """
    if not force_rebuild and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    raw = _load_raw()
    df = raw[[REVIEW_COLUMN, RATING_COLUMN]].rename(
        columns={REVIEW_COLUMN: "review", RATING_COLUMN: "rating"}
    )
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() > 0]
    df["label"] = df["rating"].apply(rating_to_label)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["review"] = df["review"].str.slice(0, max_chars)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    return df


def train_eval_split(df, eval_frac=0.2, seed=SPLIT_SEED):
    """Stratified split into (train_df, eval_df).

    There is no model *training* here -- the skills are pre-trained -- so
    "train" is really a held-out development slice. The split is stratified
    on the proxy label and seeded so the evaluation set is identical run to
    run.
    """
    from sklearn.model_selection import train_test_split

    train_df, eval_df = train_test_split(
        df, test_size=eval_frac, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def label_balance(df):
    """Return {label: fraction} -- used by the EDA notebook and to justify
    why macro-F1 (not accuracy) is the acceptance metric for Triage."""
    counts = df["label"].value_counts()
    return (counts / counts.sum()).to_dict()


def load_demo_reviews():
    """Load the original five-review demo CSV if present, else return None.

    The file uses ';' as the delimiter (DataCamp's format) with columns
    Review and Class. Returns a DataFrame with columns review, label.
    """
    if not DEMO_CSV.exists():
        return None
    df = pd.read_csv(DEMO_CSV, delimiter=";")
    return df.rename(columns={"Review": "review", "Class": "label"})


if __name__ == "__main__":
    df = load_labeled_reviews()
    print(f"Loaded {len(df)} labeled reviews from {HF_DATASET}")
    print(f"Label balance: {json.dumps(label_balance(df), indent=2)}")
    print(df.head())
