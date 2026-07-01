"""Load and clean the Women's Clothing E-Commerce Reviews dataset.

The raw file (~23.5k rows, 10 columns) is the public Kaggle dataset by
nicapotato, anonymized real commercial reviews. It is **downloaded on first
use** from a public GitHub mirror and cached under ``data/`` (gitignored), so
the repo stays reproducible without committing the dataset or needing Kaggle
auth -- the same pattern the sibling classifier uses to pull from Hugging Face.

The whole project keys off one column, ``Review Text``. The other columns are
not the prediction target -- they are the *labels we happen to have* that let us
check, honestly, whether the embedding space is meaningful (see src/evaluate.py):
``Recommended IND`` (binary sentiment) and ``Department Name`` (product area).
"""
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "womens_clothing_reviews.csv"

# Public mirror of the Kaggle "Women's E-Commerce Clothing Reviews" dataset.
DATA_URL = (
    "https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/"
    "master/Womens%20Clothing%20E-Commerce%20Reviews.csv"
)

TEXT_COLUMN = "Review Text"
LABEL_COLUMNS = ["Recommended IND", "Department Name", "Class Name", "Rating"]


def _download(dest=CSV_PATH):
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading reviews dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, dest)
    print(f"Saved to {dest}")


def load_raw():
    """Return the raw dataset exactly as published (23,486 rows), downloading
    it to the local cache on first use. The leading unnamed column is the
    original row index and is dropped."""
    if not CSV_PATH.exists():
        _download()
    return pd.read_csv(CSV_PATH, index_col=0)


def load_reviews(dedupe=True):
    """Return a clean, analysis-ready frame of reviews.

    Keeps only rows with usable ``Review Text``, trims whitespace, optionally
    drops exact-duplicate texts (the dataset has a handful of repeats that would
    otherwise dominate any "most similar" result), and adds a ``review_length``
    helper. The original row index is preserved in ``source_index`` so a review
    can always be traced back to the published file.
    """
    df = load_raw().reset_index(names="source_index")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype("string").str.strip()
    df = df[df[TEXT_COLUMN].notna() & (df[TEXT_COLUMN].str.len() > 0)].copy()
    if dedupe:
        df = df.drop_duplicates(subset=TEXT_COLUMN, keep="first")
    df = df.reset_index(drop=True)
    df["review_length"] = df[TEXT_COLUMN].str.len()
    return df


def review_texts(df=None):
    """The list of review strings -- the input to every embedding call."""
    if df is None:
        df = load_reviews()
    return df[TEXT_COLUMN].tolist()


if __name__ == "__main__":
    df = load_reviews()
    print(f"{len(df):,} reviews after cleaning "
          f"({load_raw()[TEXT_COLUMN].isna().sum():,} rows had no review text).")
    print(df[["source_index", TEXT_COLUMN, *LABEL_COLUMNS]].head())
