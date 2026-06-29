"""Product recommenders for the suggested-product surface.

Two recommenders are defined here so they can be compared honestly on the
same offline metrics (see src/evaluate_reco.py):

  - PopularityRecommender -- the baseline that models the simple system
    "already in production": within the query item's subCategory, it
    returns items of the most popular articleTypes in the catalog. It is
    category-correct by construction but blind to the specific item the
    user is looking at -- the "generic, category-obvious" behavior the
    commercial stakeholder is unhappy with.

  - HybridRecommender -- the proposed multi-modal model. It ranks by
    metadata similarity (sentence-transformer embeddings over the catalog
    text, the same model the live Chroma index uses) and boosts candidates
    that match the query's *predicted category* (the image-classification
    signal) and its gender. The category signal is injectable, so the
    offline evaluation can supply the ground-truth category (isolating
    retrieval quality from classifier error) while the live app can supply
    the image model's prediction.

Relevance ground truth is a content-based PROXY built only from columns
that exist in the catalog: an item is relevant to a query when it shares
the query's articleType AND gender. This is finer than the subCategory
the baseline groups by, which is what makes the comparison meaningful --
and it is labeled as a proxy wherever it is reported, because this
dataset has no user-interaction data to measure real relevance.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_filtered_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CATALOG_CACHE = PROCESSED_DIR / "catalog.csv"
CATALOG_EMB_CACHE = PROCESSED_DIR / "catalog_embeddings.npy"

# Columns the recommenders need, beyond the target/attribute columns the
# classifier already caches. Kept aligned (same row order, same null/rare
# filtering) with the cached image array, so an anchor's catalog row and
# its cached image share an index.
CATALOG_COLUMNS = [
    "id", "productDisplayName", "articleType",
    "subCategory", "gender", "baseColour", "season", "usage",
]
# The free-text document the similarity signal indexes. It deliberately
# excludes `articleType` and `gender` -- the two fields that define the
# relevance proxy -- so the retriever cannot read the answer off a
# verbatim categorical token. It must recover relevance from the natural
# product name and the remaining attributes, the way a real content-based
# recommender does. The category (image signal) and gender enter as
# structured boosts in HybridRecommender, not as indexed text.
TEXT_FIELDS = ["productDisplayName", "baseColour", "season", "usage"]
# Columns that must be non-empty for a row to be a usable document/candidate.
CATALOG_NONTEXT_FILL = ["articleType", "subCategory", "gender", "baseColour", "season", "usage"]


def load_catalog(force_rebuild=False):
    """Returns the catalog as a DataFrame whose row index lines up with
    the cached image array (data.load_filtered_dataset applies the exact
    same filtering), so the optional image-based category signal can look
    an anchor's photo up by index. Rows with a missing display name keep
    their position but are flagged unusable via the `has_text` column,
    rather than being dropped (dropping would break that alignment)."""
    if not force_rebuild and CATALOG_CACHE.exists():
        return pd.read_csv(CATALOG_CACHE, keep_default_na=False)

    ds = load_filtered_dataset(extra_columns=("id", "productDisplayName", "articleType"))
    # Pull only the metadata columns -- never materialize the image column
    # into the DataFrame.
    df = pd.DataFrame({col: ds[col] for col in CATALOG_COLUMNS})
    for col in [*TEXT_FIELDS, *CATALOG_NONTEXT_FILL]:
        df[col] = df[col].fillna("").astype(str)
    df["has_text"] = df["productDisplayName"].str.strip() != ""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG_CACHE, index=False)
    return df


def build_text(df):
    """Concatenates the catalog's own metadata columns into one searchable
    string per row -- the document the similarity signal indexes."""
    return (df[TEXT_FIELDS].agg(" ".join, axis=1)).str.strip()


def relevant_mask(anchor_idx, df):
    """Boolean array over `df`: a candidate is relevant to the anchor when
    it shares the anchor's articleType AND gender, excluding the anchor
    itself. This is the content-based relevance PROXY -- finer than the
    subCategory the popularity baseline groups by."""
    anchor = df.iloc[anchor_idx]
    mask = (df["articleType"] == anchor["articleType"]) & (df["gender"] == anchor["gender"])
    mask = mask.to_numpy().copy()
    mask[anchor_idx] = False
    return mask


class PopularityRecommender:
    """Baseline: within the anchor's subCategory, rank candidates by how
    popular their articleType is across the whole catalog (ties broken by
    id, for determinism). Models a naive production recommender."""

    name = "popularity_baseline"

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.article_popularity = Counter(self.df["articleType"])
        self._pop = self.df["articleType"].map(self.article_popularity).to_numpy()
        self._subcat = self.df["subCategory"].to_numpy()
        self._ids = self.df["id"].to_numpy()

    def recommend(self, anchor_idx, k=10):
        anchor_subcat = self._subcat[anchor_idx]
        candidates = np.where(self._subcat == anchor_subcat)[0]
        candidates = candidates[candidates != anchor_idx]
        # Sort by articleType popularity desc, then id asc -- a stable,
        # query-independent ranking (the source of the "generic" feel).
        order = sorted(candidates, key=lambda i: (-self._pop[i], self._ids[i]))
        return order[:k]


class EmbeddingRetriever:
    """Similarity over the catalog text, using the shared sentence-encoder
    (src/embeddings.py) -- the *same* model the live Chroma index uses, so
    offline evaluation and the served system measure similarity identically.

    Embeddings are L2-normalized, so cosine similarity is a plain dot
    product. Vectors are cached to disk on first build."""

    def __init__(self, df, cache_path=CATALOG_EMB_CACHE):
        from .embeddings import embed_cached

        self.df = df
        documents = build_text(df).tolist()
        self.matrix = embed_cached(documents, cache_path)  # [n, dim], normalized

    def similarities(self, anchor_idx):
        sims = (self.matrix @ self.matrix[anchor_idx]).astype(float)
        sims[anchor_idx] = -np.inf  # never recommend the anchor itself
        return sims


class HybridRecommender:
    """Proposed model: similarity ranking, boosted by the query's
    predicted category (the image-classification signal) and its gender.

    `category_provider(anchor_idx) -> subCategory` is injected so the
    same code serves both the offline evaluation (ground-truth category,
    to isolate retrieval quality) and the live app (image-model
    prediction). `category_boost` and `attr_boost` are added on top of the
    cosine similarity, which lives in [0, 1]."""

    name = "hybrid_proposed"

    def __init__(self, df, retriever=None, category_provider=None,
                 category_boost=0.5, attr_boost=0.25):
        self.df = df.reset_index(drop=True)
        self.retriever = retriever or EmbeddingRetriever(self.df)
        self.category_provider = category_provider or (lambda i: self.df["subCategory"].iat[i])
        self.category_boost = category_boost
        self.attr_boost = attr_boost
        self._subcat = self.df["subCategory"].to_numpy()
        self._gender = self.df["gender"].to_numpy()
        self._has_text = self.df["has_text"].to_numpy()

    def recommend(self, anchor_idx, k=10):
        sims = self.retriever.similarities(anchor_idx).astype(float)
        predicted_cat = self.category_provider(anchor_idx)
        anchor_gender = self._gender[anchor_idx]

        scores = sims.copy()
        scores += self.category_boost * (self._subcat == predicted_cat)
        scores += self.attr_boost * (self._gender == anchor_gender)
        scores[~self._has_text] = -np.inf  # untitled rows are not recommendable
        scores[anchor_idx] = -np.inf

        # argpartition for the top-k, then sort just those (stable order).
        k = min(k, np.isfinite(scores).sum())
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])].tolist()


class ImageCategoryProvider:
    """Category signal from the trained image classifier, for the full
    live pipeline. Classifies an anchor's cached photo and returns the
    predicted subCategory. Needs the model checkpoints and the cached
    image array; the offline evaluation defaults to ground-truth category
    instead, so this is only used with `--category-signal image`."""

    def __init__(self, df, model_name="image_classifier", seed=0):
        import torch

        from .data import IMAGES_CACHE
        from .inference import get_device, load_model

        self._torch = torch
        self.images = np.load(IMAGES_CACHE)
        if len(self.images) != len(df):
            raise ValueError(
                "Image cache and catalog are misaligned; rebuild the catalog "
                "with load_catalog(force_rebuild=True) after (re)caching images."
            )
        self.model, self.maps, self.device = load_model(model_name, seed=seed, device=get_device())
        self.target_classes = self.maps["target_classes"]

    def __call__(self, anchor_idx):
        img = self.images[anchor_idx].astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # The image classifier ignores the attribute argument; pass a zero
        # vector to satisfy the shared forward signature.
        attr = self._torch.zeros(1, self.maps["attribute_dim"], device=self.device)
        with self._torch.no_grad():
            logits = self.model(tensor, attr)
            idx = int(logits.argmax(dim=1).item())
        return self.target_classes[idx]
