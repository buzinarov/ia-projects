"""Recommendation service behind the storefront page.

Markets-standard content-based recommendation, served in-process:
  - text search  -> embed the query, cosine-rank the catalog (semantic search)
  - "more like this" -> item-to-item hybrid ranking (similarity + category + gender)
  - "recommended for you" -> a diverse default shelf across categories

It reuses the project's own components -- the shared sentence-encoder
(`src/embeddings.py`) and `HybridRecommender` (`src/recommender.py`) -- and
maps every catalog row to its 80x80 photo via the cached image array, so any
recommended product can be shown with its image, not just the 24 bundled
samples. Every product on screen is a real catalog item; nothing is generated.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import IMAGES_CACHE  # noqa: E402
from src.embeddings import embed  # noqa: E402
from src.recommender import EmbeddingRetriever, HybridRecommender, load_catalog  # noqa: E402


class RecommenderService:
    """Loaded once and shared across requests (see `get_service`)."""

    def __init__(self):
        self.df = load_catalog().reset_index(drop=True)
        # EmbeddingRetriever loads/builds the cached catalog embeddings; we
        # reuse its matrix for query search and hand it to the hybrid model.
        self.retriever = EmbeddingRetriever(self.df)
        self.matrix = self.retriever.matrix  # [n, dim], L2-normalized
        self.hybrid = HybridRecommender(self.df, retriever=self.retriever)
        self.images = np.load(IMAGES_CACHE, mmap_mode="r")  # [n, 80, 80, 3], aligned to df
        self._has_text = self.df["has_text"].to_numpy()
        self._image_cache = {}

    # -- product rendering ------------------------------------------------
    def _image_uri(self, idx):
        if idx not in self._image_cache:
            arr = np.asarray(self.images[idx], dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            self._image_cache[idx] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return self._image_cache[idx]

    def _product(self, idx):
        row = self.df.iloc[idx]
        name = str(row["productDisplayName"]).strip() or f"Product {row['id']}"
        return {
            "idx": str(idx),
            "name": name,
            "subcategory": str(row["subCategory"]),
            "article_type": str(row["articleType"]),
            "gender": str(row["gender"]),
            "colour": str(row["baseColour"]),
            "image": self._image_uri(idx),
        }

    # -- recommendation entry points -------------------------------------
    def default_shelf(self, n=12):
        """A diverse 'recommended for you' shelf: one representative product
        from each of the most populous subcategories (deterministic)."""
        idxs = []
        for sub in self.df["subCategory"].value_counts().index:
            rows = np.where((self.df["subCategory"].to_numpy() == sub) & self._has_text)[0]
            if rows.size:
                idxs.append(int(rows[0]))
            if len(idxs) >= n:
                break
        return [self._product(i) for i in idxs]

    def search(self, query, n=12):
        """Semantic search: embed the query, cosine-rank the catalog."""
        q = embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)
        out = []
        for i in order:
            i = int(i)
            if self._has_text[i]:
                out.append(self._product(i))
            if len(out) >= n:
                break
        return out

    def similar(self, idx, n=12):
        """Item-to-item: hybrid ranking around a chosen product."""
        idxs = self.hybrid.recommend(int(idx), k=n)
        return [self._product(i) for i in idxs]

    def product_name(self, idx):
        return self._product(int(idx))["name"]


_service = None


def get_service():
    """Lazy singleton -- the first call loads the catalog, embeddings, and
    image cache (a few seconds); later calls are instant."""
    global _service
    if _service is None:
        _service = RecommenderService()
    return _service


# -- conversational layer -------------------------------------------------
def assistant_reply(query, products):
    """A concrete, grounded one-liner for the chat: it states what was found
    (count + the dominant categories) and points to the results, instead of
    vague marketing copy. Deterministic, instant, and never invents product
    names -- the products themselves are shown as cards under this reply."""
    n = len(products)
    if not n:
        return (
            f"I couldn't find a good match for “{query}”. Try describing it "
            "differently — a color, a kind of item, or an occasion."
        )
    cats = list(dict.fromkeys(p["subcategory"] for p in products))[:2]
    cat_phrase = " and ".join(cats) if cats else "items"
    return (
        f"Found {n} matches for “{query}” — mostly {cat_phrase}. Here are the "
        "top picks; the full set is on the shelf. Tap “Find similar” on any item "
        "to see more like it."
    )
