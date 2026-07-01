"""Single source of truth for text embeddings.

Every part of this project -- the t-SNE map, the theme triage, the linear
probe, and the live similarity search -- embeds review text with the *same*
model, so "the numbers" in the evaluation and "the product" in the app measure
similarity with identical math. A reviewer should never have to wonder whether
two components used different vectors.

Model: sentence-transformers ``all-MiniLM-L6-v2`` -- a small, fast, widely-used
sentence encoder (384-dim). Embeddings are L2-normalized, so a dot product is
cosine similarity and Chroma's cosine space agrees with the offline math.

No API keys: this is a public portfolio repo, so embeddings are computed
locally. Nothing here can leak.
"""
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model_cache = {}


def get_model(name=EMBEDDING_MODEL):
    """Lazily load (and cache) the sentence encoder. Imported lazily so that
    importing this module stays cheap and the pure-logic unit tests never pull
    the model down."""
    if name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def embed(texts, batch_size=256, show_progress=False):
    """Encode strings into L2-normalized float32 embeddings of shape
    ``[len(texts), EMBEDDING_DIM]``. Normalized so cosine similarity downstream
    is a plain dot product."""
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_cached(texts, cache_path, show_progress=True):
    """Like :func:`embed`, but persists the matrix to ``cache_path`` and reuses
    it when the row count matches. Review text is deterministic given the
    cleaned dataset, so a row-count match is a sufficient cache key here and
    keeps repeated runs (evaluation, app startup) fast."""
    cache_path = Path(cache_path)
    texts = list(texts)
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(texts), EMBEDDING_DIM):
            return cached
    vectors = embed(texts, show_progress=show_progress)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors
