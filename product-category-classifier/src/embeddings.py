"""Single source of truth for text embeddings.

Both the offline recommender (src/recommender.py) and the live retrieval
index (src/rag.py) embed product text with the *same* model, so the
offline evaluation and the served system measure similarity the same way
-- a reviewer should never have to wonder whether "the numbers" and "the
product" used different math.

Model: sentence-transformers `all-MiniLM-L6-v2` -- a small, fast,
widely-used sentence encoder (384-dim). Embeddings are L2-normalized, so
a dot product is cosine similarity.
"""
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model_cache = {}


def get_model(name=EMBEDDING_MODEL):
    """Lazily load (and cache) the sentence encoder. Imported lazily so
    importing this module -- and the recommender -- stays cheap, and the
    offline unit tests never pull the model down."""
    if name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def embed(texts, batch_size=256, show_progress=False):
    """Encode a list of strings into L2-normalized embeddings
    (float32, shape [len(texts), EMBEDDING_DIM]). Normalized so cosine
    similarity is a plain dot product downstream."""
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_cached(texts, cache_path):
    """Like embed(), but persists the matrix to `cache_path` and reuses it
    when the row count matches -- keeps repeated offline evaluations fast.
    The catalog text is deterministic given the dataset, so a row-count
    match is a sufficient cache key here."""
    cache_path = Path(cache_path)
    texts = list(texts)
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts) and cached.shape[1] == EMBEDDING_DIM:
            return cached
    vectors = embed(texts, show_progress=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors
