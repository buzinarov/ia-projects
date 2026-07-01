"""Chroma-backed similarity search over the reviews -- the live retrieval that
powers "find the 3 most similar past reviews" in the app and the brief's
``find_similar_reviews`` deliverable.

The index is built from the **same** precomputed, L2-normalized embeddings used
everywhere else (src/embeddings.py) and queried with **cosine** distance, so the
served search and the offline evaluation rank reviews identically. Passing the
vectors in directly (rather than handing Chroma the encoder) guarantees there is
exactly one set of vectors in the whole project.
"""
import math
from pathlib import Path

import chromadb

from .data import load_reviews
from .embeddings import embed, embed_cached

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "artifacts" / "chroma"
EMBED_CACHE = ROOT_DIR / "artifacts" / "embeddings" / "review_embeddings.npy"
COLLECTION_NAME = "reviews"


def _clean(value, default=""):
    """Chroma metadata must be str/int/float/bool -- never None/NaN."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def build_index(df=None, embeddings=None, force_rebuild=False, batch_size=1000):
    """Build (or reuse) the persistent Chroma collection over all reviews.

    Returns the collection. Embeddings are loaded from the shared cache if not
    supplied, so building the index never recomputes vectors the pipeline
    already produced.
    """
    if df is None:
        df = load_reviews()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {c.name for c in client.list_collections()}

    if not force_rebuild and COLLECTION_NAME in existing:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == len(df):
            return collection
        client.delete_collection(COLLECTION_NAME)  # stale (dataset changed) -> rebuild
    elif COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine, not Chroma's L2 default
    )

    if embeddings is None:
        embeddings = embed_cached(df["Review Text"].tolist(), EMBED_CACHE)

    ids = df["source_index"].astype(str).tolist()
    documents = df["Review Text"].tolist()
    metadatas = [
        {
            "rating": int(_clean(row["Rating"], 0)),
            "recommended": int(_clean(row["Recommended IND"], 0)),
            "department": str(_clean(row["Department Name"], "")),
            "class": str(_clean(row["Class Name"], "")),
        }
        for _, row in df.iterrows()
    ]

    for i in range(0, len(ids), batch_size):
        sl = slice(i, i + batch_size)
        collection.add(
            ids=ids[sl],
            documents=documents[sl],
            embeddings=[v.tolist() for v in embeddings[sl]],
            metadatas=metadatas[sl],
        )
    print(f"Indexed {len(ids):,} reviews into Chroma at {CHROMA_DIR}")
    return collection


def get_collection():
    """Return the reviews collection, building it on first use."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return build_index()
        return collection
    except Exception:
        return build_index()


def find_similar_reviews(input_text, n=3, collection=None):
    """Return the ``n`` reviews most similar to ``input_text`` as dicts with the
    matched text, its metadata, and the cosine distance (0 = identical).

    Like the brief's reference, the query is matched against the whole catalog
    including itself, so calling this with a review that exists in the data
    returns that review first.
    """
    if collection is None:
        collection = get_collection()
    query_vec = embed([input_text])[0].tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=n)
    out = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        out.append({"review": doc, **meta, "distance": round(float(dist), 4)})
    return out


if __name__ == "__main__":
    build_index(force_rebuild=True)
    example = "Absolutely wonderful - silky and sexy and comfortable"
    for hit in find_similar_reviews(example, n=3):
        print(f"[{hit['distance']:.3f}] ({hit['department']}, {hit['rating']}★) {hit['review'][:90]}")
