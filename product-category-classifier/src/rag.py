"""Chroma-backed retrieval over product metadata -- the live
metadata-similarity signal behind the recommender.

Powers the `search_similar_products` tool used by the tool-calling agent
(src/agent.py). It embeds product text with the *same* sentence-encoder as
the offline recommender (src/embeddings.py, `all-MiniLM-L6-v2`) and queries
with **cosine** distance, so the served system and the offline evaluation
measure similarity identically rather than on two different metrics.
"""
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .embeddings import EMBEDDING_MODEL

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
COLLECTION_NAME = "fashion_products"
HF_DATASET = "ashraq/fashion-product-images-small"
INDEX_SIZE = 5000  # a representative slice of the catalog -- this is a demo, not a production index
METADATA_COLUMNS = [
    "id", "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "usage", "productDisplayName",
]

# Same model as the offline retriever (src/embeddings.py). Cached on the
# module so building and querying the collection share one encoder.
_embedding_fn_cache = {}


def _embedding_fn():
    if "fn" not in _embedding_fn_cache:
        _embedding_fn_cache["fn"] = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn_cache["fn"]


def _product_text(row):
    parts = [
        row.get("productDisplayName"), row.get("articleType"),
        row.get("masterCategory"), row.get("subCategory"),
        row.get("baseColour"), row.get("gender"),
        row.get("season"), row.get("usage"),
    ]
    return ", ".join(str(p) for p in parts if p)


def build_index(force_rebuild=False, n=INDEX_SIZE):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {c.name for c in client.list_collections()}

    if not force_rebuild and COLLECTION_NAME in existing:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=_embedding_fn())
        if collection.count() > 0:
            return collection

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},  # cosine, not Chroma's L2 default
    )

    from datasets import load_dataset

    print(f"Loading metadata from {HF_DATASET} to build the RAG index...")
    ds = load_dataset(HF_DATASET, split="train")
    ds = ds.select_columns(METADATA_COLUMNS)
    ds = ds.filter(lambda row: row["productDisplayName"] is not None)
    if n and n < len(ds):
        ds = ds.shuffle(seed=42).select(range(n))

    ids, documents, metadatas = [], [], []
    for row in ds:
        ids.append(str(row["id"]))
        documents.append(_product_text(row))
        metadatas.append({
            "productDisplayName": row["productDisplayName"] or "",
            "masterCategory": row["masterCategory"] or "",
            "subCategory": row["subCategory"] or "",
            "articleType": row["articleType"] or "",
            "baseColour": row["baseColour"] or "",
            "gender": row["gender"] or "",
        })

    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
    print(f"Indexed {len(ids)} products into Chroma at {CHROMA_DIR}")
    return collection


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=_embedding_fn())
        if collection.count() == 0:
            return build_index()
        return collection
    except Exception:
        return build_index()


def search_similar_products(query, n_results=5, category=None):
    """Retrieve catalog items similar to `query`. When `category` is given
    (e.g. the subCategory predicted from a product photo), results are
    restricted to that category -- this is how the recommender combines the
    image-classification signal with the metadata-similarity signal."""
    collection = get_collection()
    where = {"subCategory": category} if category else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where)
    products = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        products.append({**meta, "match_text": doc, "distance": round(float(dist), 4)})
    return products


if __name__ == "__main__":
    build_index(force_rebuild=True)
    print(json.dumps(search_similar_products("blue men's casual shoes", n_results=3), indent=2))
