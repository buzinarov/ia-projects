"""Index-construction logic is tested with synthetic embeddings in a temp Chroma
dir (no encoder needed). The full encode->retrieve path is checked against the
persisted index when it and the model are available, and skipped otherwise."""
import importlib

import numpy as np
import pandas as pd
import pytest

# `src.rag` imports chromadb at module level; skip the whole file cleanly (rather
# than erroring at collection) when chromadb isn't installed, e.g. in lean CI.
pytest.importorskip("chromadb")
from src import rag  # noqa: E402


def test_clean_handles_missing():
    assert rag._clean(None, "x") == "x"
    assert rag._clean(float("nan"), 0) == 0
    assert rag._clean("Tops") == "Tops"
    assert rag._clean(5, 0) == 5


def _tiny_df():
    return pd.DataFrame({
        "source_index": [10, 11, 12],
        "Review Text": ["soft and cozy", "runs too small", "great value for money"],
        "Rating": [5, 2, 4],
        "Recommended IND": [1, 0, 1],
        "Department Name": ["Tops", "Dresses", "Bottoms"],
        "Class Name": ["Knits", "Dresses", "Pants"],
    })


@pytest.mark.skipif(importlib.util.find_spec("chromadb") is None, reason="chromadb not installed")
def test_build_index_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path / "chroma")
    df = _tiny_df()
    emb = np.eye(3, 8, dtype=np.float32)  # 3 orthogonal unit vectors

    collection = rag.build_index(df=df, embeddings=emb, force_rebuild=True)
    assert collection.count() == 3

    # Querying with row 1's own vector must return row 1 first at ~0 distance.
    res = collection.query(query_embeddings=[emb[1].tolist()], n_results=3)
    assert res["ids"][0][0] == "11"
    assert res["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    assert res["metadatas"][0][0]["department"] == "Dresses"


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or not (rag.CHROMA_DIR.exists() and any(rag.CHROMA_DIR.iterdir())),
    reason="needs the encoder and a prebuilt index (run `python -m src.run_all`)",
)
def test_find_similar_returns_query_first():
    example = "Absolutely wonderful - silky and sexy and comfortable"
    hits = rag.find_similar_reviews(example, n=3)
    assert len(hits) == 3
    assert hits[0]["review"] == example          # the query exists in the data
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-3)
    assert all("department" in h for h in hits)
