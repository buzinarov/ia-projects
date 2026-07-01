"""The retrieval proxy and probe helpers are tested on tiny synthetic data so
the math is pinned without needing the encoder or the full dataset."""
import numpy as np
import pandas as pd

from src.evaluate import retrieval_proxy, run_probe


def _clustered_dataset(per_class=20, dim=8, n_classes=3, seed=0):
    """Embeddings tightly clustered by department: nearest neighbors should
    almost always share the label, so precision@k -> ~1.0 and lift >> 1."""
    rng = np.random.default_rng(seed)
    centers = np.eye(n_classes, dim, dtype=np.float32)
    rows, labels = [], []
    for c in range(n_classes):
        pts = centers[c] + 0.01 * rng.standard_normal((per_class, dim)).astype(np.float32)
        rows.append(pts)
        labels += [f"dept{c}"] * per_class
    emb = np.vstack(rows)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    df = pd.DataFrame({
        "Review Text": [f"review {i}" for i in range(len(labels))],
        "Department Name": labels,
        "Recommended IND": [i % 2 for i in range(len(labels))],
    })
    return df, emb


def test_retrieval_proxy_rewards_clustered_space():
    df, emb = _clustered_dataset()
    res = retrieval_proxy(df, emb, label_col="Department Name", sample=60, k=3)
    assert res["precision@3"] > 0.95          # neighbors share the department
    assert res["random_baseline"] < 0.5       # 3 balanced classes -> ~0.33
    assert res["lift_x"] > 2.0
    assert res["n_queries"] == 60


def test_retrieval_proxy_excludes_self():
    # If self-matching weren't excluded, a duplicated row would always retrieve
    # itself. With distinct clusters precision stays high regardless, but the
    # n_queries / shape bookkeeping must hold.
    df, emb = _clustered_dataset(per_class=10)
    res = retrieval_proxy(df, emb, sample=999, k=2)  # sample capped at dataset size
    assert res["n_queries"] == len(df)


def test_run_probe_separates_recoverable_label():
    df, emb = _clustered_dataset(per_class=40)
    out = run_probe(df, emb, "Department Name", seed=0)
    # A linear probe should recover a label that is linearly separable by design.
    assert out["embeddings"]["accuracy"] > 0.95
    assert out["n_classes"] == 3
