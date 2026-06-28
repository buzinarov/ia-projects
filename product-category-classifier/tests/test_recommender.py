"""Unit tests for the recommendation metrics and the recommender logic.

These run on a small in-memory catalog -- no dataset download, no
checkpoints -- so the ranking math and relevance proxy are verified on
every clone, the same way the data contract is."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate_reco import ndcg_at_k, precision_at_k, recall_at_k
from src.recommender import HybridRecommender, PopularityRecommender, build_text, relevant_mask


class StubRetriever:
    """Deterministic, offline stand-in for the EmbeddingRetriever: lexical
    token-overlap similarity over the same catalog text. Lets the hybrid
    ranking logic be tested without downloading the sentence encoder."""

    def __init__(self, df):
        self.token_sets = [set(t.lower().split()) for t in build_text(df).tolist()]

    def similarities(self, anchor_idx):
        anchor = self.token_sets[anchor_idx]
        sims = np.array([len(anchor & other) for other in self.token_sets], dtype=float)
        sims[anchor_idx] = -np.inf
        return sims


def _toy_catalog():
    rows = [
        # id, name, articleType, subCategory, gender
        (1, "blue running shoes", "Sports Shoes", "Shoes", "Men"),
        (2, "red running shoes", "Sports Shoes", "Shoes", "Men"),
        (3, "black formal shoes", "Formal Shoes", "Shoes", "Men"),
        (4, "blue casual tshirt", "Tshirts", "Topwear", "Men"),
        (5, "green casual tshirt", "Tshirts", "Topwear", "Men"),
        (6, "white running shoes", "Sports Shoes", "Shoes", "Women"),
    ]
    df = pd.DataFrame(rows, columns=["id", "productDisplayName", "articleType", "subCategory", "gender"])
    df["baseColour"] = ["Blue", "Red", "Black", "Blue", "Green", "White"]
    df["season"] = "Summer"
    df["usage"] = "Casual"
    df["has_text"] = True
    return df


# --- metrics --------------------------------------------------------------

def test_precision_at_k_counts_hits():
    assert precision_at_k([0, 1, 2, 3], {1, 3}, k=4) == 0.5
    assert precision_at_k([0, 1, 2, 3], {1, 3}, k=2) == 0.5


def test_recall_at_k_is_over_relevant_total():
    assert recall_at_k([0, 1], {1, 3, 5}, k=2) == pytest.approx(1 / 3)


def test_recall_and_ndcg_are_zero_without_relevant_items():
    assert recall_at_k([0, 1], set(), k=2) == 0.0
    assert ndcg_at_k([0, 1], set(), k=2) == 0.0


def test_ndcg_rewards_higher_ranked_hits():
    top = ndcg_at_k([1, 0, 0], {1}, k=3)
    low = ndcg_at_k([0, 0, 1], {1}, k=3)
    assert top == 1.0
    assert low < top


# --- relevance proxy ------------------------------------------------------

def test_relevant_mask_matches_articletype_and_gender_excluding_self():
    df = _toy_catalog()
    mask = relevant_mask(0, df)  # anchor: men's Sports Shoes
    assert mask[1]               # another men's Sports Shoes -> relevant
    assert not mask[0]           # never the anchor itself
    assert not mask[2]           # men's Formal Shoes -> different articleType
    assert not mask[5]           # women's Sports Shoes -> different gender


# --- recommenders ---------------------------------------------------------

def test_popularity_baseline_stays_in_subcategory():
    df = _toy_catalog()
    rec = PopularityRecommender(df)
    recs = rec.recommend(0, k=5)  # anchor is in "Shoes"
    assert all(df.iloc[i]["subCategory"] == "Shoes" for i in recs)
    assert 0 not in recs


def test_hybrid_beats_popularity_on_a_specific_query():
    """For a men's Sports-Shoes query, the only true-relevant item is the
    other men's Sports Shoes (id 2). The popularity baseline ranks by the
    most common articleType and can bury it; the hybrid's similarity +
    gender boost should surface it first."""
    df = _toy_catalog()
    relevant = set(np.where(relevant_mask(0, df))[0].tolist())
    assert relevant == {1}

    hybrid = HybridRecommender(df, retriever=StubRetriever(df))  # ground-truth category by default
    top_hybrid = hybrid.recommend(0, k=1)
    assert top_hybrid[0] == 1  # the matching men's running shoe ranks first


def test_hybrid_excludes_untitled_rows():
    df = _toy_catalog()
    df.loc[1, "has_text"] = False  # pretend id 2 has no usable text
    hybrid = HybridRecommender(df, retriever=StubRetriever(df))
    recs = hybrid.recommend(0, k=5)
    assert 1 not in recs
