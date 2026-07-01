"""Guards the headline numbers against silent regressions. Reads the artifact
written by `python -m src.run_all` / `src.evaluate`; skips if it isn't there."""
import json

import pytest

from src.evaluate import SUMMARY_PATH


@pytest.fixture(scope="module")
def summary():
    if not SUMMARY_PATH.exists():
        pytest.skip("no metrics_summary.json; run `python -m src.evaluate` first")
    return json.loads(SUMMARY_PATH.read_text())


def test_retrieval_beats_chance(summary):
    proxy = summary["retrieval_proxy"]
    assert proxy["precision@3"] > proxy["random_baseline"]
    assert proxy["lift_x"] >= 2.0          # observed ~2.6x; alert if it collapses


def test_embeddings_competitive_on_department(summary):
    dept = summary["linear_probe"]["department"]
    # Embeddings should at least match TF-IDF on the semantic (department) label.
    assert dept["embeddings"]["weighted_f1"] >= dept["tfidf_baseline"]["weighted_f1"] - 0.01
    assert dept["embeddings"]["weighted_f1"] > 0.80


def test_probes_clear_majority_baseline(summary):
    for name in ("recommended", "department"):
        probe = summary["linear_probe"][name]
        assert probe["embeddings"]["accuracy"] > probe["majority_class_accuracy"]
        assert probe["tfidf_baseline"]["accuracy"] > probe["majority_class_accuracy"]
