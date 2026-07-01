"""Theme triage is pure vector logic -- tested with deterministic synthetic
vectors, so these run in CI without downloading the encoder."""
import numpy as np

from src.themes import assign_themes, build_theme_matrix, theme_names


def _basis(dim, k):
    """k near-orthonormal vectors: the first k standard basis vectors."""
    v = np.zeros((k, dim), dtype=np.float32)
    for i in range(k):
        v[i, i] = 1.0
    return v


def test_assign_picks_nearest_anchor_not_index_zero():
    # A naive "min by index" selection always returns theme 0; we must return
    # the theme whose anchor is actually closest.
    names = ["A", "B", "C"]
    anchors = _basis(8, 3)              # one anchor per theme, orthogonal
    owners = np.array([0, 1, 2])
    reviews = _basis(8, 3)             # review i sits exactly on anchor i
    labels = assign_themes(reviews, anchors, owners, names)
    assert labels == ["A", "B", "C"]   # not ["A", "A", "A"]


def test_max_over_anchors_groups_by_theme():
    # Two anchors for theme 0, one for theme 1. A review near the *second*
    # anchor of theme 0 should still resolve to theme 0.
    names = ["A", "B"]
    anchors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    owners = np.array([0, 0, 1])       # anchors 0,1 -> A ; anchor 2 -> B
    review = np.array([[0, 0.9, 0.1]], dtype=np.float32)
    assert assign_themes(review, anchors, owners, names) == ["A"]


def test_scores_returned_in_unit_range():
    names = ["A", "B"]
    anchors = _basis(4, 2)
    owners = np.array([0, 1])
    reviews = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    labels, scores = assign_themes(reviews, anchors, owners, names, return_scores=True)
    assert labels == ["A", "B"]
    assert np.allclose(scores, 1.0)


def test_build_theme_matrix_with_injected_encoder():
    # Inject a fake encoder so no model is downloaded: each phrase -> a random
    # but deterministic unit vector. Just checks the shapes/owner bookkeeping.
    rng = np.random.default_rng(0)

    def fake_embed(texts):
        v = rng.standard_normal((len(texts), 16)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    names, anchors, owners = build_theme_matrix(embed_fn=fake_embed)
    assert names == theme_names()
    assert anchors.shape[0] == len(owners)
    assert set(owners.tolist()) == set(range(len(names)))  # every theme has an anchor
