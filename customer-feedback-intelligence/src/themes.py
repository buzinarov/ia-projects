"""Zero-shot theme triage for reviews.

The business asks a simple question: *what is this review about?* -- quality,
fit, style, comfort, value, or look. There are no theme labels in the dataset,
so we can't train a classifier; instead we describe each theme with a few anchor
phrases, embed those phrases with the same encoder as the reviews, and assign a
review to the theme whose *best-matching* anchor is closest by cosine
similarity.

Why best-matching-anchor rather than an averaged theme centroid: averaging
blurs a theme into one broad vector, which lets a couple of generic themes
("fit", "style") absorb almost everything. Scoring each anchor phrase
separately and taking the max keeps the themes sharp and the distribution
balanced -- the same reason zero-shot pipelines score hypotheses individually.

Honesty boundary (see docs/requirement.md): because there is no ground truth for
these themes, this is an **unsupervised triage aid, not a measured classifier**.
We never report a "theme accuracy". What we *do* validate, in src/evaluate.py, is
that the underlying embedding space is meaningful -- using the labels the dataset
actually has (``Recommended IND``, ``Department Name``).

Note on the reference exercise: the original DataCamp snippet selects the theme
with ``min(..., key=lambda x: x["index"])`` -- which always returns the first
theme regardless of distance. The correct selection, used here, is the nearest
anchor by cosine similarity.
"""
import numpy as np

# Each theme is described by a few short anchor phrases. More phrasings give the
# theme more ways to match a review without blurring it into one vector.
THEME_ANCHORS = {
    "Quality": [
        "the quality of the material and the construction",
        "well made, sturdy, and durable",
        "cheaply made, poorly constructed, fell apart, a defect",
        "the seams, the stitching, the zipper, the buttons",
    ],
    "Fit": [
        "how the item fits and the sizing",
        "it runs small, runs large, or is true to size",
        "the length, the cut, too tight or too loose on the body",
    ],
    "Style": [
        "the style and design of the piece",
        "flattering, elegant, fashionable, cute",
        "looks great, perfect for the season or an occasion",
    ],
    "Comfort": [
        "how comfortable the item feels to wear",
        "soft, cozy, lightweight fabric against the skin",
        "itchy, scratchy, stiff, or uncomfortable",
    ],
    "Value": [
        "the price and whether it is worth the money",
        "a great deal, affordable, worth every penny",
        "overpriced, expensive for what you get",
    ],
    "Look": [
        "the color and the pattern of the item",
        "the print, the shade, the way the colors look in person",
        "the color was different from the picture online",
    ],
}


def theme_names():
    return list(THEME_ANCHORS.keys())


def build_theme_matrix(embed_fn=None):
    """Return ``(names, anchors, owners)``: the list of theme names, an
    L2-normalized matrix with one row per *anchor phrase*, and an integer array
    mapping each anchor row to its theme's index in ``names``.

    ``embed_fn`` defaults to the project encoder but is injectable so tests can
    supply deterministic vectors without downloading the model.
    """
    if embed_fn is None:
        from .embeddings import embed as embed_fn

    names = theme_names()
    phrases, owners = [], []
    for i, name in enumerate(names):
        for phrase in THEME_ANCHORS[name]:
            phrases.append(phrase)
            owners.append(i)
    anchors = np.asarray(embed_fn(phrases), dtype=np.float32)
    return names, anchors, np.asarray(owners)


def assign_themes(embeddings, anchors, owners, names, return_scores=False):
    """Assign each row of ``embeddings`` to a theme by best-matching anchor.

    Cosine similarity to every anchor (a dot product, since both sides are
    L2-normalized), reduced to a per-theme score by taking the max over that
    theme's anchors, then argmax over themes. Returns theme names (and,
    optionally, the winning similarity per review).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    owners = np.asarray(owners)
    n_themes = len(names)

    anchor_sims = embeddings @ anchors.T  # [n_reviews, n_anchors]
    theme_sims = np.full((len(embeddings), n_themes), -np.inf, dtype=np.float32)
    for t in range(n_themes):
        cols = owners == t
        theme_sims[:, t] = anchor_sims[:, cols].max(axis=1)

    idx = theme_sims.argmax(axis=1)
    labels = [names[i] for i in idx]
    if return_scores:
        return labels, theme_sims[np.arange(len(idx)), idx]
    return labels


def categorize_review(text, theme_index=None, embed_fn=None):
    """Convenience for a single raw review string -> theme name (used by the app
    and the notebook). ``theme_index`` is the ``(names, anchors, owners)`` tuple
    from :func:`build_theme_matrix`; built on demand if not supplied."""
    if embed_fn is None:
        from .embeddings import embed as embed_fn
    if theme_index is None:
        theme_index = build_theme_matrix(embed_fn)
    names, anchors, owners = theme_index
    vec = embed_fn([text])
    return assign_themes(vec, anchors, owners, names)[0]
