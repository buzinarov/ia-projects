"""Dimensionality reduction for the 2D topic map.

t-SNE projects the 384-dim review embeddings down to two dimensions so the
review space can be *seen* -- the "2D visual representation" the brief asks for.
t-SNE is O(n^2)-ish and slow on the full ~22.6k reviews, so by default we
project a random sample; the structure (sentiment poles, product-area clusters)
is already clear at a few thousand points. The full embeddings still drive the
evaluation and the live search -- the sample is only for the picture.
"""
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def apply_tsne(embeddings, sample_size=4000, seed=0):
    """Project embeddings to a 2D array. If ``sample_size`` is smaller than the
    dataset, a random subset is projected. Returns ``(coords_2d, sample_idx)``
    where ``coords_2d`` is ``[n, 2]`` and ``sample_idx`` indexes back into the
    original embeddings (the identity range when no subsampling happens)."""
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    if sample_size and sample_size < n:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    else:
        sample_idx = np.arange(n)

    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
    coords = tsne.fit_transform(embeddings[sample_idx])
    return coords.astype(np.float32), sample_idx


def plot_topic_map(coords, labels, title, out_path, legend_title="Theme"):
    """Scatter the 2D projection, colored by a categorical label (theme or
    department). Saves a PNG and returns its path."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(11, 8))
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5, label=str(label))
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title=legend_title, markerscale=2, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
