"""End-to-end pipeline: data -> embeddings -> themes -> 2D map -> index ->
evaluation -> the brief's deliverables. One command reproduces every artifact
the README, the notebooks, and the app rely on.

    python -m src.run_all

Writes, under artifacts/:
  embeddings/review_embeddings.npy   the shared embedding matrix (cached)
  chroma/                            the persistent similarity index
  topic_map_themes.png               t-SNE colored by assigned theme
  topic_map_departments.png          t-SNE colored by product department
  theme_distribution.json            how many reviews fall in each theme
  metrics_summary.json               the linear-probe + retrieval-proxy results
  most_similar_reviews.json          the brief's worked example, end to end
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import load_reviews
from .embeddings import embed_cached
from .evaluate import evaluate, _print_summary
from .rag import build_index, find_similar_reviews
from .reduce import apply_tsne, plot_topic_map
from .themes import assign_themes, build_theme_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EMBED_CACHE = ARTIFACTS_DIR / "embeddings" / "review_embeddings.npy"

EXAMPLE_REVIEW = "Absolutely wonderful - silky and sexy and comfortable"


def main(tsne_sample=4000):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/6  Loading and cleaning reviews...")
    df = load_reviews()
    texts = df["Review Text"].tolist()
    print(f"      {len(df):,} reviews.")

    print("2/6  Embedding reviews (cached)...")
    embeddings = embed_cached(texts, EMBED_CACHE)

    print("3/6  Assigning themes...")
    names, anchors, owners = build_theme_matrix()
    themes = assign_themes(embeddings, anchors, owners, names)
    distribution = dict(Counter(themes).most_common())
    (ARTIFACTS_DIR / "theme_distribution.json").write_text(json.dumps(distribution, indent=2))
    print("      " + ", ".join(f"{k}: {v:,}" for k, v in distribution.items()))

    print(f"4/6  t-SNE on a {tsne_sample}-review sample and plotting...")
    coords, sample_idx = apply_tsne(embeddings, sample_size=tsne_sample)
    sample_themes = np.asarray(themes)[sample_idx]
    sample_depts = df["Department Name"].fillna("Unknown").to_numpy()[sample_idx]
    plot_topic_map(coords, sample_themes,
                   "Review embeddings by assigned theme (t-SNE)",
                   ARTIFACTS_DIR / "topic_map_themes.png", legend_title="Theme")
    plot_topic_map(coords, sample_depts,
                   "Review embeddings by product department (t-SNE)",
                   ARTIFACTS_DIR / "topic_map_departments.png", legend_title="Department")

    print("5/6  Building the Chroma similarity index...")
    build_index(df=df, embeddings=embeddings, force_rebuild=True)

    print("6/6  Evaluating the embedding space...")
    summary = evaluate()
    _print_summary(summary)

    # The brief's worked example: the 3 reviews closest to the first one.
    similar = find_similar_reviews(EXAMPLE_REVIEW, n=3)
    most_similar_reviews = [hit["review"] for hit in similar]
    (ARTIFACTS_DIR / "most_similar_reviews.json").write_text(
        json.dumps({"query": EXAMPLE_REVIEW, "most_similar_reviews": most_similar_reviews,
                    "detail": similar}, indent=2))
    print("\nmost_similar_reviews for the example review:")
    for r in most_similar_reviews:
        print(f"  - {r[:100]}")

    print(f"\nDone. Artifacts in {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
