"""Offline evaluation of the recommenders: popularity baseline vs. the
proposed hybrid model.

Honesty boundary (see docs/requirement.md): this dataset has no
user-interaction data, so revenue and engagement are NOT measured here.
Success is measured with standard offline ranking metrics --
precision@k, recall@k, NDCG@k -- against a content-based relevance PROXY
(same articleType and gender as the query item). Every reported number
carries that caveat.

    python -m src.evaluate_reco --n-queries 1000 --ks 5 10
    python -m src.evaluate_reco --category-signal image   # full pipeline; needs checkpoints

Writes artifacts/reco_metrics_summary.json for the app and the README.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .recommender import (
    EmbeddingRetriever,
    HybridRecommender,
    ImageCategoryProvider,
    PopularityRecommender,
    load_catalog,
    relevant_mask,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
SUMMARY_PATH = ARTIFACTS_DIR / "reco_metrics_summary.json"


def precision_at_k(recommended, relevant_set, k):
    if k == 0:
        return 0.0
    hits = sum(1 for i in recommended[:k] if i in relevant_set)
    return hits / k


def recall_at_k(recommended, relevant_set, k):
    if not relevant_set:
        return 0.0
    hits = sum(1 for i in recommended[:k] if i in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended, relevant_set, k):
    """Binary-relevance NDCG@k: DCG over the top-k recommendations,
    normalized by the ideal DCG (all relevant items ranked first)."""
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def sample_query_indices(df, n_queries, seed=42):
    """Pick query anchors that have usable text and at least one relevant
    item under the proxy (otherwise the query can't score anything)."""
    rng = np.random.default_rng(seed)
    eligible = np.where(df["has_text"].to_numpy())[0]
    rng.shuffle(eligible)
    chosen = []
    for idx in eligible:
        if relevant_mask(idx, df).sum() > 0:
            chosen.append(int(idx))
        if len(chosen) >= n_queries:
            break
    return chosen


def evaluate_recommender(recommender, df, query_indices, ks):
    max_k = max(ks)
    per_metric = {f"precision@{k}": [] for k in ks}
    per_metric.update({f"recall@{k}": [] for k in ks})
    per_metric.update({f"ndcg@{k}": [] for k in ks})

    for anchor_idx in query_indices:
        relevant_set = set(np.where(relevant_mask(anchor_idx, df))[0].tolist())
        recommended = recommender.recommend(anchor_idx, k=max_k)
        for k in ks:
            per_metric[f"precision@{k}"].append(precision_at_k(recommended, relevant_set, k))
            per_metric[f"recall@{k}"].append(recall_at_k(recommended, relevant_set, k))
            per_metric[f"ndcg@{k}"].append(ndcg_at_k(recommended, relevant_set, k))

    return {
        metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for metric, values in per_metric.items()
    }


def _format_table(ks, baseline, proposed):
    lines = [f"{'metric':<14}{'baseline':>12}{'proposed':>12}{'lift':>10}"]
    lines.append("-" * 48)
    for family in ("precision", "recall", "ndcg"):
        for k in ks:
            metric = f"{family}@{k}"
            b = baseline[metric]["mean"]
            p = proposed[metric]["mean"]
            lift = (p - b) / b * 100 if b > 0 else float("nan")
            lines.append(f"{metric:<14}{b:>12.4f}{p:>12.4f}{lift:>9.1f}%")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seed", type=int, default=42, help="query-sampling seed")
    parser.add_argument(
        "--category-signal", choices=["true", "image"], default="true",
        help="'true' supplies ground-truth category (isolates retrieval quality, "
             "runs without checkpoints); 'image' uses the trained classifier (full pipeline).",
    )
    parser.add_argument("--vision-model", default="image_classifier", help="checkpoint for --category-signal image")
    args = parser.parse_args()

    print("Loading catalog...")
    df = load_catalog()
    query_indices = sample_query_indices(df, args.n_queries, seed=args.seed)
    print(f"Evaluating on {len(query_indices)} query items at k={args.ks} "
          f"(category signal: {args.category_signal}).")

    retriever = EmbeddingRetriever(df)
    if args.category_signal == "image":
        category_provider = ImageCategoryProvider(df, model_name=args.vision_model)
    else:
        category_provider = None  # HybridRecommender defaults to ground-truth category

    baseline = PopularityRecommender(df)
    proposed = HybridRecommender(df, retriever=retriever, category_provider=category_provider)

    baseline_metrics = evaluate_recommender(baseline, df, query_indices, args.ks)
    proposed_metrics = evaluate_recommender(proposed, df, query_indices, args.ks)

    table = _format_table(args.ks, baseline_metrics, proposed_metrics)
    print("\n" + table + "\n")

    summary = {
        "n_queries": len(query_indices),
        "ks": args.ks,
        "category_signal": args.category_signal,
        "relevance_proxy": "same articleType and gender (content-based; no interaction data)",
        "baseline": {"name": baseline.name, "metrics": baseline_metrics},
        "proposed": {"name": proposed.name, "metrics": proposed_metrics},
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
