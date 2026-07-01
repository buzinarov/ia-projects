"""Is the embedding space actually meaningful? -- the honest, measurable core.

The theme triage (src/themes.py) has no ground truth, so we can't score it
directly. What we *can* do is check whether the embeddings carry the structure
the business cares about, using the labels the dataset really has:

1. **Linear probe.** Freeze the embeddings and fit a plain logistic regression
   to predict a held-out label. If a *linear* model reads the label off the
   vectors, the geometry already encodes it. We do this for ``Recommended IND``
   (binary sentiment) and ``Department Name`` (product area), and compare against
   a **TF-IDF bag-of-words baseline** run through the identical classifier and
   split. Beating TF-IDF is the bar -- the same shape as the sibling project's
   "beat the baseline".

2. **Retrieval proxy.** For a sample of reviews, do the 3 nearest neighbors in
   embedding space share the query's department more often than a random review
   would? This is a proxy for "are similar reviews really similar", measured
   against a random-chance baseline -- and it's the same retrieval that powers
   the app's "find similar reviews".

Honesty boundary: these labels are a *stand-in* for theme quality, not a
measurement of it. We report what we can measure and say plainly what we can't.

    python -m src.evaluate            # writes artifacts/metrics_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_reviews
from .embeddings import embed_cached

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EMBED_CACHE = ARTIFACTS_DIR / "embeddings" / "review_embeddings.npy"
SUMMARY_PATH = ARTIFACTS_DIR / "metrics_summary.json"

PROBE_TARGETS = {
    "recommended": "Recommended IND",
    "department": "Department Name",
}


def _probe_scores(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
    }


def run_probe(df, embeddings, target_col, seed=42):
    """Embeddings vs TF-IDF on one label, identical classifier and split."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    mask = df[target_col].notna().to_numpy()
    texts = df["Review Text"].to_numpy()[mask]
    X_emb = embeddings[mask]
    y = df[target_col].to_numpy()[mask]

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=seed, stratify=y)

    # Baseline: TF-IDF bag-of-words, fit on the training texts only.
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_tfidf_tr = tfidf.fit_transform(texts[tr])
    X_tfidf_te = tfidf.transform(texts[te])

    # Majority-class accuracy for context.
    values, counts = np.unique(y[tr], return_counts=True)
    majority = values[counts.argmax()]
    majority_acc = float((y[te] == majority).mean())

    return {
        "n": int(len(y)),
        "n_classes": int(len(values)),
        "majority_class_accuracy": majority_acc,
        "tfidf_baseline": _probe_scores(X_tfidf_tr, X_tfidf_te, y[tr], y[te]),
        "embeddings": _probe_scores(X_emb[tr], X_emb[te], y[tr], y[te]),
    }


def retrieval_proxy(df, embeddings, label_col="Department Name", sample=2000, k=3, seed=42):
    """Top-k neighbor agreement: do a review's k nearest neighbors share its
    ``label_col`` more than a random review would? Embeddings are L2-normalized,
    so cosine similarity is a dot product. Self-matches are excluded."""
    rng = np.random.default_rng(seed)
    labels = df[label_col].to_numpy()
    valid = np.where(df[label_col].notna().to_numpy())[0]
    queries = rng.choice(valid, size=min(sample, len(valid)), replace=False)

    E = embeddings.astype(np.float32)
    hits = []
    for start in range(0, len(queries), 256):  # batch to keep the score matrix small
        q = queries[start:start + 256]
        sims = E[q] @ E.T                       # [b, n] cosine similarities
        sims[np.arange(len(q)), q] = -1.0       # never retrieve the query itself
        topk = np.argpartition(-sims, k, axis=1)[:, :k]
        for row, neigh in zip(q, topk):
            hits.append(np.mean(labels[neigh] == labels[row]))
    precision = float(np.mean(hits))

    # Random-chance baseline: probability two random reviews share a label.
    _, counts = np.unique(labels[valid], return_counts=True)
    p = counts / counts.sum()
    random_baseline = float(np.sum(p ** 2))

    return {
        "label": label_col,
        "k": k,
        "n_queries": int(len(queries)),
        f"precision@{k}": precision,
        "random_baseline": random_baseline,
        "lift_x": round(precision / random_baseline, 2) if random_baseline else None,
    }


def evaluate(sample_retrieval=2000, seed=42):
    df = load_reviews()
    embeddings = embed_cached(df["Review Text"].tolist(), EMBED_CACHE)

    probes = {name: run_probe(df, embeddings, col, seed=seed)
              for name, col in PROBE_TARGETS.items()}
    proxy = retrieval_proxy(df, embeddings, sample=sample_retrieval, seed=seed)

    summary = {
        "dataset": {"n_reviews": int(len(df)), "embedding_model": "all-MiniLM-L6-v2"},
        "linear_probe": probes,
        "retrieval_proxy": proxy,
        "honesty_note": (
            "Theme triage is unsupervised (no labels), so it is not scored. These "
            "probes test whether the embedding space encodes the labels we DO have "
            "(sentiment, department); they stand in for, but do not measure, theme quality."
        ),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


def _print_summary(summary):
    for name, p in summary["linear_probe"].items():
        b = p["tfidf_baseline"]["weighted_f1"]
        e = p["embeddings"]["weighted_f1"]
        print(f"\n[{name}]  ({p['n']:,} reviews, {p['n_classes']} classes, "
              f"majority acc {p['majority_class_accuracy']:.3f})")
        print(f"  TF-IDF baseline : acc {p['tfidf_baseline']['accuracy']:.3f}  "
              f"weighted-F1 {b:.3f}")
        print(f"  Embeddings      : acc {p['embeddings']['accuracy']:.3f}  "
              f"weighted-F1 {e:.3f}  ({(e - b):+.3f})")
    pr = summary["retrieval_proxy"]
    precision = pr["precision@{}".format(pr["k"])]
    print(f"\n[retrieval proxy]  precision@{pr['k']} {precision:.3f} "
          f"vs random {pr['random_baseline']:.3f}  ({pr['lift_x']}x)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = evaluate(sample_retrieval=args.retrieval_sample, seed=args.seed)
    _print_summary(summary)
    print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
