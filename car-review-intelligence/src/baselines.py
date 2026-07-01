"""The baselines the transformer skills must beat (or be replaced by).

A skill that can't clear a dumb baseline isn't worth its latency. Two
baselines are defined, matching the acceptance bar in docs/requirement.md:

  - Triage: VADER lexicon sentiment, plus a majority-class floor.
  - Digest: lead-3 (first sentences) extractive summary.

These are intentionally simple and dependency-light so the comparison is
honest: if the transformer wins, it wins against a real, reasonable
alternative, not a strawman.
"""
import re
from functools import lru_cache


# --- Triage baselines ----------------------------------------------------

@lru_cache(maxsize=1)
def _vader():
    """Lazily build VADER, downloading its lexicon on first use."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()


def vader_sentiment(text):
    """Return 'POSITIVE'/'NEGATIVE' from VADER's compound score.

    Ties (compound == 0, e.g. empty or neutral text) break to POSITIVE,
    matching the dataset's positive skew -- a deliberate, stated choice so
    the baseline isn't accidentally handicapped on neutral rows.
    """
    score = _vader().polarity_scores(text)["compound"]
    return "POSITIVE" if score >= 0 else "NEGATIVE"


def majority_class_predictor(train_labels):
    """Return a function predicting the most common training label for any
    input -- the accuracy floor any real classifier must clear."""
    from collections import Counter

    majority = Counter(train_labels).most_common(1)[0][0]
    return lambda _text: majority


# --- Digest baseline -----------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def lead_n_summary(text, n=3):
    """Extractive baseline: the first `n` sentences of the review.

    'Lead-3' is a famously strong summarization baseline -- abstractive
    models routinely fail to beat it -- which is exactly why it's the bar.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return " ".join(sentences[:n])
