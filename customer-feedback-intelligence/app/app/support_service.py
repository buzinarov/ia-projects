"""Support-console service, served in-process.

Wraps the project's own components -- the shared encoder (src/embeddings.py),
the theme triage (src/themes.py), and the Chroma index (src/rag.py) -- to turn a
pasted customer review into: its theme, a k-NN sentiment estimate, the 3 most
similar past reviews, and a templated routing suggestion. One embedding call
serves all three, so the console and the offline evaluation use identical math.

No LLM: the routing/tone suggestion is a transparent rule on the assigned theme
and the retrieved sentiment, not generated prose. Honest and runs anywhere.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_reviews  # noqa: E402
from src.embeddings import embed, embed_cached  # noqa: E402
from src.rag import EMBED_CACHE, build_index  # noqa: E402
from src.themes import THEME_ANCHORS, assign_themes, build_theme_matrix  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"

# Where each theme should be routed, and a one-line stance for the reply.
ROUTING = {
    "Quality":  ("Product Quality", "Acknowledge the defect; offer a replacement or refund."),
    "Fit":      ("Sizing & Merchandising", "Share the size guide; offer a free size exchange."),
    "Style":    ("Merchandising", "Thank them for the style feedback; pass it to buying."),
    "Comfort":  ("Product", "Note the fabric/comfort feedback for the product team."),
    "Value":    ("Pricing & Promotions", "Address price perception; mention current promotions."),
    "Look":     ("Catalog & Photography", "Flag a possible color/photo mismatch on the listing."),
}


class SupportService:
    """Loaded once and shared across requests (see :func:`get_service`)."""

    def __init__(self):
        self.df = load_reviews()
        self.embeddings = embed_cached(self.df["Review Text"].tolist(), EMBED_CACHE,
                                       show_progress=False)
        self.names, self.anchors, self.owners = build_theme_matrix()
        self.collection = build_index(df=self.df, embeddings=self.embeddings)
        sampled = self.df["Review Text"].sample(6, random_state=7).tolist()
        self._examples = [
            {"short": (t[:40] + "…") if len(t) > 40 else t, "full": t}
            for t in sampled
        ]

    # -- catalog-level summaries ----------------------------------------------
    def theme_distribution(self):
        path = ARTIFACTS / "theme_distribution.json"
        if path.exists():
            dist = json.loads(path.read_text())
        else:
            themes = assign_themes(self.embeddings, self.anchors, self.owners, self.names)
            dist = dict(Counter(themes).most_common())
        total = sum(dist.values()) or 1
        out = []
        for k, v in dist.items():
            pct = round(100 * v / total, 1)
            out.append({"theme": k, "count": int(v), "pct": pct, "width": f"{pct}%"})
        return out

    def examples(self):
        return self._examples

    # -- the core interaction -------------------------------------------------
    def analyze(self, text, n=3):
        text = (text or "").strip()
        if not text:
            return None
        vec = embed([text])  # one encode, reused everywhere below

        theme, score = assign_themes(vec, self.anchors, self.owners, self.names,
                                     return_scores=True)
        theme, score = theme[0], float(score[0])

        res = self.collection.query(query_embeddings=[vec[0].tolist()], n_results=n)
        similar = [
            {
                "review": doc,
                "rating": int(meta.get("rating", 0)),
                "department": meta.get("department", ""),
                "recommended": int(meta.get("recommended", 0)),
                "similarity": round(1 - float(dist), 3),  # cosine sim from cosine distance
            }
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

        # k-NN sentiment: the majority recommend-vote among the nearest reviews.
        rec_share = np.mean([h["recommended"] for h in similar]) if similar else 0.0
        avg_rating = float(np.mean([h["rating"] for h in similar])) if similar else 0.0
        sentiment = "Positive" if rec_share >= 0.5 else "At risk"

        team, stance = ROUTING.get(theme, ("Customer Care", "Respond promptly and personally."))
        tone = "warm, appreciative" if sentiment == "Positive" else "empathetic, solution-first"

        return {
            "theme": theme,
            "theme_confidence": round(score, 3),
            "anchors": THEME_ANCHORS[theme],
            "sentiment": sentiment,
            "neighbor_recommend_share": round(float(rec_share), 2),
            "neighbor_avg_rating": round(avg_rating, 1),
            "route_to": team,
            "suggested_stance": stance,
            "suggested_tone": tone,
            "similar": similar,
        }


_service = None


def get_service():
    global _service
    if _service is None:
        _service = SupportService()
    return _service
