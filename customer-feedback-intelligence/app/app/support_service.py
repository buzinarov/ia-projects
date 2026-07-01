"""Reply-assistant service, served in-process.

Wraps the project's own components to turn a pasted review into a ready-to-edit
reply. Each signal uses the tool that measured best for its job:

  - topic    -- embedding theme triage (src/themes.py)
  - mood     -- a TF-IDF sentiment classifier (bag-of-words beat embeddings on
                short-text sentiment in src/evaluate.py), scoring THIS review
  - similar  -- the shared-embedding Chroma index (src/rag.py)
  - draft    -- a local LLM (Ollama), grounded in the topic + mood, with a
                transparent template fallback if Ollama isn't reachable

No external API, no keys; everything runs locally.
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
from src.themes import assign_themes, build_theme_matrix  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"

# Per-theme: which team owns it, plus the body line for a happy vs. an unhappy
# customer. The draft is assembled from these -- transparent and offline, no LLM.
THEME_PLAY = {
    "Quality": {
        "team": "Product Quality",
        "up": "We take a lot of pride in how our pieces are made, so it means a lot to read this.",
        "down": "I'm truly sorry about the quality issue you ran into -- that's not the standard we hold ourselves to. I'd like to make it right with a replacement or a full refund, whichever you prefer.",
    },
    "Fit": {
        "team": "Sizing & Merchandising",
        "up": "I'm so glad the fit worked out beautifully for you!",
        "down": "I'm sorry the fit wasn't right -- sizing can be tricky. Our size guide has detailed measurements, and I'd be happy to set up a free exchange for a size that works better.",
    },
    "Style": {
        "team": "Merchandising",
        "up": "So glad you love the style -- I'll be sure to pass your kind words to our design team.",
        "down": "I'm sorry the style wasn't quite what you were hoping for. Your feedback goes straight to our buying team so we can keep improving the collection.",
    },
    "Comfort": {
        "team": "Product",
        "up": "It's wonderful to hear it feels as good as it looks -- comfort is something we really care about.",
        "down": "I'm sorry the fabric and feel didn't live up to your expectations. I've shared your note with our product team, and I'd love to help you find something more comfortable.",
    },
    "Value": {
        "team": "Pricing & Promotions",
        "up": "Great to hear it felt like real value for the price -- thank you!",
        "down": "I understand the price didn't feel worth it, and I appreciate you telling us. I'd be glad to share any current promotions that might help.",
    },
    "Look": {
        "team": "Catalog & Photography",
        "up": "Love that the color and look came through just right in person -- thank you for sharing!",
        "down": "I'm sorry the color and look differed from what you saw online. We're reviewing the product photos, and I'd be happy to help with a return or exchange.",
    },
}


OLLAMA_MODEL = "llama3.1:8b"

_SYSTEM_PROMPT = (
    "You are a warm, professional customer-care agent for an online women's "
    "clothing retailer. You write short, sincere email replies to customer "
    "reviews. Rules: 2 to 4 sentences. Speak to the customer's specific point. "
    "If they are unhappy, apologize sincerely and offer one concrete next step "
    "(a free size exchange, a replacement, a refund, the size guide, or "
    "escalation) that fits the issue; if they are happy, thank them warmly. "
    "Never invent order numbers, names, discount codes, or policies beyond a "
    "standard exchange/refund. Do not use placeholders like [Name] or [Product]. "
    "End with exactly:\nWarm regards,\nThe Customer Care Team\n"
    "Output only the reply text, nothing else."
)


def template_reply(theme, positive):
    """Offline fallback: assemble a reply from the theme and the customer's mood
    -- a greeting, a theme-appropriate body, a close. Used when Ollama is not
    reachable, so the app always produces a draft."""
    play = THEME_PLAY.get(theme, {
        "up": "Thank you so much -- it's great to hear from you.",
        "down": "I'm sorry to hear this didn't meet your expectations, and I'd like to help make it right.",
    })
    greeting = "Hi there, and thank you so much for taking the time to share your review."
    body = play["up"] if positive else play["down"]
    close = ("Thanks again for shopping with us -- it truly means a lot!"
             if positive else
             "Please reach out any time -- we're here to help and we'd love to make this right.")
    return f"{greeting}\n\n{body}\n\n{close}\n\nWarm regards,\nThe Customer Care Team"


def ollama_reply(review, theme, positive):
    """Generate the reply with a local Ollama model (no external API, no keys),
    grounded in the review, its topic, and the detected mood. Raises on any
    failure so the caller can fall back to the template."""
    import ollama

    play = THEME_PLAY.get(theme, {})
    angle = play.get("up" if positive else "down", "")
    mood = "satisfied" if positive else "dissatisfied"
    user = (
        f'Customer review:\n"{review}"\n\n'
        f"Internal notes (do not quote these verbatim):\n"
        f"- Topic: {theme}\n"
        f"- The customer appears {mood}.\n"
        f"- Suggested angle: {angle}\n\n"
        "Write the reply."
    )
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        options={"temperature": 0.7},
        keep_alive="30m",  # keep the model resident so later drafts stay ~3s
    )
    text = resp["message"]["content"].strip()
    if not text:
        raise ValueError("empty completion")
    return text


class SupportService:
    """Loaded once and shared across requests (see :func:`get_service`)."""

    def __init__(self):
        self.df = load_reviews()
        self.embeddings = embed_cached(self.df["Review Text"].tolist(), EMBED_CACHE,
                                       show_progress=False)
        self.names, self.anchors, self.owners = build_theme_matrix()
        self.collection = build_index(df=self.df, embeddings=self.embeddings)
        self._fit_sentiment()
        self._warm_ollama()
        sampled = self.df["Review Text"].sample(6, random_state=7).tolist()
        self._examples = [
            {"short": (t[:40] + "…") if len(t) > 40 else t, "full": t}
            for t in sampled
        ]

    def _fit_sentiment(self):
        """Read the customer's mood from THIS review's own text -- not from its
        neighbors, which for a topic like 'runs small' are often still-happy
        customers and would mislabel a return as positive. We use TF-IDF here on
        purpose: the project's own evaluation found bag-of-words beats embeddings
        on short-text sentiment, so the product uses the model that measured best
        (embeddings still drive topic + retrieval)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        mask = self.df["Recommended IND"].notna()
        self._tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                      min_df=3, sublinear_tf=True)
        X = self._tfidf.fit_transform(self.df.loc[mask, "Review Text"])
        y = self.df.loc[mask, "Recommended IND"].astype(int)
        self._sentiment = LogisticRegression(max_iter=2000).fit(X, y)

    def _warm_ollama(self):
        """Preload the LLM in the background so the first real draft isn't paying
        the ~15s model-load cost. Best-effort: silently does nothing if Ollama
        isn't running (the app falls back to the template)."""
        import threading

        def _warm():
            try:
                import ollama
                ollama.chat(model=OLLAMA_MODEL,
                            messages=[{"role": "user", "content": "ok"}],
                            keep_alive="30m", options={"num_predict": 1})
            except Exception:
                pass

        threading.Thread(target=_warm, daemon=True).start()

    def _mood(self, text):
        """P(customer is satisfied) for one review, from the TF-IDF classifier."""
        proba = self._sentiment.predict_proba(self._tfidf.transform([text]))[0]
        pos_col = list(self._sentiment.classes_).index(1)
        return float(proba[pos_col])

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

        theme = assign_themes(vec, self.anchors, self.owners, self.names)[0]

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

        # Customer mood, read from THIS review's own text (TF-IDF classifier).
        satisfaction = self._mood(text)
        positive = satisfaction >= 0.5
        avg_rating = float(np.mean([h["rating"] for h in similar])) if similar else 0.0

        play = THEME_PLAY.get(theme, {"team": "Customer Care"})
        priority = "High" if not positive else "Normal"

        # Draft with a local LLM (Ollama); fall back to the template if it's not
        # reachable, so the app always returns something.
        try:
            draft = ollama_reply(text, theme, positive)
            draft_source = f"Drafted locally by {OLLAMA_MODEL}"
        except Exception:
            draft = template_reply(theme, positive)
            draft_source = "Drafted from a template (Ollama not reachable)"

        return {
            "topic": theme,
            "positive": bool(positive),
            "mood_label": "Happy customer" if positive else "Needs attention",
            "priority": priority,
            "avg_rating": round(avg_rating, 1),
            "route_to": play["team"],
            "draft": draft,
            "draft_source": draft_source,
            "similar": similar,
        }


_service = None


def get_service():
    global _service
    if _service is None:
        _service = SupportService()
    return _service
