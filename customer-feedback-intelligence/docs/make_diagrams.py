"""Renders the architecture diagram used in the README as a PNG.

Diagrams-as-code: re-run `python docs/make_diagrams.py` if the architecture
changes, so the image never silently drifts from reality.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).resolve().parent

INK = "#1f2933"
MUTED = "#52606d"
BLUE = "#4C72B0"
TEAL = "#2A9D8F"
GREEN = "#55A868"
PURPLE = "#8172B3"
AMBER = "#DD8452"
PANEL = "#f4f6f8"


def _box(ax, cx, cy, w, h, text, face, fc="white", fontsize=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=face, facecolor=fc, zorder=2,
    ))
    ax.text(cx, cy, text, ha="center", va="center", color=INK,
            fontsize=fontsize, weight=weight, zorder=3)


def _arrow(ax, p1, p2, color=MUTED):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=color, zorder=1, shrinkA=2, shrinkB=2,
    ))


def project_pipeline():
    """One-glance overview: reviews -> one shared embedding -> four uses ->
    the support console, with the honest headline numbers underneath."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(6, 5.85, "Customer-Feedback Intelligence — one embedding, four uses",
            ha="center", fontsize=13, weight="bold", color=INK)

    # Left spine: data -> embed
    _box(ax, 1.5, 4.4, 2.5, 0.8, "22,634 reviews\n(Review Text)", BLUE, fc="#eaf0f8", fontsize=9.5)
    _box(ax, 1.5, 3.0, 2.5, 0.9, "Shared encoder\nall-MiniLM-L6-v2\n384-d, cosine", TEAL,
         fc="#e6f4f1", fontsize=9.5, weight="bold")
    _arrow(ax, (1.5, 4.0), (1.5, 3.45))

    # Four uses fanning out
    uses = [
        (4.55, 4.75, GREEN, "#eaf3ec", "2D topic map", "t-SNE projection"),
        (4.55, 3.45, PURPLE, "#f0edf6", "Theme triage", "quality / fit / style /\ncomfort / value / look"),
        (4.55, 2.15, AMBER, "#fbeee3", "Similarity search", "Chroma index, cosine"),
        (4.55, 0.85, INK, PANEL, "Linear probe (eval)", "embeddings vs TF-IDF"),
    ]
    for cx, cy, edge, fill, head, body in uses:
        ax.add_patch(FancyBboxPatch(
            (cx - 1.5, cy - 0.55), 3.0, 1.05,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=2))
        ax.text(cx, cy + 0.2, head, ha="center", fontsize=10, weight="bold", color=INK, zorder=3)
        ax.text(cx, cy - 0.22, body, ha="center", fontsize=8.3, color=MUTED, zorder=3)
        _arrow(ax, (2.75, 3.0), (cx - 1.5, cy))

    # Right: the app
    _box(ax, 9.7, 3.0, 3.4, 1.5,
         "Support Console\n\nincoming review →\ntheme · sentiment ·\n3 similar cases · routing",
         TEAL, fc="#e6f4f1", fontsize=9.5, weight="bold")
    for cy in (4.75, 3.45, 2.15):
        _arrow(ax, (6.05, cy), (8.0, 3.2))

    ax.text(6, 0.18,
            "Department probe: embeddings ≥ TF-IDF (F1 0.84 vs 0.82)   ·   "
            "Retrieval precision@3 0.80 vs 0.30 chance (2.6×)   ·   "
            "Sentiment: TF-IDF wins — embeddings don't win everything",
            ha="center", fontsize=8.4, style="italic", color=MUTED)

    fig.tight_layout()
    fig.savefig(DOCS / "project_pipeline.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    project_pipeline()
    print(f"Wrote project_pipeline.png to {DOCS}")
