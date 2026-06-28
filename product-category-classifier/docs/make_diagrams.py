"""Renders the architecture diagrams used in the README as PNGs.

Diagrams-as-code: re-run `python docs/make_diagrams.py` if the
architecture changes, so the images never silently drift from reality.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).resolve().parent

INK = "#1f2933"
MUTED = "#52606d"
IMG_BLUE = "#4C72B0"
ATTR_ORANGE = "#DD8452"
HEAD_GREEN = "#55A868"
TOOL_PURPLE = "#8172B3"
PANEL = "#f4f6f8"


def _box(ax, cx, cy, w, h, text, face, fc="white", fontsize=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=face, facecolor=fc, zorder=2,
    ))
    ax.text(cx, cy, text, ha="center", va="center", color=INK,
            fontsize=fontsize, weight=weight, zorder=3, wrap=True)


def _arrow(ax, p1, p2, color=MUTED):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=color, zorder=1,
        shrinkA=2, shrinkB=2,
    ))


def data_product_overview():
    """One-glance overview: the four data-product lifecycle pillars across
    the top, and the recommender's query -> signals -> ranking flow below.
    The headline diagram for a recruiter skimming the README."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.7, "Multi-modal product recommender — data-product lifecycle",
            ha="center", fontsize=13, weight="bold", color=INK)

    pillars = [
        (1.65, IMG_BLUE, "#eaf0f8", "1. Descriptive", "Catalog EDA +\npopularity baseline"),
        (4.55, ATTR_ORANGE, "#fbeee3", "2. Diagnostic", "Why popularity is\ngeneric, category-blind"),
        (7.45, HEAD_GREEN, "#eaf3ec", "3. Predictive", "Hybrid: image signal\n+ metadata similarity"),
        (10.35, TOOL_PURPLE, "#f0edf6", "4. Prescriptive", "Ship + OKRs,\nvalidate via A/B test"),
    ]
    for cx, edge, fill, header, body in pillars:
        ax.add_patch(FancyBboxPatch(
            (cx - 1.3, 4.1), 2.6, 1.15,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2,
        ))
        ax.text(cx, 4.92, header, ha="center", va="center", fontsize=10.5, weight="bold", color=INK, zorder=3)
        ax.text(cx, 4.45, body, ha="center", va="center", fontsize=9, color=MUTED, zorder=3)
    for cx in (3.0, 5.9, 8.8):
        _arrow(ax, (cx, 4.67), (cx + 0.2, 4.67))

    # Flow band
    _box(ax, 1.7, 2.5, 2.7, 0.95, "Query\nphoto or description", IMG_BLUE, fc="#eaf0f8", fontsize=9.5)
    _box(ax, 6.0, 2.5, 3.6, 0.95,
         "Image signal -> subcategory\n+ similarity (MiniLM, cosine)", HEAD_GREEN, fc="#eaf3ec", fontsize=9.5)
    _box(ax, 10.3, 2.5, 2.7, 0.95, "Ranked\nrecommendations", TOOL_PURPLE, fc="#f0edf6", fontsize=9.5)
    _arrow(ax, (3.05, 2.5), (4.2, 2.5))
    _arrow(ax, (7.8, 2.5), (8.95, 2.5))

    ax.text(6, 0.95,
            "Offline result vs. popularity baseline:  precision@5  0.33 -> 0.90     NDCG@5  0.32 -> 0.91",
            ha="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(6, 0.45,
            "Relevance is a content-based proxy (articleType + gender) — no user-interaction data; revenue/engagement are motivation, not measured.",
            ha="center", fontsize=8, style="italic", color=MUTED)

    fig.tight_layout()
    fig.savefig(DOCS / "project_pipeline.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(2.75, 6.0, "Baseline — image only", ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(8.25, 6.0, "Proposed — image + attributes", ha="center", fontsize=12, weight="bold", color=INK)
    ax.axvline(5.5, color="#cbd2d9", linewidth=1, linestyle=(0, (4, 4)))

    # Baseline column
    _box(ax, 2.75, 5.25, 3.0, 0.6, "Image  3×80×80", IMG_BLUE, fc="#eaf0f8")
    _box(ax, 2.75, 4.25, 3.6, 0.7, "Conv-BN-ReLU-Pool  ×3\n(shared image trunk)", IMG_BLUE, fc="#eaf0f8")
    _box(ax, 2.75, 3.3, 3.0, 0.55, "Flatten → Linear(128)", IMG_BLUE, fc="#eaf0f8")
    _box(ax, 2.75, 2.0, 3.0, 0.6, "Classifier head", HEAD_GREEN, fc="#eaf3ec")
    _box(ax, 2.75, 0.9, 2.4, 0.6, "Subcategory", INK, fc=PANEL, weight="bold")
    for a, b in [(4.95, 4.6), (3.9, 3.575), (3.05, 2.3), (1.7, 1.2)]:
        pass
    _arrow(ax, (2.75, 4.95), (2.75, 4.6))
    _arrow(ax, (2.75, 3.9), (2.75, 3.575))
    _arrow(ax, (2.75, 3.025), (2.75, 2.3))
    _arrow(ax, (2.75, 1.7), (2.75, 1.2))

    # Proposed column
    _box(ax, 7.0, 5.25, 2.7, 0.6, "Image  3×80×80", IMG_BLUE, fc="#eaf0f8")
    _box(ax, 7.0, 4.25, 3.0, 0.7, "Conv-BN-ReLU-Pool  ×3\n(same trunk)", IMG_BLUE, fc="#eaf0f8")
    _box(ax, 7.0, 3.3, 2.7, 0.55, "Flatten → Linear(128)", IMG_BLUE, fc="#eaf0f8")
    _box(ax, 9.7, 5.25, 2.2, 0.6, "Attributes (62)", ATTR_ORANGE, fc="#fbeee3")
    _box(ax, 9.7, 4.0, 2.2, 0.7, "Linear → ReLU\n→ Dropout", ATTR_ORANGE, fc="#fbeee3")
    _box(ax, 8.25, 2.35, 1.5, 0.5, "concat", MUTED, fc="white", fontsize=10)
    _box(ax, 8.25, 1.5, 3.0, 0.6, "Classifier head", HEAD_GREEN, fc="#eaf3ec")
    _box(ax, 8.25, 0.6, 2.2, 0.55, "Subcategory", INK, fc=PANEL, weight="bold")

    _arrow(ax, (7.0, 4.95), (7.0, 4.6))
    _arrow(ax, (7.0, 3.9), (7.0, 3.575))
    _arrow(ax, (9.7, 4.95), (9.7, 4.375))
    _arrow(ax, (7.0, 3.025), (8.05, 2.6))     # image -> concat
    _arrow(ax, (9.7, 3.65), (8.55, 2.6))      # attr -> concat
    _arrow(ax, (8.25, 2.1), (8.25, 1.8))
    _arrow(ax, (8.25, 1.2), (8.25, 0.875))

    ax.text(5.5, 0.15,
            "Both models share the identical image trunk by construction — the only difference is the attribute branch.",
            ha="center", fontsize=9.5, style="italic", color=MUTED)

    fig.tight_layout()
    fig.savefig(DOCS / "model_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def system_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    _box(ax, 1.6, 4.4, 2.4, 0.6, "Product photo", IMG_BLUE, fc="#eaf0f8")
    _box(ax, 1.6, 3.4, 2.4, 0.6, "Attributes", ATTR_ORANGE, fc="#fbeee3")
    _box(ax, 4.9, 3.9, 2.4, 0.8, "Trained\nclassifier", HEAD_GREEN, fc="#eaf3ec", weight="bold")
    _box(ax, 8.4, 3.9, 3.0, 0.8, "Prediction\n+ data contract", INK, fc=PANEL)

    _arrow(ax, (2.8, 4.4), (3.7, 4.05))
    _arrow(ax, (2.8, 3.4), (3.7, 3.75))
    _arrow(ax, (6.1, 3.9), (6.9, 3.9))

    _box(ax, 1.6, 2.0, 2.4, 0.6, "Checkpoints", MUTED, fc="white")
    _box(ax, 1.6, 1.0, 2.4, 0.6, "Catalog metadata", MUTED, fc="white")
    _box(ax, 4.9, 1.0, 2.4, 0.6, "Chroma index", TOOL_PURPLE, fc="#f0edf6")
    _box(ax, 8.4, 2.0, 3.0, 0.6, "classify_product  (tool)", TOOL_PURPLE, fc="#f0edf6")
    _box(ax, 8.4, 1.0, 3.0, 0.6, "search_similar  (tool)", TOOL_PURPLE, fc="#f0edf6")

    _arrow(ax, (2.8, 2.0), (6.9, 2.0))          # checkpoints -> classify tool
    _arrow(ax, (2.8, 1.0), (3.7, 1.0))          # catalog -> chroma
    _arrow(ax, (6.1, 1.0), (6.9, 1.0))          # chroma -> search tool

    _box(ax, 6.65, 3.0, 4.2, 0.55, "Local LLM agent  ·  Ollama (llama3.1:8b)", "#b07b2e", fc="#fdf6ec", weight="bold")
    _arrow(ax, (8.4, 2.3), (7.6, 3.275))        # classify tool -> agent
    _arrow(ax, (8.4, 1.3), (7.2, 3.275))        # search tool -> agent
    ax.text(6.65, 2.55, "Chat", ha="center", fontsize=9.5, color=MUTED)

    ax.text(5.5, 5.05, "From a photo or a description to product recommendations",
            ha="center", fontsize=12.5, weight="bold", color=INK)
    ax.text(5.5, 0.35, "No external API calls — the agent runs entirely against a local model.",
            ha="center", fontsize=9.5, style="italic", color=MUTED)

    fig.tight_layout()
    fig.savefig(DOCS / "system_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    data_product_overview()
    model_architecture()
    system_architecture()
    print(f"Wrote project_pipeline.png, model_architecture.png and system_architecture.png to {DOCS}")
