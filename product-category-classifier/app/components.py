"""Shared rendering helpers for the Streamlit app pages."""
import matplotlib.pyplot as plt


def confidence_bar_chart(probabilities, title=""):
    items = sorted(probabilities.items(), key=lambda kv: -kv[1])
    classes = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    bars = ax.barh(classes, values, color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    for bar, v in zip(bars, values):
        ax.text(min(v + 0.02, 0.9), bar.get_y() + bar.get_height() / 2, f"{v:.0%}", va="center", fontsize=8)
    fig.tight_layout()
    return fig
