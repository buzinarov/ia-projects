"""Quality Monitoring: a single page with every metric that matters,
read top to bottom -- headline KPIs first, technical detail after.
Numbers are aggregated across all training seeds (mean +/- std), not a
single lucky run.
"""
import numpy as np
import reflex as rx

from ..backend import ARTIFACTS_DIR, generate_verdict, image_data_uri, load_history, load_metrics_summary
from ..layout import page

BASELINE_SUMMARY = load_metrics_summary("baseline")
PROPOSED_SUMMARY = load_metrics_summary("proposed")
DEMO_SEED = 0

VERDICT = generate_verdict(BASELINE_SUMMARY, PROPOSED_SUMMARY) if BASELINE_SUMMARY and PROPOSED_SUMMARY else ""

CLASS_NAMES = PROPOSED_SUMMARY["class_names"] if PROPOSED_SUMMARY else []
PER_CLASS_ROWS = sorted(
    (
        {
            "category": c,
            "support": PROPOSED_SUMMARY["per_class_support"][c],
            "baseline_f1": round(BASELINE_SUMMARY["per_class_f1"][c]["mean"], 3),
            "proposed_f1": round(PROPOSED_SUMMARY["per_class_f1"][c]["mean"], 3),
            "delta": round(
                PROPOSED_SUMMARY["per_class_f1"][c]["mean"] - BASELINE_SUMMARY["per_class_f1"][c]["mean"], 3
            ),
        }
        for c in CLASS_NAMES
    ),
    key=lambda row: -row["support"],
) if BASELINE_SUMMARY and PROPOSED_SUMMARY else []

CONFUSION_BASELINE_URI = image_data_uri(ARTIFACTS_DIR / "confusion_matrix_baseline_summary.png")
CONFUSION_PROPOSED_URI = image_data_uri(ARTIFACTS_DIR / "confusion_matrix_proposed_summary.png")

BASELINE_TEST_CONF = np.array(BASELINE_SUMMARY["test_confidences_all_seeds"]) if BASELINE_SUMMARY else np.array([])
PROPOSED_TEST_CONF = np.array(PROPOSED_SUMMARY["test_confidences_all_seeds"]) if PROPOSED_SUMMARY else np.array([])

BASELINE_HISTORY = load_history("baseline", seed=DEMO_SEED)
PROPOSED_HISTORY = load_history("proposed", seed=DEMO_SEED)


def _history_chart_data(history):
    if not history:
        return []
    return [
        {"epoch": e["epoch"], "train_loss": round(e["train_loss"], 4), "val_loss": round(e["val_loss"], 4)}
        for e in history["epochs"]
    ]


class MonitoringState(rx.State):
    threshold: float = 0.85

    def set_threshold(self, value: float):
        self.threshold = value

    @rx.var
    def baseline_auto_tag_rate(self) -> float:
        if BASELINE_TEST_CONF.size == 0:
            return 0.0
        return float((BASELINE_TEST_CONF >= self.threshold).mean())

    @rx.var
    def proposed_auto_tag_rate(self) -> float:
        if PROPOSED_TEST_CONF.size == 0:
            return 0.0
        return float((PROPOSED_TEST_CONF >= self.threshold).mean())


def _kpi_card(label: str, baseline_value: str, proposed_value: str) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.text(label, size="2", color="gray", weight="medium"),
            rx.hstack(
                rx.vstack(rx.text("Baseline", size="1", color="gray"), rx.heading(baseline_value, size="6"), spacing="0"),
                rx.divider(orientation="vertical"),
                rx.vstack(rx.text("Proposed", size="1", color="gray"), rx.heading(proposed_value, size="6"), spacing="0"),
                spacing="4",
                align="center",
            ),
            spacing="2",
            align="start",
        ),
        width="100%",
    )


def _per_class_row(row: dict) -> rx.Component:
    delta = row["delta"]
    return rx.table.row(
        rx.table.cell(row["category"]),
        rx.table.cell(row["support"]),
        rx.table.cell(f"{row['baseline_f1']:.3f}"),
        rx.table.cell(f"{row['proposed_f1']:.3f}"),
        rx.table.cell(
            rx.badge(
                f"{delta:+.3f}",
                color_scheme="green" if delta >= 0 else "red",
                variant="soft",
            )
        ),
    )


def quality_monitoring() -> rx.Component:
    if not BASELINE_SUMMARY or not PROPOSED_SUMMARY:
        return page(
            rx.heading("Quality Monitoring", size="7"),
            rx.callout(
                "No aggregated metrics found yet. Run `python -m src.run_all --seeds 0 1 2` first.",
                icon="triangle_alert",
                color_scheme="amber",
            ),
        )

    return page(
        rx.heading("Quality Monitoring", size="7"),
        rx.text(
            f"Aggregated across {len(PROPOSED_SUMMARY['seeds'])} training seeds "
            f"{PROPOSED_SUMMARY['seeds']} -- not a single lucky run.",
            color="gray",
        ),
        rx.callout(VERDICT, icon="info", color_scheme="indigo", size="3"),
        rx.grid(
            _kpi_card(
                "Accuracy (mean)",
                f"{BASELINE_SUMMARY['accuracy']['mean']:.1%}",
                f"{PROPOSED_SUMMARY['accuracy']['mean']:.1%}",
            ),
            _kpi_card(
                "Macro F1 (mean)",
                f"{BASELINE_SUMMARY['macro_f1']['mean']:.3f}",
                f"{PROPOSED_SUMMARY['macro_f1']['mean']:.3f}",
            ),
            _kpi_card(
                "Weighted F1 (mean)",
                f"{BASELINE_SUMMARY['weighted_f1']['mean']:.3f}",
                f"{PROPOSED_SUMMARY['weighted_f1']['mean']:.3f}",
            ),
            columns="3",
            spacing="4",
            width="100%",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Operational view: auto-tag rate", size="4"),
                rx.text(
                    "Share of products each model would tag without a human review, at the "
                    "confidence threshold below.",
                    color="gray",
                    size="2",
                ),
                rx.hstack(
                    rx.text("Confidence threshold:", size="2"),
                    rx.text(f"{MonitoringState.threshold:.2f}", weight="bold", size="2"),
                    spacing="2",
                ),
                rx.slider(
                    default_value=[85],
                    min=50,
                    max=99,
                    on_value_commit=lambda v: MonitoringState.set_threshold(v[0] / 100),
                    width="100%",
                ),
                rx.hstack(
                    rx.vstack(
                        rx.text("Baseline auto-tag rate", size="2", color="gray"),
                        rx.heading(f"{MonitoringState.baseline_auto_tag_rate * 100:.1f}%", size="6"),
                        spacing="0",
                    ),
                    rx.vstack(
                        rx.text("Proposed auto-tag rate", size="2", color="gray"),
                        rx.heading(f"{MonitoringState.proposed_auto_tag_rate * 100:.1f}%", size="6"),
                        spacing="0",
                    ),
                    spacing="6",
                ),
                spacing="3",
                align="stretch",
            ),
            width="100%",
        ),
        rx.card(
            rx.vstack(
                rx.heading(f"Per-class F1 ({len(CLASS_NAMES)} classes)", size="4"),
                rx.text("Sorted by test-set support.", color="gray", size="2"),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Category"),
                            rx.table.column_header_cell("Support"),
                            rx.table.column_header_cell("Baseline F1"),
                            rx.table.column_header_cell("Proposed F1"),
                            rx.table.column_header_cell("Delta"),
                        )
                    ),
                    rx.table.body(*[_per_class_row(row) for row in PER_CLASS_ROWS]),
                    width="100%",
                ),
                spacing="3",
                align="stretch",
                max_height="500px",
                overflow_y="auto",
            ),
            width="100%",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Confusion matrices (summed across seeds)", size="4"),
                rx.flex(
                    rx.image(src=CONFUSION_BASELINE_URI, width="100%"),
                    rx.image(src=CONFUSION_PROPOSED_URI, width="100%"),
                    spacing="4",
                    wrap="wrap",
                ),
                spacing="3",
                align="stretch",
            ),
            width="100%",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Training curves", size="4"),
                rx.text(f"Representative run: seed {DEMO_SEED}.", color="gray", size="2"),
                rx.flex(
                    rx.vstack(
                        rx.text("Baseline", weight="medium"),
                        rx.recharts.line_chart(
                            rx.recharts.line(data_key="train_loss", stroke="#94a3b8", name="train_loss"),
                            rx.recharts.line(data_key="val_loss", stroke="#6366f1", name="val_loss"),
                            rx.recharts.x_axis(data_key="epoch"),
                            rx.recharts.y_axis(),
                            rx.recharts.legend(),
                            data=_history_chart_data(BASELINE_HISTORY),
                            height=260,
                            width=420,
                        ),
                        align="stretch",
                    ),
                    rx.vstack(
                        rx.text("Proposed", weight="medium"),
                        rx.recharts.line_chart(
                            rx.recharts.line(data_key="train_loss", stroke="#94a3b8", name="train_loss"),
                            rx.recharts.line(data_key="val_loss", stroke="#6366f1", name="val_loss"),
                            rx.recharts.x_axis(data_key="epoch"),
                            rx.recharts.y_axis(),
                            rx.recharts.legend(),
                            data=_history_chart_data(PROPOSED_HISTORY),
                            height=260,
                            width=420,
                        ),
                        align="stretch",
                    ),
                    spacing="4",
                    wrap="wrap",
                ),
                spacing="3",
                align="stretch",
            ),
            width="100%",
        ),
        rx.divider(),
        rx.hstack(
            rx.text("Model version: v2-subcategory", size="1", color="gray"),
            rx.text("•", color="gray"),
            rx.text(
                f"Last trained: {PROPOSED_HISTORY['trained_at'][:19].replace('T', ' ')} UTC"
                if PROPOSED_HISTORY else "Last trained: unknown",
                size="1",
                color="gray",
            ),
            spacing="2",
        ),
    )
