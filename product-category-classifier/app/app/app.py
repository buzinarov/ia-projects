"""Reflex entry point: registers every page and the landing content."""
import reflex as rx

from .layout import page
from .pages.ask_catalog import ask_catalog
from .pages.live_demo import LiveDemoState, live_demo
from .pages.quality_monitoring import quality_monitoring

ARCHITECTURE_DIAGRAM = """\
Product photo ──┐
                 ├──► CNN image branch ──┐
Attributes ──────┘    + MLP branch ──────┼──► Classifier ──► Subcategory
                                          │
Trained checkpoints ─────────────────────┼──► classify_product (tool)
Catalog metadata ──► Chroma index ───────┼──► search_similar_products (tool)
                                          │
                              Local LLM (Ollama, llama3.1:8b) ──► Chat
"""


def index() -> rx.Component:
    return page(
        rx.heading("Product Category Classifier", size="8"),
        rx.text(
            "A multi-modal computer vision case study, evaluated with statistical rigor, "
            "with a local LLM agent layered on top.",
            color="gray",
            size="4",
        ),
        rx.card(
            rx.vstack(
                rx.heading("What this is", size="5"),
                rx.text(
                    "An e-commerce catalog needs every new product classified against a fixed "
                    "schema -- the result feeds other ML models and executive/operational "
                    "reports, so it has to be both accurate and reliable. This project trains a "
                    "model to predict a product's subcategory from its photo and a set of "
                    "structured attributes, benchmarks it honestly against an image-only "
                    "baseline across multiple training runs, and wraps the result in a small "
                    "local agent and a data contract with automated tests -- so it's something "
                    "you could actually hand to an operations team, not just a notebook metric."
                ),
                spacing="3",
                align="start",
            ),
            width="100%",
        ),
        rx.card(
            rx.vstack(
                rx.heading("How it's put together", size="5"),
                rx.el.pre(
                    ARCHITECTURE_DIAGRAM,
                    font_family="var(--code-font-family)",
                    font_size="0.85em",
                    background="var(--gray-2)",
                    padding="1em",
                    border_radius="8px",
                    overflow_x="auto",
                    width="100%",
                ),
                spacing="3",
                align="stretch",
            ),
            width="100%",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Pages", size="5"),
                rx.text("• ", rx.link("Live Demo", href="/live-demo"), " -- pick a sample or upload a photo, compare both models."),
                rx.text("• ", rx.link("Quality Monitoring", href="/quality-monitoring"), " -- the full baseline-vs-proposed comparison, technical and operational."),
                rx.text("• ", rx.link("Ask the Catalog", href="/ask-catalog"), " -- chat with the local agent."),
                spacing="2",
                align="start",
            ),
            width="100%",
        ),
        rx.text(
            "No external API calls anywhere in this app -- the agent runs entirely against a "
            "local Ollama model.",
            color="gray",
            size="2",
        ),
    )


app = rx.App()
app.add_page(index, route="/", title="Product Category Classifier")
app.add_page(live_demo, route="/live-demo", title="Live Demo", on_load=LiveDemoState.load_initial)
app.add_page(quality_monitoring, route="/quality-monitoring", title="Quality Monitoring")
app.add_page(ask_catalog, route="/ask-catalog", title="Ask the Catalog")
