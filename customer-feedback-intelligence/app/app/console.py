"""The Support Console: paste a customer review, get its theme, a sentiment
read from the nearest past reviews, the 3 most similar reviews, and a routing
suggestion -- so an agent can triage and reply faster and more consistently.

Everything on screen comes from real reviews and the project's own retrieval;
the routing/tone line is a transparent rule, not generated text.
"""
import asyncio

import reflex as rx

from .support_service import get_service

ACCENT = "teal"

EXAMPLE_HINT = "Absolutely wonderful - silky and sexy and comfortable"

THEME_COLORS = {
    "Quality": "amber", "Fit": "indigo", "Style": "purple",
    "Comfort": "teal", "Value": "green", "Look": "pink",
}


class ConsoleState(rx.State):
    review_text: str = EXAMPLE_HINT
    result: dict = {}
    similar: list[dict[str, str]] = []
    analyzed: bool = False
    loading: bool = False
    distribution: list[dict[str, str]] = []
    examples: list[dict[str, str]] = []

    async def load_default(self):
        svc = await asyncio.to_thread(get_service)
        self.distribution = svc.theme_distribution()
        self.examples = svc.examples()

    def set_review_text(self, value: str):
        self.review_text = value

    def use_example(self, text: str):
        self.review_text = text

    async def analyze(self):
        text = self.review_text.strip()
        if not text:
            return
        self.loading = True
        yield
        svc = await asyncio.to_thread(get_service)
        result = await asyncio.to_thread(svc.analyze, text, 3)
        self.similar = [
            {
                "review": h["review"],
                "department": h["department"] or "—",
                "rating": str(h["rating"]),
                "similarity": str(h["similarity"]),
            }
            for h in result.pop("similar")
        ]
        self.result = result
        self.analyzed = True
        self.loading = False


# -- components ---------------------------------------------------------------
def _navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon("messages-square", size=22, color=rx.color(ACCENT, 9)),
            rx.heading("Customer-Feedback Intelligence", size="5", weight="bold"),
            rx.badge("Support console", color_scheme=ACCENT, variant="soft", radius="full"),
            rx.spacer(),
            rx.color_mode.button(),
            width="100%",
            align="center",
            padding="0.8em 1.6em",
        ),
        border_bottom=f"1px solid {rx.color('gray', 4)}",
        position="sticky", top="0", background=rx.color("gray", 1),
        backdrop_filter="blur(6px)", z_index="100", width="100%",
    )


def _example_chip(item: rx.Var) -> rx.Component:
    return rx.box(
        rx.text(item["short"], size="1", white_space="nowrap"),
        on_click=lambda: ConsoleState.use_example(item["full"]),
        padding="0.3em 0.7em", border_radius="999px",
        border=f"1px solid {rx.color('gray', 5)}", background=rx.color("gray", 2),
        cursor="pointer", _hover={"background": rx.color(ACCENT, 3)},
    )


def _similar_card(item: rx.Var) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.badge(item["similarity"], color_scheme=ACCENT, variant="soft"),
            rx.badge(item["department"], color_scheme="gray", variant="soft"),
            rx.spacer(),
            rx.hstack(
                rx.icon("star", size=13, color=rx.color("amber", 9)),
                rx.text(item["rating"], size="1", weight="medium"),
                spacing="1", align="center",
            ),
            width="100%", align="center",
        ),
        rx.text(item["review"], size="2", margin_top="0.4em",
                color=rx.color("gray", 11)),
        padding="0.8em", border_radius="12px",
        border=f"1px solid {rx.color('gray', 4)}", background=rx.color("gray", 1),
        width="100%",
    )


def _stat(label: str, value: rx.Var, color: str = "gray") -> rx.Component:
    return rx.vstack(
        rx.text(label, size="1", color_scheme="gray"),
        rx.badge(value, color_scheme=color, variant="soft", size="2"),
        spacing="1", align="start",
    )


def _result_panel() -> rx.Component:
    r = ConsoleState.result
    return rx.cond(
        ConsoleState.analyzed,
        rx.vstack(
            rx.hstack(
                _stat("Theme", r["theme"], ACCENT),
                _stat("Confidence", r["theme_confidence"]),
                _stat("Sentiment", r["sentiment"],
                      rx.cond(r["sentiment"] == "Positive", "green", "red")),
                _stat("Avg rating (neighbors)", r["neighbor_avg_rating"], "amber"),
                spacing="5", wrap="wrap",
            ),
            rx.box(
                rx.hstack(
                    rx.icon("route", size=16, color=rx.color(ACCENT, 9)),
                    rx.text("Route to ", size="2"),
                    rx.text(r["route_to"], size="2", weight="bold"),
                    spacing="1", align="center",
                ),
                rx.text(r["suggested_stance"], size="2", color_scheme="gray", margin_top="0.2em"),
                rx.text(rx.text.span("Suggested tone: ", weight="medium"),
                        r["suggested_tone"],
                        size="1", color_scheme="gray", margin_top="0.2em", font_style="italic"),
                padding="0.9em", border_radius="12px",
                background=rx.color(ACCENT, 2), border=f"1px solid {rx.color(ACCENT, 5)}",
                width="100%",
            ),
            rx.text("3 most similar past reviews", size="2", weight="bold",
                    margin_top="0.4em"),
            rx.foreach(ConsoleState.similar, _similar_card),
            spacing="3", align="stretch", width="100%",
        ),
        rx.center(
            rx.cond(
                ConsoleState.loading,
                rx.spinner(size="3"),
                rx.text("Paste a review and hit Analyze.", color_scheme="gray"),
            ),
            min_height="220px", width="100%",
        ),
    )


def _analyzer() -> rx.Component:
    return rx.vstack(
        rx.heading("Triage an incoming review", size="5"),
        rx.text("Assign a theme, read sentiment from the nearest past reviews, and "
                "pull the 3 most similar cases for a consistent reply.",
                size="2", color_scheme="gray"),
        rx.text_area(
            value=ConsoleState.review_text,
            on_change=ConsoleState.set_review_text,
            placeholder="Paste a customer review…",
            rows="4", width="100%", size="3",
        ),
        rx.hstack(
            rx.button(rx.icon("wand-sparkles", size=16), "Analyze",
                      on_click=ConsoleState.analyze, loading=ConsoleState.loading,
                      color_scheme=ACCENT, size="3"),
            rx.spacer(),
            width="100%",
        ),
        rx.text("Try one:", size="1", color_scheme="gray"),
        rx.hstack(rx.foreach(ConsoleState.examples, _example_chip),
                  wrap="wrap", spacing="2"),
        rx.divider(margin_y="0.6em"),
        _result_panel(),
        spacing="3", align="stretch", width="100%",
    )


def _dist_row(item: rx.Var) -> rx.Component:
    return rx.hstack(
        rx.text(item["theme"], size="1", width="64px"),
        rx.box(
            rx.box(width=item["width"], height="100%",
                   background=rx.color(ACCENT, 9), border_radius="999px"),
            width="100%", height="12px", background=rx.color("gray", 4),
            border_radius="999px", overflow="hidden",
        ),
        rx.text(item["width"], size="1", width="48px",
                color_scheme="gray", text_align="right"),
        width="100%", align="center", spacing="2",
    )


def _pulse() -> rx.Component:
    return rx.vstack(
        rx.heading("Catalog pulse", size="4"),
        rx.text("What 22,634 reviews talk about (zero-shot theme triage).",
                size="1", color_scheme="gray"),
        rx.vstack(rx.foreach(ConsoleState.distribution, _dist_row),
                  spacing="2", width="100%", margin_y="0.6em"),
        rx.divider(),
        rx.text("Review space (t-SNE)", size="2", weight="bold", margin_top="0.4em"),
        rx.image(src="/topic_map_departments.png", width="100%", border_radius="10px",
                 border=f"1px solid {rx.color('gray', 4)}"),
        rx.text("Embeddings cluster by product department — evidence the space is "
                "semantically meaningful.", size="1", color_scheme="gray"),
        spacing="2", align="stretch", width="100%",
        padding="1.2em", border_radius="16px",
        border=f"1px solid {rx.color('gray', 4)}", background=rx.color("gray", 1),
        position=rx.breakpoints(initial="static", md="sticky"), top="84px",
    )


def console() -> rx.Component:
    return rx.box(
        _navbar(),
        rx.box(
            rx.vstack(
                rx.heading("Turn raw reviews into faster, more consistent support.",
                           size="7", weight="bold", line_height="1.2"),
                rx.text("Embed every review once, then triage themes, estimate "
                        "sentiment, and retrieve similar past cases — all locally, "
                        "no API keys.", color_scheme="gray", size="3", max_width="760px"),
                spacing="3", align="start", padding_bottom="1.4em",
            ),
            rx.flex(
                rx.box(
                    rx.box(_analyzer(), padding="1.4em", border_radius="16px",
                           border=f"1px solid {rx.color('gray', 4)}",
                           background=rx.color("gray", 1)),
                    flex="1", min_width="0",
                ),
                rx.box(_pulse(), width=rx.breakpoints(initial="100%", md="380px"),
                       flex_shrink="0"),
                direction=rx.breakpoints(initial="column", md="row"),
                spacing="6", width="100%", align="start",
            ),
            max_width="1240px", margin="0 auto", padding="2em 1.6em 4em", width="100%",
        ),
        background=rx.color("gray", 2), min_height="100vh", width="100%",
    )
