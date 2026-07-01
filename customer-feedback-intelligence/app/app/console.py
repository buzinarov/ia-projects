"""Review Reply Assistant: paste a customer's review and get a ready-to-edit
reply, drafted from the review's topic, the customer's mood, and how comparable
past reviews read. A support agent's whole task -- understand it, then respond --
on one screen.

The draft is written by a local LLM (Ollama), grounded in the detected topic and
mood, with a transparent template fallback if Ollama isn't running -- no external
API and no keys.
"""
import asyncio

import reflex as rx

from .support_service import get_service

ACCENT = "teal"
EXAMPLE_HINT = "Absolutely wonderful - silky and sexy and comfortable"


class ConsoleState(rx.State):
    review_text: str = EXAMPLE_HINT
    draft: str = ""
    draft_source: str = ""
    topic: str = ""
    mood_label: str = ""
    positive: bool = True
    priority: str = ""
    route_to: str = ""
    avg_rating: str = ""
    similar: list[dict[str, str]] = []
    examples: list[dict[str, str]] = []
    distribution: list[dict[str, str]] = []
    analyzed: bool = False
    loading: bool = False

    async def load_default(self):
        svc = await asyncio.to_thread(get_service)
        self.examples = svc.examples()
        self.distribution = svc.theme_distribution()

    def set_review_text(self, value: str):
        self.review_text = value

    def set_draft(self, value: str):
        self.draft = value

    def use_example(self, text: str):
        self.review_text = text

    async def analyze(self):
        text = self.review_text.strip()
        if not text:
            return
        self.loading = True
        yield
        svc = await asyncio.to_thread(get_service)
        r = await asyncio.to_thread(svc.analyze, text, 3)
        self.topic = r["topic"]
        self.mood_label = r["mood_label"]
        self.positive = r["positive"]
        self.priority = r["priority"]
        self.route_to = r["route_to"]
        self.avg_rating = f"{r['avg_rating']}"
        self.draft = r["draft"]
        self.draft_source = r["draft_source"]
        self.similar = [
            {"review": h["review"], "department": h["department"] or "—",
             "rating": str(h["rating"])}
            for h in r["similar"]
        ]
        self.analyzed = True
        self.loading = False


# -- small building blocks ----------------------------------------------------
def _navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon("mail", size=22, color=rx.color(ACCENT, 9)),
            rx.heading("Review Reply Assistant", size="5", weight="bold"),
            rx.badge("Customer Care", color_scheme=ACCENT, variant="soft", radius="full"),
            rx.spacer(),
            rx.color_mode.button(),
            width="100%", align="center", padding="0.8em 1.6em",
        ),
        border_bottom=f"1px solid {rx.color('gray', 4)}",
        position="sticky", top="0", background=rx.color("gray", 1),
        backdrop_filter="blur(6px)", z_index="100", width="100%",
    )


def _step(n: str, label: str) -> rx.Component:
    return rx.hstack(
        rx.center(rx.text(n, size="1", weight="bold", color="white"),
                  width="20px", height="20px", border_radius="999px",
                  background=rx.color(ACCENT, 9)),
        rx.text(label, size="3", weight="bold"),
        spacing="2", align="center",
    )


def _chip(icon: str, label: rx.Var, value: rx.Var, color: str = "gray") -> rx.Component:
    return rx.hstack(
        rx.icon(icon, size=15, color=rx.color(color, 9)),
        rx.text(label, size="1", color_scheme="gray"),
        rx.badge(value, color_scheme=color, variant="soft", size="2"),
        spacing="2", align="center",
    )


def _example_chip(item: rx.Var) -> rx.Component:
    return rx.box(
        rx.text(item["short"], size="1", white_space="nowrap"),
        on_click=lambda: ConsoleState.use_example(item["full"]),
        padding="0.3em 0.7em", border_radius="999px",
        border=f"1px solid {rx.color('gray', 5)}", background=rx.color("gray", 2),
        cursor="pointer", _hover={"background": rx.color(ACCENT, 3)},
    )


def _evidence_card(item: rx.Var) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon("star", size=13, color=rx.color("amber", 9)),
                rx.text(item["rating"], size="1", weight="medium"),
                spacing="1", align="center",
            ),
            rx.badge(item["department"], color_scheme="gray", variant="soft"),
            width="100%", align="center",
        ),
        rx.text(item["review"], size="2", margin_top="0.4em", color=rx.color("gray", 11)),
        padding="0.8em", border_radius="12px",
        border=f"1px solid {rx.color('gray', 4)}", background=rx.color("gray", 1),
        width="100%",
    )


# -- the "read" + the reply ---------------------------------------------------
def _analysis_row() -> rx.Component:
    return rx.flex(
        _chip("tag", "Topic", ConsoleState.topic, ACCENT),
        _chip(rx.cond(ConsoleState.positive, "smile", "frown"),
              "Customer", ConsoleState.mood_label,
              rx.cond(ConsoleState.positive, "green", "amber")),
        _chip("flag", "Priority", ConsoleState.priority,
              rx.cond(ConsoleState.priority == "High", "red", "gray")),
        _chip("users", "Send to", ConsoleState.route_to, "indigo"),
        spacing="6", wrap="wrap",
        padding="0.9em 1em", border_radius="12px",
        background=rx.color("gray", 2), width="100%",
    )


def _reply_block() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            _step("2", "Suggested reply"),
            rx.spacer(),
            rx.text("edit before sending", size="1", color_scheme="gray", font_style="italic"),
            width="100%", align="center",
        ),
        rx.text_area(
            value=ConsoleState.draft, on_change=ConsoleState.set_draft,
            rows="9", width="100%", size="3",
            style={"lineHeight": "1.5"},
        ),
        rx.hstack(
            rx.button(rx.icon("copy", size=15), "Copy reply",
                      on_click=rx.set_clipboard(ConsoleState.draft),
                      color_scheme=ACCENT, size="2"),
            rx.spacer(),
            rx.hstack(
                rx.icon("sparkles", size=13, color=rx.color(ACCENT, 9)),
                rx.text(ConsoleState.draft_source, size="1", color_scheme="gray"),
                spacing="1", align="center",
            ),
            width="100%", align="center",
        ),
        rx.text("Based on 3 similar past reviews", size="2", weight="bold",
                margin_top="0.4em"),
        rx.foreach(ConsoleState.similar, _evidence_card),
        spacing="3", align="stretch", width="100%",
    )


def _results() -> rx.Component:
    return rx.cond(
        ConsoleState.analyzed,
        rx.vstack(_analysis_row(), _reply_block(), spacing="4",
                  align="stretch", width="100%", margin_top="0.5em"),
        rx.center(
            rx.cond(
                ConsoleState.loading,
                rx.hstack(rx.spinner(size="3"),
                          rx.text("Reading the review…", color_scheme="gray")),
                rx.text("Paste a review and hit “Draft a reply”.", color_scheme="gray"),
            ),
            min_height="140px", width="100%",
        ),
    )


def _tool_card() -> rx.Component:
    return rx.box(
        rx.vstack(
            _step("1", "Paste the customer's review"),
            rx.text_area(
                value=ConsoleState.review_text, on_change=ConsoleState.set_review_text,
                placeholder="Paste a customer review…", rows="4", width="100%", size="3",
            ),
            rx.hstack(
                rx.button(rx.icon("wand-sparkles", size=16), "Draft a reply",
                          on_click=ConsoleState.analyze, loading=ConsoleState.loading,
                          color_scheme=ACCENT, size="3"),
                rx.spacer(), width="100%",
            ),
            rx.hstack(
                rx.text("Try one:", size="1", color_scheme="gray"),
                rx.foreach(ConsoleState.examples, _example_chip),
                wrap="wrap", spacing="2", align="center",
            ),
            rx.divider(margin_y="0.4em"),
            _results(),
            spacing="3", align="stretch", width="100%",
        ),
        padding="1.6em", border_radius="16px",
        border=f"1px solid {rx.color('gray', 4)}", background=rx.color("gray", 1),
        width="100%",
    )


def _pulse_bar(item: rx.Var) -> rx.Component:
    return rx.vstack(
        rx.text(item["theme"], size="1", weight="medium"),
        rx.box(
            rx.box(width=item["width"], height="100%",
                   background=rx.color(ACCENT, 9), border_radius="999px"),
            width="100%", height="8px", background=rx.color("gray", 4),
            border_radius="999px", overflow="hidden",
        ),
        rx.text(item["width"], size="1", color_scheme="gray"),
        spacing="1", align="start", flex="1", min_width="120px",
    )


def _pulse() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon("chart-no-axes-column", size=16, color=rx.color(ACCENT, 9)),
            rx.text("What customers write about", size="2", weight="bold"),
            rx.text("across 22,634 reviews", size="1", color_scheme="gray"),
            spacing="2", align="center",
        ),
        rx.flex(rx.foreach(ConsoleState.distribution, _pulse_bar),
                spacing="5", wrap="wrap", width="100%", margin_top="0.7em"),
        padding="1.1em 1.4em", border_radius="16px",
        border=f"1px solid {rx.color('gray', 4)}", background=rx.color("gray", 1),
        width="100%",
    )


def console() -> rx.Component:
    return rx.box(
        _navbar(),
        rx.box(
            rx.vstack(
                rx.heading("Reply to a customer review in seconds.",
                           size="8", weight="bold", line_height="1.2"),
                rx.text("Paste a review — the assistant reads the topic and the "
                        "customer's mood, sees how similar feedback was handled, "
                        "and drafts a reply you can edit and send.",
                        color_scheme="gray", size="4", max_width="640px"),
                _tool_card(),
                _pulse(),
                spacing="5", align="stretch", width="100%",
                max_width="820px", margin="0 auto",
            ),
            padding="2.4em 1.6em 4em", width="100%",
        ),
        background=rx.color("gray", 2), min_height="100vh", width="100%",
    )
