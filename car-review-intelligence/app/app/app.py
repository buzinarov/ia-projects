"""Car-ing is sharing — Review Intelligence Assistant (Reflex chat UI).

A single chat surface that demonstrates the routing agent: type a message
(triage a review, translate it, ask a question grounded in a review, or
summarize one) and the assistant routes it to the right skill and shows which
skill ran. The heavy lifting lives in `src/`; this file is presentation only.
"""
import asyncio

import reflex as rx

from .assistant_service import get_service

# One source of truth for the four skills: drives the capability chips, the
# example cards, and the per-message icon so the UI never drifts from src/.
SKILLS = [
    {
        "skill": "triage",
        "icon": "gauge",
        "label": "Triage",
        "color": "amber",
        "desc": "Sentiment of a review",
        "prompt": "Is this review positive or negative? 'The transmission failed twice in the first year.'",
    },
    {
        "skill": "translate",
        "icon": "languages",
        "label": "Translate",
        "color": "blue",
        "desc": "English → Spanish",
        "prompt": "Translate to Spanish: I love this car, it is comfortable and reliable.",
    },
    {
        "skill": "answer",
        "icon": "circle-help",
        "label": "Answer",
        "color": "violet",
        "desc": "Ask about a review",
        "prompt": (
            "What did the customer like about the brand? Review: 'I chose Subaru for its "
            "reputation for safety, and it has not disappointed.'"
        ),
    },
    {
        "skill": "digest",
        "icon": "align-left",
        "label": "Digest",
        "color": "teal",
        "desc": "Summarize a long review",
        "prompt": (
            "Summarize: I have owned this SUV for two years and it has been excellent for our "
            "family. The third row is usable, the cargo is huge, and we have had no issues. The "
            "only downside is the dated infotainment system."
        ),
    },
]


class State(rx.State):
    # Messages are plain dicts (role/text/skill/via) rather than a custom model
    # class -- reflex 0.9 removed rx.Base, and dict-typed state vars are the
    # supported way to hold structured chat history.
    messages: list[dict[str, str]] = []
    draft: str = ""
    pending: bool = False
    use_ollama: bool = False  # default off: keyword routing needs no local LLM

    def set_draft(self, value: str):
        self.draft = value

    def toggle_ollama(self, value: bool):
        self.use_ollama = value

    @rx.event(background=True)
    async def send(self):
        async with self:
            message = self.draft
        await self._process(message)

    @rx.event(background=True)
    async def handle_key(self, key: str):
        if key == "Enter":
            async with self:
                message = self.draft
            await self._process(message)

    async def _process(self, message: str):
        message = (message or "").strip()
        if not message:
            return
        async with self:
            if self.pending:
                return
            use_ollama = self.use_ollama
            self.messages.append({"role": "user", "text": message, "skill": "", "via": ""})
            self.draft = ""
            self.pending = True

        try:
            # Inference is blocking; run it off the event loop so the "Routing…"
            # indicator stays responsive.
            result = await asyncio.to_thread(get_service().respond, message, use_ollama)
        except Exception as exc:  # surface failures in the chat, don't 500
            result = {"reply": f"Something went wrong: {exc}", "skill": "—", "via": "error"}

        async with self:
            self.messages.append({
                "role": "assistant",
                "text": result["reply"],
                "skill": result["skill"],
                "via": result["via"],
            })
            self.pending = False


# --- small presentational helpers ----------------------------------------

def _skill_icon(skill_var, size=13):
    """Map a skill Var to its lucide icon (falls back to a spark)."""
    return rx.match(
        skill_var,
        *[(s["skill"], rx.icon(s["icon"], size=size)) for s in SKILLS],
        rx.icon("sparkles", size=size),
    )


def capability_chip(s: dict) -> rx.Component:
    # Clickable: loads that skill's example into the input so it's always
    # possible to switch capability, even after the empty-state cards are gone.
    return rx.button(
        rx.icon(s["icon"], size=13),
        s["label"],
        on_click=lambda: State.set_draft(s["prompt"]),
        color_scheme=s["color"],
        variant="soft",
        radius="full",
        size="2",
        title=f"{s['desc']} — loads an example into the box",
    )


def example_card(s: dict) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.flex(
                rx.icon(s["icon"], size=16, color=f"var(--{s['color']}-11)"),
                align="center",
                justify="center",
                width="2rem",
                height="2rem",
                border_radius="0.6rem",
                background=f"var(--{s['color']}-3)",
                flex_shrink="0",
            ),
            rx.vstack(
                rx.text(s["label"], size="2", weight="bold"),
                rx.text(s["desc"], size="1", color="var(--gray-10)"),
                spacing="0",
                align="start",
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        on_click=lambda: State.set_draft(s["prompt"]),
        cursor="pointer",
        width="100%",
        padding="0.85rem",
        border="1px solid var(--gray-4)",
        border_radius="0.9rem",
        background="var(--gray-1)",
        transition="all 0.15s ease",
        _hover={"border_color": f"var(--{s['color']}-7)", "background": f"var(--{s['color']}-2)"},
    )


def empty_state() -> rx.Component:
    return rx.vstack(
        rx.flex(
            rx.icon("sparkles", size=26, color="var(--indigo-11)"),
            align="center", justify="center",
            width="3.4rem", height="3.4rem",
            border_radius="1rem", background="var(--indigo-3)",
        ),
        rx.heading("How can I help with your reviews?", size="5", weight="bold"),
        rx.text(
            "Pick a capability below (or the chips up top) to load an example, then press "
            "Enter — the assistant routes each message for you.",
            size="2", color="var(--gray-10)", text_align="center",
        ),
        rx.grid(
            *[example_card(s) for s in SKILLS],
            columns=rx.breakpoints(initial="1", sm="2"),
            spacing="3",
            width="100%",
            margin_top="0.5rem",
        ),
        spacing="3",
        align="center",
        width="100%",
        padding_y="2.5rem",
    )


def message_bubble(msg: dict[str, str]) -> rx.Component:
    is_user = msg["role"] == "user"
    has_skill = ~is_user & (msg["skill"] != "") & (msg["skill"] != "—")

    bubble = rx.box(
        rx.cond(
            has_skill,
            rx.hstack(
                rx.badge(_skill_icon(msg["skill"]), msg["skill"], color_scheme="indigo", variant="soft", radius="full"),
                rx.badge(msg["via"], color_scheme="gray", variant="surface", radius="full", size="1"),
                spacing="1",
                margin_bottom="0.4rem",
                align="center",
            ),
        ),
        rx.text(msg["text"], white_space="pre-wrap", size="2", line_height="1.55"),
        background=rx.cond(is_user, "var(--indigo-9)", "var(--gray-2)"),
        color=rx.cond(is_user, "white", "var(--gray-12)"),
        border=rx.cond(is_user, "none", "1px solid var(--gray-4)"),
        border_radius="1rem",
        padding="0.7rem 0.95rem",
        max_width="82%",
        box_shadow="0 1px 2px rgba(0,0,0,0.04)",
    )

    assistant_avatar = rx.flex(
        rx.icon("bot", size=16, color="white"),
        align="center", justify="center",
        width="2rem", height="2rem", flex_shrink="0",
        border_radius="0.6rem", background="var(--indigo-9)",
    )

    return rx.hstack(
        rx.cond(~is_user, assistant_avatar),
        bubble,
        width="100%",
        spacing="2",
        align="start",
        justify=rx.cond(is_user, "end", "start"),
    )


def typing_indicator() -> rx.Component:
    return rx.hstack(
        rx.flex(
            rx.icon("bot", size=16, color="white"),
            align="center", justify="center",
            width="2rem", height="2rem", flex_shrink="0",
            border_radius="0.6rem", background="var(--indigo-9)",
        ),
        rx.hstack(
            rx.spinner(size="1"),
            rx.text("Routing…", size="2", color="var(--gray-10)"),
            spacing="2", align="center",
            background="var(--gray-2)", border="1px solid var(--gray-4)",
            border_radius="1rem", padding="0.7rem 0.95rem",
        ),
        spacing="2", align="center", width="100%",
    )


def header() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.flex(
                    rx.icon("car", size=20, color="white"),
                    align="center", justify="center",
                    width="2.4rem", height="2.4rem",
                    border_radius="0.75rem", background="var(--indigo-9)",
                ),
                rx.vstack(
                    rx.text("Car-ing is sharing", size="1", color="var(--gray-10)", weight="medium"),
                    rx.heading("Review Intelligence Assistant", size="5", weight="bold"),
                    spacing="0", align="start",
                ),
                spacing="3", align="center",
            ),
            rx.spacer(),
            rx.hstack(
                rx.icon("cpu", size=14, color="var(--gray-10)"),
                rx.text("Local LLM router", size="1", color="var(--gray-11)"),
                rx.switch(checked=State.use_ollama, on_change=State.toggle_ollama, size="1"),
                spacing="2", align="center",
                padding="0.35rem 0.7rem", border="1px solid var(--gray-4)", border_radius="full",
                title="On: a local Ollama model routes. Off: fast keyword routing (no model needed).",
            ),
            width="100%", align="center",
        ),
        rx.hstack(
            *[capability_chip(s) for s in SKILLS],
            spacing="2", wrap="wrap",
        ),
        spacing="3", width="100%",
    )


def chat_panel() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.cond(
                State.messages.length() > 0,
                rx.vstack(
                    rx.foreach(State.messages, message_bubble),
                    rx.cond(State.pending, typing_indicator()),
                    spacing="3", width="100%",
                ),
                empty_state(),
            ),
            width="100%",
            flex="1",
            overflow_y="auto",
            padding="1.1rem",
        ),
        rx.box(
            rx.hstack(
                rx.input(
                    value=State.draft,
                    placeholder="Paste a review or ask a question…  (Enter to send)",
                    on_change=State.set_draft,
                    on_key_down=State.handle_key,
                    size="3",
                    width="100%",
                    radius="large",
                ),
                rx.button(
                    rx.icon("arrow-up", size=18),
                    on_click=State.send,
                    loading=State.pending,
                    size="3",
                    radius="large",
                ),
                width="100%", spacing="2", align="center",
            ),
            width="100%",
            padding="0.9rem 1.1rem",
            border_top="1px solid var(--gray-4)",
        ),
        spacing="0",
        width="100%",
        height="70vh",
        border="1px solid var(--gray-4)",
        border_radius="1.25rem",
        background="var(--color-panel-solid)",
        box_shadow="0 8px 30px rgba(0,0,0,0.06)",
        overflow="hidden",
    )


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            header(),
            chat_panel(),
            spacing="5",
            width="100%",
        ),
        max_width="820px",
        padding_x="1rem",
        padding_y="2rem",
    )


app = rx.App()
app.add_page(index, title="Review Intelligence Assistant")
