"""Car-ing is sharing — Review Intelligence Assistant (Reflex chat UI).

A single chat surface that demonstrates the routing agent: type a message
(triage a review, translate it, ask a question grounded in a review, or
summarize one) and the assistant routes it to the right skill and shows which
skill ran. The heavy lifting lives in `src/`; this file is presentation only.
"""
import reflex as rx

from .assistant_service import get_service

EXAMPLES = [
    "Is this review positive or negative? 'The transmission failed twice in the first year.'",
    "Translate to Spanish: I love this car, it is comfortable and reliable.",
    "What did the customer like about the brand? Review: 'I chose Subaru for its reputation for safety, and it has not disappointed.'",
    "Summarize: I have owned this SUV for two years and it has been excellent for our family. The third row is usable, the cargo is huge, and we have had no issues. The only downside is the dated infotainment system.",
]


class State(rx.State):
    # Messages are plain dicts (role/text/skill/via) rather than a custom
    # model class -- reflex 0.9 removed rx.Base, and dict-typed state vars are
    # the supported way to hold structured chat history.
    messages: list[dict[str, str]] = []
    draft: str = ""
    pending: bool = False
    use_ollama: bool = True

    def set_draft(self, value: str):
        self.draft = value

    def toggle_ollama(self, value: bool):
        self.use_ollama = value

    def use_example(self, text: str):
        self.draft = text

    @rx.event(background=True)
    async def send(self):
        message = self.draft.strip()
        if not message or self.pending:
            return
        async with self:
            self.messages.append({"role": "user", "text": message, "skill": "", "via": ""})
            self.draft = ""
            self.pending = True

        try:
            result = get_service().respond(message, use_ollama=self.use_ollama)
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


def message_bubble(msg: dict[str, str]) -> rx.Component:
    is_user = msg["role"] == "user"
    return rx.box(
        rx.cond(
            ~is_user & (msg["skill"] != "") & (msg["skill"] != "—"),
            rx.hstack(
                rx.badge(msg["skill"], color_scheme="indigo", variant="soft"),
                rx.badge(msg["via"], color_scheme="gray", variant="surface"),
                spacing="1",
                margin_bottom="0.35rem",
            ),
        ),
        rx.text(msg["text"], white_space="pre-wrap"),
        background=rx.cond(is_user, "var(--indigo-3)", "var(--gray-2)"),
        border=rx.cond(is_user, "1px solid var(--indigo-5)", "1px solid var(--gray-4)"),
        border_radius="12px",
        padding="0.75rem 1rem",
        max_width="80%",
        margin_left=rx.cond(is_user, "auto", "0"),
        margin_bottom="0.6rem",
    )


def example_chip(text: str) -> rx.Component:
    return rx.button(
        rx.text(text, size="1", no_of_lines=1),
        on_click=lambda: State.use_example(text),
        variant="surface",
        color_scheme="gray",
        width="100%",
        justify="start",
    )


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Car-ing is sharing — Review Intelligence Assistant", size="6"),
            rx.text(
                "Routes each message to one of four skills: triage (sentiment), translate "
                "(EN→ES), answer (extractive QA), digest (summarize).",
                color="var(--gray-10)", size="2",
            ),
            rx.hstack(
                rx.switch(checked=State.use_ollama, on_change=State.toggle_ollama),
                rx.text("Use local Ollama router (off = keyword fallback)", size="2"),
                spacing="2",
                align="center",
            ),
            rx.divider(),
            rx.text("Try an example:", size="2", weight="bold"),
            rx.vstack(*[example_chip(e) for e in EXAMPLES], spacing="1", width="100%"),
            rx.divider(),
            rx.box(
                rx.foreach(State.messages, message_bubble),
                rx.cond(State.pending, rx.text("Routing…", color="var(--gray-9)", size="2")),
                min_height="220px",
                width="100%",
            ),
            rx.hstack(
                rx.input(
                    value=State.draft,
                    placeholder="Paste a review or ask a question…",
                    on_change=State.set_draft,
                    width="100%",
                ),
                rx.button("Send", on_click=State.send, loading=State.pending),
                width="100%",
                spacing="2",
            ),
            spacing="3",
            width="100%",
        ),
        max_width="760px",
        padding_y="2rem",
    )


app = rx.App()
app.add_page(index, title="Review Intelligence Assistant")
