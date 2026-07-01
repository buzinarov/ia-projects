"""Reflex entry point: the single-page Review Reply Assistant -- paste a
customer review and get a ready-to-edit reply, drafted from the review's topic,
the customer's mood, and how similar past feedback reads."""
import reflex as rx

from .console import ConsoleState, console

app = rx.App()
app.add_page(
    console,
    route="/",
    title="Review Reply Assistant — Customer-Feedback Intelligence",
    on_load=ConsoleState.load_default,
)
