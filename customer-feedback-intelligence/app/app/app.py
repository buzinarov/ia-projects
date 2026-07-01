"""Reflex entry point: the single-page Support Console -- paste a review, get
its theme, a sentiment read, the 3 most similar past reviews, and a routing
suggestion, alongside a catalog-pulse view of what customers talk about."""
import reflex as rx

from .console import ConsoleState, console

app = rx.App()
app.add_page(
    console,
    route="/",
    title="Customer-Feedback Intelligence — Support Console",
    on_load=ConsoleState.load_default,
)
