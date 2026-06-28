"""Reflex entry point: a single storefront page — a recommended-products
shelf plus an AI shopping assistant, both backed by the project's
recommender (semantic search + item-to-item similarity)."""
import reflex as rx

from .storefront import StoreState, storefront

app = rx.App()
app.add_page(
    storefront,
    route="/",
    title="Atelier — AI product recommendations",
    on_load=StoreState.load_default,
)
