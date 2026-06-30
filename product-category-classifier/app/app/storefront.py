"""The storefront: a single page that pairs a recommended-products shelf
with an AI shopping assistant.

Two market-standard recommendation patterns drive it:
  - semantic text search (describe what you want)            -> the catalog, cosine-ranked
  - item-to-item "find similar" (more like this product)     -> hybrid ranking

Every product shown is a real catalog item with its real photo; the chat
assistant's prose is grounded in the retrieved results.
"""
import asyncio

import reflex as rx

from .recommender_service import assistant_reply, get_service

STORE_NAME = "Atelier"
ACCENT = "indigo"

WELCOME = (
    "Hi! Describe what you're looking for — a color, an occasion, a kind of "
    "item — and I'll find it in the catalog. You can also tap “Find similar” "
    "on any product to see more like it."
)


class StoreState(rx.State):
    products: list[dict[str, str]] = []
    preview: list[dict[str, str]] = []  # top results echoed inside the chat
    heading: str = "Recommended for you"
    subheading: str = "A curated shelf across the catalog"
    messages: list[dict[str, str]] = [{"role": "assistant", "content": WELCOME}]
    query: str = ""
    loading: bool = False

    async def load_default(self):
        self.loading = True
        yield
        svc = await asyncio.to_thread(get_service)
        self.products = await asyncio.to_thread(svc.default_shelf, 12)
        self.loading = False

    def set_query(self, value: str):
        self.query = value

    async def send(self):
        q = self.query.strip()
        if not q:
            return
        self.messages = self.messages + [{"role": "user", "content": q}]
        self.query = ""
        self.loading = True
        yield

        svc = await asyncio.to_thread(get_service)
        products = await asyncio.to_thread(svc.search, q, 12)
        reply = await asyncio.to_thread(assistant_reply, q, products)

        self.products = products
        self.preview = products[:4]
        self.heading = f"Results for “{q}”"
        self.subheading = f"{len(products)} semantically-ranked matches"
        self.messages = self.messages + [{"role": "assistant", "content": reply}]
        self.loading = False

    async def find_similar(self, idx: str, name: str):
        self.loading = True
        self.messages = self.messages + [
            {"role": "assistant", "content": f"Here are items similar to {name}."}
        ]
        yield
        svc = await asyncio.to_thread(get_service)
        products = await asyncio.to_thread(svc.similar, idx, 12)
        self.products = products
        self.preview = products[:4]
        self.heading = f"Similar to {name}"
        self.subheading = "Item-to-item recommendations"
        self.loading = False


# -- components ------------------------------------------------------------
def _navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon("shopping-bag", size=22, color=rx.color(ACCENT, 9)),
                rx.heading(STORE_NAME, size="6", weight="bold"),
                rx.badge("AI recommendations", color_scheme=ACCENT, variant="soft", radius="full"),
                spacing="3",
                align="center",
            ),
            rx.spacer(),
            rx.color_mode.button(),
            width="100%",
            align="center",
            padding="0.9em 1.6em",
        ),
        border_bottom=f"1px solid {rx.color('gray', 4)}",
        position="sticky",
        top="0",
        background=rx.color("gray", 1),
        backdrop_filter="blur(6px)",
        z_index="100",
        width="100%",
    )


def _product_card(product: rx.Var) -> rx.Component:
    return rx.box(
        rx.box(
            rx.image(
                src=product["image"],
                width="100%",
                height="180px",
                object_fit="contain",
            ),
            background="white",
            padding="0.6em",
            border_radius="10px",
            border=f"1px solid {rx.color('gray', 3)}",
        ),
        rx.vstack(
            rx.text(
                product["name"],
                weight="medium",
                size="2",
                height="2.6em",
                overflow="hidden",
                style={"display": "-webkit-box", "-webkit-line-clamp": "2", "-webkit-box-orient": "vertical"},
            ),
            rx.hstack(
                rx.badge(product["subcategory"], color_scheme=ACCENT, variant="soft"),
                rx.badge(product["gender"], color_scheme="gray", variant="soft"),
                spacing="1",
                wrap="wrap",
            ),
            rx.button(
                rx.icon("sparkles", size=14),
                "Find similar",
                on_click=lambda: StoreState.find_similar(product["idx"], product["name"]),
                variant="soft",
                color_scheme=ACCENT,
                size="1",
                width="100%",
            ),
            spacing="2",
            align="stretch",
            width="100%",
            padding_top="0.6em",
        ),
        padding="0.7em",
        border_radius="14px",
        border=f"1px solid {rx.color('gray', 3)}",
        background=rx.color("gray", 1),
        transition="box-shadow 0.15s ease, transform 0.15s ease",
        _hover={"box_shadow": "0 8px 24px rgba(0,0,0,0.08)", "transform": "translateY(-2px)"},
    )


def _product_grid() -> rx.Component:
    return rx.vstack(
        rx.vstack(
            rx.heading(StoreState.heading, size="6"),
            rx.text(StoreState.subheading, color_scheme="gray", size="2"),
            spacing="1",
            align="start",
        ),
        rx.cond(
            StoreState.products,
            rx.grid(
                rx.foreach(StoreState.products, _product_card),
                columns=rx.breakpoints(initial="2", sm="3", lg="4"),
                spacing="4",
                width="100%",
            ),
            rx.center(
                rx.cond(
                    StoreState.loading,
                    rx.spinner(size="3"),
                    rx.text("No products to show yet.", color_scheme="gray"),
                ),
                width="100%",
                min_height="300px",
            ),
        ),
        spacing="4",
        align="stretch",
        width="100%",
    )


def _chat_bubble(message: rx.Var) -> rx.Component:
    is_user = message["role"] == "user"
    return rx.box(
        rx.text(message["content"], size="2"),
        background=rx.cond(is_user, rx.color(ACCENT, 9), rx.color("gray", 3)),
        color=rx.cond(is_user, "white", rx.color("gray", 12)),
        padding="0.6em 0.85em",
        border_radius="12px",
        max_width="85%",
        align_self=rx.cond(is_user, "flex-end", "flex-start"),
    )


def _preview_item(product: rx.Var) -> rx.Component:
    return rx.hstack(
        rx.image(
            src=product["image"],
            width="44px",
            height="44px",
            object_fit="contain",
            background="white",
            border_radius="6px",
            border=f"1px solid {rx.color('gray', 4)}",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.text(
                product["name"],
                size="1",
                weight="medium",
                white_space="nowrap",
                overflow="hidden",
                text_overflow="ellipsis",
                width="100%",
            ),
            rx.text(product["subcategory"], size="1", color_scheme="gray"),
            spacing="0",
            align="start",
            min_width="0",
            flex="1",
        ),
        on_click=lambda: StoreState.find_similar(product["idx"], product["name"]),
        spacing="2",
        align="center",
        width="100%",
        padding="0.35em 0.5em",
        border_radius="8px",
        background=rx.color("gray", 2),
        cursor="pointer",
        _hover={"background": rx.color(ACCENT, 3)},
    )


def _chat_preview() -> rx.Component:
    return rx.vstack(
        rx.text("Top matches", size="1", weight="bold", color_scheme="gray"),
        rx.foreach(StoreState.preview, _preview_item),
        spacing="1",
        width="100%",
        align="stretch",
    )


def _chat_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("bot", size=18, color=rx.color(ACCENT, 9)),
                rx.heading("Shopping assistant", size="4"),
                spacing="2",
                align="center",
            ),
            rx.text(
                "AI-powered semantic search over the catalog.",
                color_scheme="gray",
                size="1",
            ),
            rx.divider(),
            rx.vstack(
                rx.foreach(StoreState.messages, _chat_bubble),
                rx.cond(StoreState.preview, _chat_preview()),
                rx.cond(
                    StoreState.loading,
                    rx.hstack(rx.spinner(size="1"), rx.text("Searching…", color_scheme="gray", size="1")),
                ),
                spacing="3",
                align="stretch",
                width="100%",
                overflow_y="auto",
                flex="1",
                min_height="240px",
                max_height="46vh",
                padding_right="0.3em",
            ),
            rx.form(
                rx.hstack(
                    rx.input(
                        value=StoreState.query,
                        on_change=StoreState.set_query,
                        placeholder="Describe what you're looking for…",
                        width="100%",
                        size="3",
                    ),
                    rx.button(
                        rx.icon("send", size=16),
                        type="submit",
                        loading=StoreState.loading,
                        size="3",
                        color_scheme=ACCENT,
                    ),
                    spacing="2",
                    width="100%",
                ),
                on_submit=StoreState.send,
                reset_on_submit=True,
                width="100%",
            ),
            spacing="3",
            align="stretch",
            height="100%",
        ),
        padding="1.2em",
        border_radius="16px",
        border=f"1px solid {rx.color('gray', 4)}",
        background=rx.color("gray", 1),
        position=rx.breakpoints(initial="static", md="sticky"),
        top="84px",
        width="100%",
    )


def storefront() -> rx.Component:
    return rx.box(
        _navbar(),
        rx.box(
            rx.vstack(
                rx.heading("Find your next favorite — describe it, or discover more like the ones you love.",
                           size="7", weight="bold", line_height="1.2"),
                rx.text(
                    "A recommender that pairs image-trained category signals with semantic "
                    "similarity over the product catalog — surfaced through a single shelf and a chat assistant.",
                    color_scheme="gray",
                    size="3",
                    max_width="760px",
                ),
                spacing="3",
                align="start",
                padding_bottom="1.5em",
            ),
            rx.flex(
                rx.box(_product_grid(), flex="1", min_width="0"),
                rx.box(
                    _chat_panel(),
                    width=rx.breakpoints(initial="100%", md="360px"),
                    flex_shrink="0",
                ),
                direction=rx.breakpoints(initial="column", md="row"),
                spacing="6",
                width="100%",
                align="start",
            ),
            max_width="1240px",
            margin="0 auto",
            padding="2em 1.6em 4em",
            width="100%",
        ),
        background=rx.color("gray", 2),
        min_height="100vh",
        width="100%",
    )
