"""Shared page chrome: top nav + a consistent page wrapper."""
import reflex as rx

NAV_LINKS = [
    ("Overview", "/"),
    ("Live Demo", "/live-demo"),
    ("Quality Monitoring", "/quality-monitoring"),
    ("Ask the Catalog", "/ask-catalog"),
]


def navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon("shopping-bag", size=20),
                rx.heading("Product Category Classifier", size="5"),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.hstack(
                *[
                    rx.link(label, href=href, size="3", weight="medium", color_scheme="gray", high_contrast=True)
                    for label, href in NAV_LINKS
                ],
                rx.color_mode.button(),
                spacing="5",
                align="center",
            ),
            width="100%",
            align="center",
            padding="1em 2em",
        ),
        border_bottom="1px solid var(--gray-5)",
        position="sticky",
        top="0",
        background="var(--color-background)",
        z_index="100",
        width="100%",
    )


def page(*children, max_width="1100px") -> rx.Component:
    return rx.fragment(
        navbar(),
        rx.container(
            rx.vstack(*children, spacing="6", padding="2.5em 1.5em", align="stretch"),
            max_width=max_width,
        ),
    )
