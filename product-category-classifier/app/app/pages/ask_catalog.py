"""Ask the Catalog: chat with the local tool-calling agent. It decides
on its own whether to search the catalog or classify an attached photo.
No external API calls -- everything runs against a local Ollama model.
"""
import asyncio
import json
import sys

import reflex as rx

from ..backend import PROJECT_ROOT, get_label_maps

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import chat as agent_chat  # noqa: E402
from ..layout import page

MAPS = get_label_maps()
ATTRIBUTE_COLUMNS = MAPS["attribute_columns"]
ATTRIBUTE_CLASSES = MAPS["attribute_classes"]

NO_TOOL_TRACE = "[]"


class CatalogChatState(rx.State):
    # Every message is dict[str, str] (tool_trace pre-serialized to JSON)
    # so rx.foreach has a uniformly-typed structure to render.
    messages: list[dict[str, str]] = []
    input_text: str = ""
    uploaded_filename: str = ""
    attrs: dict[str, str] = {col: ATTRIBUTE_CLASSES[col][0] for col in ATTRIBUTE_COLUMNS}
    is_thinking: bool = False
    agent_history: list[dict] = []

    def set_attr(self, column: str, value: str):
        self.attrs = {**self.attrs, column: value}

    def set_input_text(self, value: str):
        self.input_text = value

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        file = files[0]
        data = await file.read()
        upload_dir = rx.get_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / file.name).write_bytes(data)
        self.uploaded_filename = file.name

    def clear_conversation(self):
        self.messages = []
        self.agent_history = []

    @rx.var
    def uploaded_image_src(self) -> str:
        if self.uploaded_filename:
            return rx.get_upload_url(self.uploaded_filename)
        return ""

    async def send(self):
        text = self.input_text.strip()
        if not text:
            return
        self.messages = self.messages + [{"role": "user", "content": text, "tool_trace_json": NO_TOOL_TRACE}]
        self.input_text = ""
        self.is_thinking = True
        yield

        image_path = None
        if self.uploaded_filename:
            image_path = str(rx.get_upload_dir() / self.uploaded_filename)

        result = await asyncio.to_thread(
            agent_chat, text, history=self.agent_history, image_path=image_path, attrs=dict(self.attrs)
        )

        self.agent_history = result["messages"]
        self.messages = self.messages + [
            {
                "role": "assistant",
                "content": result["reply"],
                "tool_trace_json": json.dumps(result["tool_trace"]),
            }
        ]
        self.is_thinking = False


def _attribute_controls() -> rx.Component:
    return rx.vstack(
        *[
            rx.vstack(
                rx.text(col, size="1", color="gray"),
                rx.select(
                    ATTRIBUTE_CLASSES[col],
                    value=CatalogChatState.attrs[col],
                    on_change=lambda value, c=col: CatalogChatState.set_attr(c, value),
                    size="2",
                    width="100%",
                ),
                spacing="1",
                width="100%",
            )
            for col in ATTRIBUTE_COLUMNS
        ],
        spacing="2",
        width="100%",
    )


def _message_bubble(message: dict[str, str]) -> rx.Component:
    is_user = message["role"] == "user"
    return rx.box(
        rx.vstack(
            rx.markdown(message["content"]),
            rx.cond(
                message["tool_trace_json"] != NO_TOOL_TRACE,
                rx.accordion.root(
                    rx.accordion.item(
                        header="What the agent did",
                        content=rx.code_block(
                            message["tool_trace_json"],
                            language="json",
                            can_copy=True,
                        ),
                    ),
                    collapsible=True,
                    width="100%",
                ),
            ),
            align="stretch",
            spacing="2",
        ),
        background=rx.cond(is_user, "var(--indigo-3)", "var(--gray-3)"),
        padding="0.75em 1em",
        border_radius="10px",
        max_width="80%",
        align_self=rx.cond(is_user, "flex-end", "flex-start"),
    )


def ask_catalog() -> rx.Component:
    return page(
        rx.heading("Ask the Catalog", size="7"),
        rx.text(
            "A local agent (Ollama, llama3.1:8b) decides on its own whether to search the "
            "catalog or classify an attached photo. No external API calls.",
            color="gray",
        ),
        rx.flex(
            rx.card(
                rx.vstack(
                    rx.text("Attach a product photo (optional)", size="2", weight="medium"),
                    rx.upload(
                        rx.text("Drag and drop or click to upload"),
                        id="chat_upload",
                        accept={"image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"]},
                        max_files=1,
                        on_drop=CatalogChatState.handle_upload(rx.upload_files(upload_id="chat_upload")),
                        border="1px dashed var(--gray-7)",
                        padding="1.5em",
                        width="100%",
                    ),
                    rx.cond(
                        CatalogChatState.uploaded_image_src != "",
                        rx.image(src=CatalogChatState.uploaded_image_src, width="140px", border_radius="8px"),
                    ),
                    rx.divider(),
                    rx.text("Attributes", size="2", weight="medium"),
                    _attribute_controls(),
                    rx.button("Clear conversation", on_click=CatalogChatState.clear_conversation, variant="soft", width="100%"),
                    spacing="3",
                    align="stretch",
                ),
                width="280px",
                min_width="280px",
            ),
            rx.vstack(
                rx.box(
                    rx.vstack(
                        rx.foreach(CatalogChatState.messages, _message_bubble),
                        rx.cond(
                            CatalogChatState.is_thinking,
                            rx.hstack(rx.spinner(size="2"), rx.text("Thinking...", color="gray", size="2")),
                        ),
                        spacing="3",
                        align="stretch",
                        width="100%",
                    ),
                    width="100%",
                    min_height="350px",
                    max_height="55vh",
                    overflow_y="auto",
                    padding="0.5em",
                ),
                rx.form(
                    rx.hstack(
                        rx.input(
                            value=CatalogChatState.input_text,
                            on_change=CatalogChatState.set_input_text,
                            placeholder="Ask about the catalog, or about an attached photo...",
                            width="100%",
                        ),
                        rx.button("Send", type="submit", loading=CatalogChatState.is_thinking),
                        width="100%",
                        spacing="2",
                    ),
                    on_submit=CatalogChatState.send,
                    width="100%",
                ),
                spacing="3",
                width="100%",
                align="stretch",
            ),
            spacing="5",
            width="100%",
            align="start",
        ),
    )
