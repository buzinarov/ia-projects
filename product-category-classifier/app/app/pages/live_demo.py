"""Live Demo page: pick a sample product or upload your own photo,
choose attributes, compare both models' predictions side by side."""
from pathlib import Path

import reflex as rx
from PIL import Image

from ..backend import SAMPLES_DIR, get_label_maps, load_sample_manifest, run_prediction
from ..layout import page

MAPS = get_label_maps()
ATTRIBUTE_COLUMNS = MAPS["attribute_columns"]
ATTRIBUTE_CLASSES = MAPS["attribute_classes"]
MANIFEST = load_sample_manifest()


def _top_n(probabilities, n=6):
    items = sorted(probabilities.items(), key=lambda kv: -kv[1])[:n]
    return [{"category": k, "probability": round(v, 4)} for k, v in items]


class LiveDemoState(rx.State):
    mode: str = "sample"  # "sample" | "upload"
    selected_filename: str = MANIFEST[0]["filename"] if MANIFEST else ""
    upload_filename: str = ""

    attrs: dict[str, str] = {col: ATTRIBUTE_CLASSES[col][0] for col in ATTRIBUTE_COLUMNS}
    true_category: str = MANIFEST[0]["true_category"] if MANIFEST else ""

    baseline_prediction: str = ""
    baseline_confidence: float = 0.0
    baseline_top: list[dict] = []
    proposed_prediction: str = ""
    proposed_confidence: float = 0.0
    proposed_top: list[dict] = []
    has_result: bool = False

    def load_initial(self):
        if self.selected_filename:
            self.select_sample(self.selected_filename)

    def set_mode(self, mode: str):
        self.mode = mode

    def select_sample(self, filename: str):
        self.mode = "sample"
        self.selected_filename = filename
        item = next((m for m in MANIFEST if m["filename"] == filename), None)
        if item:
            self.true_category = item["true_category"]
            self.attrs = {col: item[f"true_{col}"] for col in ATTRIBUTE_COLUMNS}
        self._predict()

    def set_attr(self, column: str, value: str):
        self.attrs = {**self.attrs, column: value}
        self._predict()

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        file = files[0]
        data = await file.read()
        upload_dir = rx.get_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / file.name
        dest.write_bytes(data)
        self.mode = "upload"
        self.upload_filename = file.name
        self.true_category = ""
        self.attrs = {col: ATTRIBUTE_CLASSES[col][0] for col in ATTRIBUTE_COLUMNS}
        self._predict()

    def _predict(self):
        if self.mode == "sample" and self.selected_filename:
            image_path = SAMPLES_DIR / self.selected_filename
        elif self.mode == "upload" and self.upload_filename:
            image_path = rx.get_upload_dir() / self.upload_filename
        else:
            return
        image = Image.open(Path(image_path))

        baseline_result = run_prediction("baseline", image, self.attrs)
        proposed_result = run_prediction("proposed", image, self.attrs)

        self.baseline_prediction = baseline_result["predicted_class"]
        self.baseline_confidence = baseline_result["confidence"]
        self.baseline_top = _top_n(baseline_result["probabilities"])
        self.proposed_prediction = proposed_result["predicted_class"]
        self.proposed_confidence = proposed_result["confidence"]
        self.proposed_top = _top_n(proposed_result["probabilities"])
        self.has_result = True

    @rx.var
    def current_image_src(self) -> str:
        if self.mode == "upload" and self.upload_filename:
            return rx.get_upload_url(self.upload_filename)
        if self.selected_filename:
            return f"/sample_images/{self.selected_filename}"
        return ""


def _prediction_card(title: str, prediction: str, confidence: float, top: list[dict]) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading(title, size="4"),
            rx.hstack(
                rx.text("Prediction:", weight="medium"),
                rx.badge(prediction, size="2", color_scheme="indigo"),
                rx.text(f"{confidence * 100:.1f}% confidence", color="gray"),
                spacing="2",
                align="center",
            ),
            rx.recharts.bar_chart(
                rx.recharts.bar(data_key="probability", fill="#6366f1"),
                rx.recharts.x_axis(data_key="category", angle=-30, text_anchor="end", height=70, font_size=11),
                rx.recharts.y_axis(domain=[0, 1]),
                data=top,
                height=260,
                width="100%",
            ),
            spacing="3",
            align="stretch",
        ),
        width="100%",
    )


def _attribute_controls() -> rx.Component:
    return rx.vstack(
        *[
            rx.vstack(
                rx.text(col, size="2", weight="medium", color="gray"),
                rx.select(
                    ATTRIBUTE_CLASSES[col],
                    value=LiveDemoState.attrs[col],
                    on_change=lambda value, c=col: LiveDemoState.set_attr(c, value),
                    width="100%",
                ),
                spacing="1",
                align="stretch",
                width="100%",
            )
            for col in ATTRIBUTE_COLUMNS
        ],
        spacing="3",
        width="100%",
    )


def _sample_picker() -> rx.Component:
    return rx.select.root(
        rx.select.trigger(width="100%"),
        rx.select.content(
            *[rx.select.item(m["product_name"], value=m["filename"]) for m in MANIFEST]
        ),
        value=LiveDemoState.selected_filename,
        on_change=LiveDemoState.select_sample,
        width="100%",
    )


def live_demo() -> rx.Component:
    return page(
        rx.heading("Live Demo", size="7"),
        rx.text(
            "Pick a sample product or upload your own photo, set the attributes, "
            "and compare both models' predictions.",
            color="gray",
        ),
        rx.flex(
            rx.card(
                rx.vstack(
                    rx.tabs.root(
                        rx.tabs.list(
                            rx.tabs.trigger("Sample from test set", value="sample"),
                            rx.tabs.trigger("Upload your own", value="upload"),
                        ),
                        rx.tabs.content(_sample_picker(), value="sample", padding_top="0.75em"),
                        rx.tabs.content(
                            rx.upload(
                                rx.text("Drag and drop or click to upload"),
                                id="upload1",
                                accept={"image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"]},
                                max_files=1,
                                on_drop=LiveDemoState.handle_upload(rx.upload_files(upload_id="upload1")),
                                border="1px dashed var(--gray-7)",
                                padding="2em",
                                width="100%",
                            ),
                            value="upload",
                            padding_top="0.75em",
                        ),
                        value=LiveDemoState.mode,
                        on_change=LiveDemoState.set_mode,
                        width="100%",
                    ),
                    rx.cond(
                        LiveDemoState.current_image_src != "",
                        rx.image(src=LiveDemoState.current_image_src, width="220px", border_radius="8px"),
                    ),
                    rx.cond(
                        LiveDemoState.true_category != "",
                        rx.text(f"True category: {LiveDemoState.true_category}", weight="medium"),
                    ),
                    rx.divider(),
                    rx.text("Attributes", size="2", weight="medium", color="gray"),
                    _attribute_controls(),
                    spacing="4",
                    align="stretch",
                ),
                width="320px",
                min_width="320px",
            ),
            rx.cond(
                LiveDemoState.has_result,
                rx.vstack(
                    _prediction_card(
                        "Baseline (image only)",
                        LiveDemoState.baseline_prediction,
                        LiveDemoState.baseline_confidence,
                        LiveDemoState.baseline_top,
                    ),
                    _prediction_card(
                        "Proposed (image + attributes)",
                        LiveDemoState.proposed_prediction,
                        LiveDemoState.proposed_confidence,
                        LiveDemoState.proposed_top,
                    ),
                    spacing="4",
                    width="100%",
                ),
                rx.center(rx.text("Pick a sample to see predictions.", color="gray"), width="100%", height="200px"),
            ),
            spacing="5",
            width="100%",
            align="start",
        ),
    )
