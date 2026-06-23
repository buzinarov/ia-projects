import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from components import confidence_bar_chart
from src.inference import load_model, predict

SAMPLES_DIR = PROJECT_ROOT / "app" / "sample_images"

st.set_page_config(page_title="Live Demo", page_icon=":mag:", layout="wide")
st.title("Live Demo")
st.caption("Pick a sample product or upload your own photo, choose a gender segment, and compare both models.")


@st.cache_resource
def get_models():
    return load_model("baseline"), load_model("proposed")


(baseline_model, maps, device), (proposed_model, _, _) = get_models()
gender_classes = maps["gender_classes"]

manifest_path = SAMPLES_DIR / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []

col_input, col_results = st.columns([1, 2])

image, true_category, default_gender = None, None, gender_classes[0]

with col_input:
    source = st.radio("Image source", ["Sample from test set", "Upload your own"])

    if source == "Sample from test set" and manifest:
        options = {f"{m['filename']} (true: {m['true_category']})": m for m in manifest}
        choice = st.selectbox("Pick a sample", list(options.keys()))
        item = options[choice]
        image = Image.open(SAMPLES_DIR / item["filename"])
        true_category = item["true_category"]
        default_gender = item["true_gender"]
    else:
        uploaded = st.file_uploader("Upload a product photo", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = Image.open(uploaded)

    gender = st.selectbox("Gender segment", gender_classes, index=gender_classes.index(default_gender))

    if image:
        st.image(image, caption="Input image", width=200)
        if true_category:
            st.caption(f"True category: **{true_category}**")

with col_results:
    if image:
        baseline_result = predict(baseline_model, maps, device, image, gender)
        proposed_result = predict(proposed_model, maps, device, image, gender)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Baseline (image only)")
            st.metric("Prediction", baseline_result["predicted_class"], f"{baseline_result['confidence']:.0%} confidence")
            st.pyplot(confidence_bar_chart(baseline_result["probabilities"], "Baseline"))
        with c2:
            st.subheader("Proposed (image + gender)")
            st.metric("Prediction", proposed_result["predicted_class"], f"{proposed_result['confidence']:.0%} confidence")
            st.pyplot(confidence_bar_chart(proposed_result["probabilities"], "Proposed"))
    else:
        st.info("Pick a sample or upload a photo to see predictions.")
