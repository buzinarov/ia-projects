import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Quality Monitoring", page_icon=":bar_chart:", layout="wide")
st.title("Quality Monitoring")
st.caption("The actual baseline-vs-proposed comparison, plus an operational read on what it means for the catalog team.")

metrics = {
    name: json.loads((ARTIFACTS_DIR / f"metrics_{name}.json").read_text())
    for name in ("baseline", "proposed")
}
class_names = metrics["baseline"]["class_names"]

tab_technical, tab_operational = st.tabs(["Technical", "Operational"])

with tab_technical:
    st.subheader("Headline metrics")
    cols = st.columns(2)
    for col, name in zip(cols, ("baseline", "proposed")):
        m = metrics[name]
        with col:
            st.markdown(f"**{name.capitalize()}**")
            st.metric("Accuracy", f"{m['accuracy']:.1%}")
            st.metric("Macro F1", f"{m['macro_f1']:.3f}")
            st.metric("Weighted F1", f"{m['weighted_f1']:.3f}")

    st.subheader("Per-class F1")
    df = pd.DataFrame({
        "class": class_names,
        "baseline": [metrics["baseline"]["per_class"][c]["f1-score"] for c in class_names],
        "proposed": [metrics["proposed"]["per_class"][c]["f1-score"] for c in class_names],
    }).set_index("class")
    st.bar_chart(df, stack=False)
    st.caption(
        "Macro-F1 overall slightly favors the baseline, almost entirely because of 'Free Items' "
        "(16 test examples) -- both models are unreliable there due to data scarcity, not "
        "architecture. The proposed model's real, attributable win is 'Personal Care', the "
        "category the EDA notebook flagged as actually correlated with gender."
    )

    st.subheader("Confusion matrices")
    c1, c2 = st.columns(2)
    c1.image(str(ARTIFACTS_DIR / "confusion_matrix_baseline.png"), caption="Baseline")
    c2.image(str(ARTIFACTS_DIR / "confusion_matrix_proposed.png"), caption="Proposed")

    st.subheader("Training curves")
    for name in ("baseline", "proposed"):
        hist = pd.DataFrame(metrics[name]["training_history"]).set_index("epoch")
        st.markdown(f"**{name.capitalize()}**")
        st.line_chart(hist[["train_loss", "val_loss"]])

with tab_operational:
    st.markdown(
        "If the model is confident enough, the catalog team can trust its tag without a human "
        "review. This view answers: **at what confidence bar, and for what share of the catalog, "
        "can we actually do that?**"
    )
    threshold = st.slider("Confidence threshold", 0.50, 0.99, 0.85, 0.01)

    cols = st.columns(2)
    for col, name in zip(cols, ("baseline", "proposed")):
        confidences = np.array(metrics[name]["test_confidences"])
        rate = (confidences >= threshold).mean()
        with col:
            st.metric(
                f"{name.capitalize()} auto-tag rate",
                f"{rate:.1%}",
                help=f"Share of test items the {name} model would tag without human review at this threshold",
            )

    st.subheader("Auto-tag readiness by category (proposed model)")
    labels_arr = np.array(metrics["proposed"]["test_labels"])
    conf_arr = np.array(metrics["proposed"]["test_confidences"])
    rows = []
    for i, c in enumerate(class_names):
        mask = labels_arr == i
        rate = float((conf_arr[mask] >= threshold).mean()) if mask.sum() else float("nan")
        rows.append({"category": c, "auto-tag rate": rate, "support (test set)": int(mask.sum())})
    st.dataframe(
        pd.DataFrame(rows).sort_values("auto-tag rate", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
