import streamlit as st

st.set_page_config(page_title="Product Category Classifier", page_icon=":shopping_bags:", layout="wide")

st.title("Product Category Classifier")
st.caption("A multi-modal computer vision case study, with a local LLM agent layered on top.")

st.markdown(
    """
### What this is

A product catalog has photos and a few structured attributes (here, the shopper segment a
product is tagged for), but no reliable category label. This project trains a model to predict
the product category from the photo **and** that attribute together, evaluates it honestly
against an image-only baseline, and wraps the result in a small local agent so the whole thing
is something you can actually talk to instead of just a notebook metric.

### How it's put together

```
Product photo ──┐
                 ├──► CNN image branch ──┐
Gender segment ──┘    + MLP branch ──────┼──► Classifier ──► Category
                                          │
Trained checkpoints ─────────────────────┼──► classify_product (tool)
Catalog metadata ──► Chroma index ───────┼──► search_similar_products (tool)
                                          │
                              Local LLM (Ollama, llama3.1:8b) ──► Chat
```

No external API calls anywhere in this app — the agent runs entirely against a local Ollama
model, and the only network dependency is the one-time dataset/model download.

### Pages

- **Live Demo** — pick a sample product or upload your own photo, choose a gender segment, and
  compare the baseline and proposed models' predictions side by side.
- **Quality Monitoring** — the actual baseline-vs-proposed comparison: accuracy, per-class F1,
  confusion matrices, training curves, and an "auto-tag rate" framed as a business metric with a
  live confidence threshold slider.
- **Ask the Catalog** — chat with the local agent. It decides on its own whether to classify an
  attached photo or search the catalog by description.

Full methodology and results write-up: see the project README in the repo root.
"""
)
