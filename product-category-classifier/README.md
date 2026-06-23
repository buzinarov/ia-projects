# Product Category Classifier

A multi-modal computer vision case study — a CNN that classifies product photos using the image *and* a structured attribute together, evaluated honestly against an image-only baseline, with a small local LLM agent layered on top so the result is something you can actually talk to.

## Objective

Online catalogs have product photos and a few structured attributes, but category tagging is often incomplete or inconsistent — someone has to do it by hand, and it doesn't scale. The question this project answers: **does a model that sees the photo and one structured attribute together actually beat a model that only sees the photo?**

That's not a rhetorical question I'm answering "yes" to by default. Multi-modal architectures are easy to build and easy to oversell — the point of this case study is to measure the lift honestly, including the parts where it doesn't show up, and to land on something that demonstrates real applied skill rather than a clean story that happens not to be true.

The underlying multi-input pattern — an image branch and a small categorical branch, concatenated into a shared classifier — is one I first built for a DataCamp AI Engineer certification exercise (an OCR model reading insurance documents). The architecture transfers almost exactly here; the dataset, the problem, the evaluation, and the agent layer are original.

## The Data

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), via the public Hugging Face mirror [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small) — no auth required, loads directly through the `datasets` library.

- 44,072 product photos with structured metadata (gender, category, color, season, usage, display name).
- **Image:** the product photo, resized to 80x80 RGB. (The original cert exercise used 64x64 grayscale — inherited from a synthetic dataset. These are real product photos at roughly a 3:4 aspect ratio, so 80x80 keeps more real detail without distorting the crop.)
- **Second modality:** `gender` (Men / Women / Boys / Girls / Unisex) — the structured attribute every product is already tagged with.
- **Target:** `masterCategory`. The raw data has 7 categories, but two are barely represented at all — "Home" has a single example in the whole dataset, "Sporting Goods" has 25. Neither survives a stratified train/val/test split, so `src/data.py` drops anything under 100 samples. **5 classes remain: Accessories, Apparel, Footwear, Free Items, Personal Care.**

## Methodology

**Two models, one true difference between them**, so the comparison actually means something:

- **Baseline** — image-only CNN.
- **Proposed** — the same image branch, plus a small `gender_layer` MLP branch, concatenated before the classifier head.

Both trained identically: same 80x80 RGB preprocessing, same 70/15/15 stratified split, same 10 epochs, same class-weighted loss (the class imbalance is real — Apparel alone is ~48% of the data). The only variable that changes is whether the model sees `gender`.

**Evaluation:** accuracy, macro-F1, weighted-F1, full per-class precision/recall/F1, and confusion matrices on a held-out test set — plus one metric that isn't standard but matters more for this problem than accuracy does: **auto-tag rate**, the share of items the model predicts at ≥85% confidence (configurable). Accuracy tells you how good the model is on average; auto-tag rate tells a catalog team how much manual review they actually get to skip.

## Architecture

**Model architecture** (see `src/models.py`):

```
BaselineImageModel:                    MultiModalProductClassifier:
  image (3x80x80)                        image (3x80x80)        gender (one-hot, 5)
       |                                       |                        |
  Conv2d -> Pool -> ReLU                  Conv2d -> Pool -> ReLU        |
  Conv2d -> Pool -> ReLU                  Conv2d -> Pool -> ReLU        |
       |                                       |                  Linear -> ReLU
  Flatten -> Linear(128)                  Flatten -> Linear(128)        |
       |                                       \________________ concat
  Linear -> num_classes                                  |
                                                  Linear -> ReLU -> Linear -> num_classes
```

**Product architecture** — how the pieces fit together end to end:

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

On top of the classifier, a small agent (served locally by Ollama — no external API, no API key, nothing that can leak from a public repo) has two tools: `classify_product` calls the trained model above, and `search_similar_products` runs a RAG lookup over the catalog's text metadata in a local Chroma index. The agent decides on its own which tool a question needs — see `src/agent.py`.

**Repository architecture:**

```
src/
  data.py          HF dataset load, resize/transform, stratified split, caching
  models.py        BaselineImageModel, MultiModalProductClassifier
  train.py         python -m src.train --model baseline|proposed --epochs 10
  evaluate.py      python -m src.evaluate --model baseline|proposed
  inference.py     shared predict() used by the notebook, the app, and the agent
  rag.py           Chroma index over product metadata
  agent.py         the tool-calling agent
notebooks/
  01_eda.ipynb         class balance, gender x category relationship, image size check
  02_case_study.ipynb  the narrative version of this README, executed end to end
artifacts/         metrics, confusion matrices, training curves (checkpoints gitignored)
app/               Streamlit app — Live Demo, Quality Monitoring, Ask the Catalog
```

**Running it:**

```bash
pip install -r requirements.txt

# train + evaluate both models (downloads the dataset on first run)
python -m src.train --model baseline --epochs 10
python -m src.train --model proposed --epochs 10
python -m src.evaluate --model baseline
python -m src.evaluate --model proposed

# optional: the local agent needs Ollama running with llama3.1:8b pulled
ollama pull llama3.1:8b

streamlit run app/streamlit_app.py
```

The Live Demo page needs the checkpoints from `train.py` to exist locally first — there's no hosted model behind this app, by design (see Limitations).

## Results

| | Baseline (image only) | Proposed (image + gender) |
|---|---|---|
| Accuracy | 95.9% | 95.7% |
| Macro F1 | 0.802 | 0.779 |
| Weighted F1 | 0.960 | **0.963** |
| Auto-tag rate (≥85% confidence) | 92.6% | 91.1% |

Read at face value, that's a wash — and a less honest write-up would stop there, or quietly pick whichever metric looked better. The per-class breakdown is where the real story is:

| Category | Baseline F1 | Proposed F1 | Test support |
|---|---|---|---|
| Apparel | 0.978 | 0.981 | 3,204 |
| Footwear | 0.981 | 0.987 | 1,380 |
| Accessories | 0.933 | 0.922 | 1,686 |
| **Personal Care** | **0.862** | **0.925** | 321 |
| Free Items | 0.258 | 0.081 | 16 |

Two things explain the headline numbers:

1. **Macro-F1 favors the baseline almost entirely because of "Free Items"** — 16 examples in the test set, ~105 in the whole dataset. Both models are unreliable there; that's a data scarcity problem, not an architecture one. Macro-F1 weights every class equally regardless of support, so this one tiny class swings the average more than it should.
2. **The proposed model's real, attributable win is Personal Care** (F1 0.925 vs. 0.862, driven by a precision jump from 0.76 to 0.93 — far fewer false positives). This is exactly the category the EDA notebook's gender × category crosstab flagged as having a genuine relationship with gender, before any model was trained. The multi-modal branch earns its keep specifically where the data says it should — not as a blanket improvement everywhere, which is a more credible result than a uniformly clean win would be.

Full confusion matrices, training curves, and the live confidence-threshold breakdown are in `notebooks/02_case_study.ipynb` and the app's Quality Monitoring page.

## Limitations & Next Steps

- **`subCategory` as a harder follow-on target.** `masterCategory` was the right scope for a clean v0 — the gender signal almost certainly matters more at finer granularity (specific article types skew by gender far more than broad categories do).
- **Two-head multi-task output** (predict `masterCategory` and `subCategory` together) is a natural v1.5, deliberately left out here to keep the baseline-vs-proposed comparison clean.
- **"Free Items" stays unreliable** for any model trained on this dataset as-is — it needs targeted data collection, not a better architecture.
- **Checkpoint hosting.** The Live Demo currently requires running `train.py` locally first. Hosting checkpoints as a GitHub Release asset or on the HF Hub would make the demo work on a fresh clone with zero setup.
- **Confidence calibration.** The 85% auto-tag threshold is a reasonable starting point, not a calibrated one — temperature scaling would make the operational numbers more trustworthy before anyone actually ships a threshold like this.
