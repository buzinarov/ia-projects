# Product Category Classifier

A multi-modal computer vision case study, built around a real enterprise scenario and evaluated with the kind of rigor you'd want before shipping — multiple training seeds, a held-out test set, automated tests, and a data contract. A local LLM agent sits on top so the trained model is something you can actually talk to, not just a number in a notebook.

**The short version:** I tested whether adding a product's structured attributes (gender, color, season, usage) to its photo improves classification over a photo-only model. It doesn't — the image-only baseline wins on every metric, and I show *why* rather than quietly dropping the result. The honest negative is the interesting part.

## Objective

**The scenario.** An e-commerce company gets new products every month and needs each one classified against a fixed schema before it can feed downstream ML models and executive/operational reports. Hand-labeling doesn't scale, so the goal is an automated classifier with a real reliability bar: a fixed output contract, automated tests, and a quality view that an engineer *and* an operations manager can both read.

**The technical question.** Multi-modal models — combining an image with structured fields — are a standard pitch. But "more inputs must help" is an assumption worth testing, not asserting. So: does feeding the model the product's attributes *alongside* the photo beat a model that only sees the photo?

**The answer, here, is no** — and that's the finding, not a footnote. The image-only baseline beats the multi-modal model across 3 seeds, and it keeps winning after I check whether more training time or a smaller attribute set closes the gap. The [production recommendation](#results) reflects that.

The underlying multi-input pattern (an image branch and a categorical branch concatenated into a shared classifier) is one I first built for a DataCamp AI Engineer certification exercise. The pattern transfers; the dataset, the problem framing, the experimental rigor, and the agent/contract layer are my own.

## The Data

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), via the public Hugging Face mirror [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small) — public, no auth, loads straight through the `datasets` library.

- **44,072 product photos** with structured metadata.
- **Image:** resized to 80×80 RGB (the EDA notebook walks through why this size).
- **Structured attributes (the second modality):** `gender`, `baseColour`, `season`, `usage`, concatenated into a 62-dim one-hot vector — the fields a real catalog would already know about a product before anyone looks at the photo.
- **Target:** `subCategory` (Topwear, Shoes, Bags, Watches, …) — **27 classes** after dropping the long tail under 100 samples (18 classes, 0.8% of rows). This is a finer grain than the obvious `masterCategory` (5 broad classes); at the broad grain the photo alone nearly saturates the signal, which makes it a weak test of whether attributes add anything. `subCategory` is also closer to what a real catalog contract requires.

`01_eda.ipynb` quantifies the target imbalance and measures each attribute's association with the target (Cramér's V) — the up-front check that justified testing a multi-modal model at all.

## Methodology

**Two models, one honest difference.**

- **Baseline** — image-only CNN (3 conv blocks, BatchNorm, dropout).
- **Proposed** — the *same* image trunk plus a small attribute MLP branch, concatenated before the classifier head.

Both train identically: same preprocessing, same 70/15/15 stratified split, same class-weighted loss, same optimizer. The only thing that changes is whether the model sees the attribute branch — enforced in code by a shared trunk, not by convention.

**Rigor: 3 seeds, not 1.** A single run isn't evidence for "model X beats model Y." Every number here is mean ± std across seeds `[0, 1, 2]`, with the train/val/test split held fixed (`SPLIT_SEED=42`) so seed variation isolates the architecture, not a lucky partition.

**Two follow-up experiments before drawing a conclusion:** an attribute ablation (does a smaller, less noisy attribute set help?) and a longer 25-epoch run (did the deeper model just need more time?). Neither moved the result — which is the point of running them.

**Metrics:** accuracy, macro-F1 (the headline, given the imbalance), weighted-F1, per-class F1, confusion matrices — plus an **auto-tag rate**: the share of items predicted at ≥85% confidence, the operational number a catalog team uses to decide how much manual review they can skip.

## Architecture

**Model** (`src/models.py` — both models share one image trunk, so the comparison stays fair by construction):

```
BaselineImageModel:                       MultiModalProductClassifier:
  image (3x80x80)                           image (3x80x80)         attrs (one-hot, 62)
       |                                          |                          |
  [Conv-BN-ReLU-Pool] x3                     [Conv-BN-ReLU-Pool] x3          |
       |                                          |                    Linear -> ReLU -> Dropout
  Flatten -> Linear(128) -> Dropout           Flatten -> Linear(128)         |
       |                                          \_________________ concat
  Linear -> num_classes                                    |
                                                   Linear -> ReLU -> Dropout -> Linear -> num_classes
```

**Product** — how the pieces fit end to end:

```
Product photo ──┐
                 ├──► CNN image branch ──┐
Attributes ──────┘    + MLP branch ──────┼──► Classifier ──► Subcategory
                                          │
Trained checkpoints ─────────────────────┼──► classify_product (tool, contract-wrapped)
Catalog metadata ──► Chroma index ───────┼──► search_similar_products (tool)
                                          │
                              Local LLM (Ollama, llama3.1:8b) ──► Chat
```

A **data contract** (`src/contract.py`) pins the output schema downstream consumers depend on — `product_id`, `predicted_subcategory`, `confidence`, `model_name`, `model_version`, `predicted_at` — with `validate_prediction_record()` raising on a malformed record instead of letting it through. A small local agent (Ollama — no external API, no keys, nothing that can leak from a public repo) has two tools: `classify_product` (the trained model, contract-wrapped) and `search_similar_products` (RAG over catalog metadata via a local Chroma index). The agent picks the tool a question needs.

**Repository:**

```
src/
  data.py          dataset load, resize/transform, stratified split, caching (generic target/attribute columns)
  models.py        BaselineImageModel, MultiModalProductClassifier
  train.py         python -m src.train --model baseline|proposed --seed 0 [--attributes col ...]
  evaluate.py      python -m src.evaluate --model baseline|proposed --seed 0
  aggregate.py     mean +/- std across seeds, summed confusion matrices
  run_all.py       python -m src.run_all --seeds 0 1 2 --epochs 25   (the one-command full experiment)
  run_ablation.py  the attribute-subset sweep
  contract.py      output data contract + validator
  inference.py     shared predict() / predict_with_contract()
  rag.py           Chroma index over product metadata
  agent.py         the tool-calling agent
notebooks/
  01_eda.ipynb         class balance, attribute-vs-target association, image-size rationale
  02_case_study.ipynb  the full narrative, executed end to end
tests/             pytest: data-contract schema/bounds, model sanity, metric-regression floors
app/               Reflex app — Live Demo, Quality Monitoring, Ask the Catalog
artifacts/         aggregated metrics, confusion matrices, training history (checkpoints gitignored)
```

**Running it:**

```bash
pip install -r requirements.txt

# trains + evaluates both models across 3 seeds, aggregates, exports the app's demo assets
python -m src.run_all --seeds 0 1 2 --epochs 25

python -m src.run_ablation --seeds 0 1 2 --epochs 10   # optional, already run (see Results)
pytest                                                  # contract + model sanity always run; regression gate needs the run above

ollama pull llama3.1:8b      # the agent needs a local Ollama model
cd app && reflex run         # Live Demo needs the checkpoints from run_all first
```

## Results

| | Baseline (image only) | Proposed (image + attributes) |
|---|---|---|
| Accuracy | **93.1% ± 0.03%** | 88.3% ± 0.9% |
| Macro F1 | **0.854 ± 0.002** | 0.803 ± 0.007 |
| Weighted F1 | **0.935 ± 0.0001** | 0.897 ± 0.005 |
| Auto-tag rate (≥85% confidence) | **87.2% ± 0.8%** | 63.4% ± 6.0% |

The baseline wins on every metric *and* is far more stable — it converges to essentially the same solution every run (std 0.0001–0.002), while the proposed model is both worse and noisier (up to 6 points of std on the auto-tag rate). **For this dataset, architecture, and target grain, the production recommendation is the image-only baseline.**

### Why the baseline wins

This survived two checks before I accepted it:

**Attribute ablation.** Maybe a smaller, less noisy attribute set helps — `baseColour` (46 values) is a plausible source of dilution. It doesn't:

| Variant | Accuracy | Macro F1 |
|---|---|---|
| Baseline | 0.863 ± 0.017 | 0.786 ± 0.014 |
| Proposed, all 4 attributes | 0.839 ± 0.022 | 0.743 ± 0.036 |
| Proposed, gender only | 0.816 ± 0.005 | 0.735 ± 0.020 |
| Proposed, gender + season + usage | 0.800 ± 0.081 | 0.738 ± 0.069 |

No subset beats the baseline, and dropping `baseColour` actually made training *less* stable (one seed collapsed to 68.6% while two others hit ~85.8%).

**More training time.** Re-ran both at 25 epochs instead of 10. Both improved — but the baseline improved more and its variance shrank to near zero. More epochs widened the gap; they weren't the bottleneck.

**Per-class read.** The attribute branch helps a few low-support classes by a hair (+0.004 to +0.028 F1) but hurts more and bigger ones (Scarves −0.276, Headwear −0.151, Sandal −0.149). At this grain the photo already carries most of the signal, and a plain concatenation gives the model no way to *learn when to ignore* a weak attribute. That's a legitimate, useful result: it tells a team not to pay the extra training/inference complexity for the attribute branch on this target.

## Limitations & Next Steps

- **Fusion mechanism.** A gating/attention layer (instead of plain concatenation) could let the model downweight irrelevant attributes per example — the most promising thing to try next.
- **Finer target.** `articleType` (141 classes) is a grain where attributes might correlate more tightly with the label; untested here.
- **Multi-task head** (predict `masterCategory` and `subCategory` jointly) remains untested.
- **Checkpoint hosting.** The Live Demo needs `run_all` to have run locally; hosting checkpoints (GitHub Release / HF Hub) would make it zero-setup on a fresh clone.
- **Confidence calibration.** The 85% auto-tag threshold is a sensible default, not a calibrated one — temperature scaling would make the operational number trustworthy before shipping it.
- **Out-of-distribution photos.** The test set comes from the same catalog style as training, so it can't catch a domain shift. The app's "Upload your own" exposed this directly: a stock photo of a white sneaker, shot differently than the catalog, was classified as "Bags" at 94.7% confidence — confidently wrong. A real deployment gate would need a separate OOD evaluation set.
