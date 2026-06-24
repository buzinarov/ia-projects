# Product Category Classifier

A multi-modal computer vision case study — built around a real enterprise scenario, evaluated with statistical rigor instead of a single lucky training run, with a local LLM agent and a data contract layered on top.

## Objective

**The scenario:** an e-commerce company receives new products every month and needs each one classified against an established data contract before it can flow into downstream ML models and executive/operational reports. Classifying products by hand doesn't scale — the company wants an automated classifier with a high bar for reliability: a fixed output schema, automated tests, and a quality view that both an engineer and an operations manager can read.

**The technical question this case study actually answers:** does feeding the model a product's structured attributes (gender, color, season, usage) *in addition to* its photo improve classification over a photo-only model? That's the standard pitch for multi-modal architectures, and it's worth testing honestly rather than assuming it's true.

It isn't, here. **The image-only baseline beats the multi-modal model on every configuration tested**, across 3 training seeds, after also testing whether more training time or a smaller attribute set would close the gap. That's the actual finding, and the production recommendation below reflects it — see [Results](#results) for the numbers and [why](#why-the-baseline-wins) the multi-modal hypothesis didn't hold up here.

The underlying multi-input pattern — an image branch and a small categorical branch, concatenated into a shared classifier — is one I first built for a DataCamp AI Engineer certification exercise (an OCR model reading insurance documents). The architecture transfers almost exactly here; the dataset, the problem, the rigor, and the agent/data-contract layer are original.

## The Data

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), via the public Hugging Face mirror [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small) — no auth required, loads directly through the `datasets` library.

- 44,072 product photos with structured metadata.
- **Image:** the product photo, resized to 80x80 RGB.
- **Structured attributes (the second modality):** `gender`, `baseColour`, `season`, `usage` — concatenated into a 62-dimensional one-hot vector. This is the closest analog to a real product data contract: fields that are already known about a product before anyone looks at the photo.
- **Target:** `subCategory` (e.g. Topwear, Bottomwear, Shoes, Bags, Watches) — **27 classes** after dropping anything under 100 samples (18 classes / 368 rows dropped; 99.2% of the data retained). This is a deliberately finer grain than a first pass at this dataset would use (`masterCategory`, 5 broad classes) — at the broad grain, the photo alone nearly saturates the signal, which makes it a weak test of whether structured attributes add anything. `subCategory` is also what a real catalog data contract would actually require.

## Methodology

**Two models, one true difference between them:**

- **Baseline** — image-only CNN (3 conv blocks, BatchNorm, Dropout).
- **Proposed** — the identical image branch, plus a small attribute MLP branch, concatenated before the classifier head.

Both trained identically: same 80x80 RGB preprocessing, same 70/15/15 stratified split, same class-weighted loss, same optimizer. The only variable that changes is whether the model sees the attribute branch.

**Statistical rigor — 3 training seeds, not one.** A single training run isn't sufficient evidence for a claim like "model X beats model Y" in a setting where the result feeds downstream decisions. Every number in this README is a mean ± standard deviation across seeds `[0, 1, 2]`, with the train/val/test split itself held fixed across seeds (`SPLIT_SEED=42`) so seed-to-seed variation isolates "is the architecture better," not "did we get a lucky data partition."

**Before concluding anything, two legitimate follow-up experiments were run** (see [Why the baseline wins](#why-the-baseline-wins)):
1. An attribute ablation — does a smaller, less noisy attribute subset help the proposed model?
2. A longer training run (25 vs. 10 epochs) — did the proposed model just need more time to converge?

Neither changed the conclusion. That's the point of running them: a result that survives both isn't a fluke.

**Evaluation:** accuracy, macro-F1, weighted-F1, full per-class precision/recall/F1, and confusion matrices on a held-out test set — plus an **auto-tag rate**: the share of items the model predicts at ≥85% confidence (configurable in the app), which is the operational number that actually matters to a catalog team deciding how much manual review they can skip.

## Architecture

**Model architecture** (see `src/models.py` — both models share the exact same image trunk, built by one function, so "only the attribute branch differs" is enforced by code, not convention):

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

**Product architecture** — how the pieces fit together end to end:

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

On top of the classifier: a **data contract** (`src/contract.py`) defines the exact output schema a downstream consumer can rely on — `product_id`, `predicted_subcategory`, `confidence`, `model_name`, `model_version`, `predicted_at` — with `validate_prediction_record()` raising loudly on a malformed record rather than passing it through. A small local agent (Ollama — no external API, no API key, nothing that can leak from a public repo) has two tools: `classify_product` (calls the trained model, contract-wrapped) and `search_similar_products` (RAG over the catalog's text metadata via a local Chroma index). The agent decides on its own which tool a question needs.

**Repository architecture:**

```
src/
  data.py          HF dataset load, resize/transform, stratified split, caching (generic target/attribute columns)
  models.py        BaselineImageModel, MultiModalProductClassifier
  train.py         python -m src.train --model baseline|proposed --seed 0 [--attributes col ...]
  evaluate.py      python -m src.evaluate --model baseline|proposed --seed 0
  aggregate.py     mean +/- std across seeds, summed confusion matrices
  run_all.py       python -m src.run_all --seeds 0 1 2 --epochs 25  (the one-command full experiment)
  run_ablation.py  attribute-subset sweep
  contract.py      the output data contract + validator
  inference.py     shared predict() / predict_with_contract(), used by the app and the agent
  rag.py           Chroma index over product metadata
  agent.py         the tool-calling agent
notebooks/
  01_eda.ipynb                  v1 EDA (masterCategory) -- see 03_ for the current case study
  02_case_study.ipynb           v1 narrative (masterCategory) -- kept as an honest historical record
  03_subcategory_case_study.ipynb   the current case study, executed end to end
tests/             pytest: data contract schema/bounds, model sanity, regression floors
artifacts/         metrics, confusion matrices, training curves (checkpoints gitignored)
app/               Reflex app -- Live Demo, Quality Monitoring, Ask the Catalog
```

**Running it:**

```bash
pip install -r requirements.txt

# one command: trains + evaluates both models across 3 seeds, aggregates, exports demo assets
python -m src.run_all --seeds 0 1 2 --epochs 25

# attribute ablation (optional -- already run, see Results)
python -m src.run_ablation --seeds 0 1 2 --epochs 10

pytest                       # data contract + model sanity always run; regression gate needs the run above

ollama pull llama3.1:8b      # the local agent needs Ollama running
cd app && reflex run         # Live Demo needs the checkpoints from run_all.py to exist locally first
```

## Results

| | Baseline (image only) | Proposed (image + attributes) |
|---|---|---|
| Accuracy | **93.1% ± 0.03%** | 88.3% ± 0.9% |
| Macro F1 | **0.854 ± 0.002** | 0.803 ± 0.007 |
| Weighted F1 | **0.935 ± 0.0001** | 0.897 ± 0.005 |
| Auto-tag rate (≥85% confidence) | **87.2% ± 0.8%** | 63.4% ± 6.0% |

The baseline wins on every metric, with much lower run-to-run variance — it converges to essentially the same solution every time (std of 0.0001–0.002), while the proposed model is both worse on average and less stable (std up to 6 percentage points on the auto-tag rate). **For this dataset, architecture, and problem grain, the production recommendation is the image-only baseline.**

### Why the baseline wins

This isn't a one-shot result — it survived two follow-up checks before being accepted:

**1. Attribute ablation.** Maybe a smaller, less noisy attribute set would help. Tested gender-only and gender+season+usage (dropping the high-cardinality `baseColour`) against the full 4-attribute set, 3 seeds each, 10 epochs:

| Variant | Accuracy | Macro F1 |
|---|---|---|
| Baseline | 0.863 ± 0.017 | 0.786 ± 0.014 |
| Proposed, all 4 attributes | 0.839 ± 0.022 | 0.743 ± 0.036 |
| Proposed, gender only | 0.816 ± 0.005 | 0.735 ± 0.020 |
| Proposed, gender + season + usage | 0.800 ± 0.081 | 0.738 ± 0.069 |

No subset beat the baseline. The gender+season+usage variant was also the least *stable* — one seed collapsed to 68.6% accuracy while the other two reached ~85.8%, a std an order of magnitude higher than the baseline ever shows. Removing `baseColour` didn't fix anything; it just added training instability.

**2. More training time.** Maybe the proposed model — deeper, with an extra branch — just needed longer to converge. Re-ran baseline and the full-attribute proposed model at 25 epochs instead of 10 (numbers in the Results table above). Both models improved with more training, but the baseline improved *more*: the gap widened, and the baseline's variance shrank to near zero while the proposed model's did not. More training time was not the bottleneck.

**3. Per-class breakdown.** The proposed model isn't uniformly worse — a handful of low-support classes (Apparel Set, Makeup, Cufflinks, Ties, Saree) get marginally better F1 (+0.004 to +0.028). But the losses are larger and more numerous: Scarves (-0.276 F1), Headwear (-0.151), Accessories (-0.149), Sandal (-0.149), Free Gifts (-0.135). The attribute branch doesn't add a targeted, explainable lift the way it did at a coarser target grain in an earlier pass at this dataset — here it net-adds optimization noise more often than it adds signal.

**Read on this:** at `subCategory` grain, the photo itself already carries most of the discriminative signal — a clear image of a "Sandal" mostly doesn't need to know the shopper's gender to be classified correctly, and a simple concatenation fusion gives the model no way to learn *when* to ignore a weak or irrelevant attribute. That's a legitimate, useful negative result, not a failed experiment: it tells a real engineering team not to spend the extra training/inference complexity on the attribute branch for this target.

## Limitations & Next Steps

- **A gating/attention fusion mechanism** (instead of plain concatenation) might let the model learn to downweight irrelevant attributes per-example rather than always incorporating all 62 dimensions — a more sophisticated next experiment than anything tried here.
- **`articleType`** (141 raw classes) is an even finer grain where attributes might matter more (a specific article type correlates much more tightly with gender than a broad subcategory does) — untested here; flagged as the natural next step rather than assumed.
- **Two-head multi-task output** (predict `masterCategory` and `subCategory` together) remains untested.
- **Checkpoint hosting.** The Live Demo currently requires running `run_all.py` locally first. Hosting checkpoints as a GitHub Release asset or on the HF Hub would make the demo work on a fresh clone with zero setup.
- **Confidence calibration.** The 85% auto-tag threshold is a reasonable starting point, not a calibrated one — temperature scaling would make the operational numbers more trustworthy before anyone ships a threshold like this.
- **Regression floors** (`tests/test_regression.py`) are calibrated from one experimental run (mean − 1.5×std per model); a production gate would want more historical runs behind it before being trusted as a real CI check.
- **The test-set accuracy doesn't transfer to arbitrary real-world photos.** Live Demo's "Upload your own" surfaced this directly: a stock photo of an all-white sneaker, shot at a different angle and rendering style than the catalog's training photos, got classified as "Bags" at 94.7% confidence by the baseline — confidently wrong, not just uncertain. The held-out test set is drawn from the same source/style as training, so it can't catch this; a deployment gate would need a separate out-of-distribution evaluation set built from photos that don't come from the training catalog.
