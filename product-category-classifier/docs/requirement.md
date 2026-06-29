# Data Product Requirement — Multi-Modal Product Recommender

> **Format.** This document follows a data-product-oriented requirement
> structure: **Objective** (the scenario, the initiative, and what counts as
> success), **Methodology** (the technical initiative and the agreements
> reached between the personas), and **Results** (the comparison and analysis
> of the outcomes). It is the kickoff contract for the initiative; the Results
> section states the acceptance criteria up front and is completed with
> measured numbers once the model is built and evaluated.

## Personas

Three roles collaborate on this data product. The requirement reflects the
agreements reached between them, not a single function's wish list.

| Persona | Owns | Cares most about |
|---|---|---|
| **Product Manager** | Scope, success criteria, sequencing across the four lifecycle pillars | A measurable, shippable improvement and a clear definition of "done" |
| **Commercial Stakeholder** | The business case for the *suggested product* feature | Revenue and engagement on the recommendation surface |
| **Senior AI Engineer** | Model design, evaluation rigor, the data contract | Defensible metrics, honest baselines, and reproducibility |

## Objective

### The scenario

The company runs a *suggested product* feature in its storefront, powered by a
**simple recommendation model already in production**. That model is the
**baseline**: it recommends the most popular items within a product category
(a popularity-by-category recommender — common as a first production system
because it is trivial to ship and needs no training).

The **Commercial Stakeholder** is not satisfied with what this baseline returns.
In their read, the suggestions are generic and category-obvious, and the
revenue and engagement KPIs on the suggested-product surface are flat. They
called a meeting with the **Product Manager** and the **Senior AI Engineer**
to agree on a replacement, to be developed with an AI assistant in the loop.

### What "success" means here — and an explicit honesty boundary

Revenue and engagement are the **motivation** for this initiative: they are why
the Commercial Stakeholder wants a better model, and they frame the Objective.
They are **not** measured in this project, and we will not pretend otherwise.

The dataset behind this product is a **static product catalog** — it contains
product attributes (gender, color, season, usage, article type, display name).

- **Measured success (numbers only from existing data):** standard *offline*
  recommendation quality metrics — **precision@k, recall@k, NDCG@k** — where
  relevance is derived from the catalog columns(see Methodology). This is a content-based evaluation **proxy**, and it is labeled as a proxy everywhere it appears.

The acceptance bar agreed in the meeting: **the new model must beat the
popularity-by-category baseline on the offline recommendation metrics above.**

## Methodology

### The initiative, technically

Build a **multi-modal product recommender** that ranks catalog items by their
relevance to a user's query. The query arrives through one of two interaction
modes the personas agreed to support:

1. **Image selection** — the user picks (or uploads) a product photo.
2. **Chat description** — the user describes the item in natural language to an
   **AI agent**, which decides when to call its tools.

The recommender combines two signals, both of which already exist in this
codebase and are reused rather than rebuilt:

- **Image classification signal** — the trained vision model predicts a
  product's subcategory from its photo. The predicted category (and the
  product's structured attributes) **filter and boost** the candidate set, so
  recommendations stay on-category instead of drifting on raw text similarity.
- **Metadata similarity signal** — semantic similarity over the product
  metadata from the catalog (display name + attributes). One sentence-encoder
  (`all-MiniLM-L6-v2`, cosine) backs both the offline evaluation and the live
  Chroma index, so the measured numbers and the served system use identical math.

The **AI agent** orchestrates these as **tools**: `classify_product` (the
vision model, contract-wrapped) and `search_similar_products` (the retrieval
index). The agent chooses which tool a given query needs; the recommender
composes the final ranking.

### Agreements reached between the personas

- **On the baseline (all three).** The production system is modeled honestly as
  popularity-by-category. The new model must outperform it on the agreed
  offline metrics — a single lucky run does not count as evidence.
- **On metrics (Senior AI Engineer ↔ Product Manager).** Success is measured
  with precision@k, recall@k, and NDCG@k. Relevance ground truth is defined
  **only** from existing catalog columns: an item is relevant to a query item
  when it shares the same `subCategory` / `articleType` and is attribute-
  compatible (e.g., matching `gender`). This proxy and its limitation are
  stated wherever results are reported.
- **On reuse (Senior AI Engineer).** The image classifier feeds the recommender
  as a **category signal**, not as the end product — its predicted subcategory
  filters and boosts the candidate set.
- **On scope across the lifecycle (Product Manager).** The deliverable is framed
  against the four pillars of the data-product lifecycle (below), so the work is
  not "just a model" but a product with a diagnostic rationale and a
  prescriptive output.

### The four pillars of the data-product lifecycle

| Pillar | Question | In this product |
|---|---|---|
| **Descriptive** — *where we are* | What does the catalog and the baseline look like? | Catalog EDA; the popularity-by-category baseline's behavior |
| **Diagnostic** — *why we are there* | Why do the baseline's suggestions underperform? | Why popularity-only recommendations are generic and category-obvious |
| **Predictive** — *what is going to happen* | What will the new model recommend? | The multi-modal recommender (image classification + metadata similarity, via the agent) |
| **Prescriptive** — *what we should do* | What action follows from the results? | A go/no-go recommendation and OKRs for the suggested-product surface |


## Results

**Acceptance criteria (agreed in the kickoff meeting, fixed before any numbers existed):**

- The new multi-modal recommender beats the popularity-by-category baseline on
  **precision@k, recall@k, and NDCG@k** (k ∈ {5, 10}).
- Every reported metric carries its proxy caveat: relevance is content-based
  ground truth derived from existing catalog columns, not observed user behavior.

**Outcome — the acceptance bar is met.** Offline evaluation over 1,000 query
items (category signal = ground truth, isolating retrieval quality):

| Metric | Baseline (popularity) | Proposed (hybrid) | Lift |
|---|---|---|---|
| precision@5 | 0.325 | **0.898** | +176% |
| precision@10 | 0.308 | **0.873** | +184% |
| NDCG@5 | 0.320 | **0.906** | +183% |
| NDCG@10 | 0.309 | **0.888** | +187% |
| recall@5 | 0.003 | 0.022 | +636% |
| recall@10 | 0.006 | 0.037 | +515% |

Reproduce with `python -m src.evaluate_reco --n-queries 1000 --ks 5 10`
(writes `artifacts/reco_metrics_summary.json`).

**Analysis (carried into the README):**

- The margin survived removing an evaluation leak (an early version indexed the
  `articleType` relevance label verbatim and scored a misleading 0.97 precision@5).
- Recall is low by construction — proxy-relevant pools far exceed k — so
  precision and NDCG are the meaningful metrics.
- This is a content-recovery result, not a preference result; a real revenue/
  engagement claim would require an online A/B test, which the dataset cannot support.

**Prescriptive recommendation (Pillar 4).** Ship the hybrid behind the
suggested-product surface with the popularity baseline as fallback; validate the
motivating KPIs with an online A/B test before claiming business value. OKRs:
*Objective* — make suggested-product genuinely useful; *Key Results* — (1) hybrid
live behind a feature flag, (2) A/B test on click-through and add-to-cart,
(3) a calibrated relevance threshold for the recommendation cutoff.
