# Kickoff contract — Customer-Feedback Intelligence

*The agreement this project was built against, written before any numbers
existed. It exists so the scope, the success criteria, and — most importantly —
the honesty boundary are fixed up front rather than reverse-engineered from
whatever the results happened to be.*

## The scenario

An online women's-clothing retailer has tens of thousands of free-text customer
reviews and no structured way to use them. Support agents answer each review
from scratch; the merchandising team has no quick read on *what* customers keep
raising. The ask: turn the raw `Review Text` into          something a support agent and a
merchandiser can act on — without standing up a labeling operation first.

## Personas

| Persona | Owns | Cares most about |
|---|---|---|
| **Customer Support Lead** | Reply quality and speed | Triaging an incoming review to a theme, reading its sentiment, and seeing similar past cases so replies are fast and consistent |
| **Merchandising / Product Manager** | Product and catalog decisions | A quick, honest read of what customers talk about (quality, fit, style, comfort, value, look) |
| **Senior AI / ML Engineer** | The embedding pipeline, evaluation rigor, the honesty boundary | Defensible metrics, an honest baseline, one set of vectors shared by eval and product|

## Objective

Build a small, reproducible system over the reviews that delivers, end to end,
the four capabilities from the brief:

1. **Embeddings** — embed every review with a single shared encoder.
2. **2D visualization** — project the embeddings to two dimensions and plot them.
3. **Feedback categorization** — assign each review a theme (quality / fit /
   style / comfort / value / look).
4. **Similarity search** — given a review, return the closest past reviews, so a
   support agent can answer in a consistent, personalized way.

Surfaced through a working **Review Reply Assistant** app (paste a review → get a
ready-to-edit reply), and documented in two notebooks (EDA + the four
deliverables).

## The honesty boundary

This is the line the project holds, because it is the easy thing to fudge:

- **The themes have no ground truth.** The dataset has no theme labels, so the
  theme assignment is an **unsupervised triage aid, not a measured classifier.**
  We never report a "theme accuracy", and we never fabricate one.
- **What we measure instead.** Whether the *embedding space* is meaningful, using
  the labels the dataset actually has: `Recommended IND` (binary sentiment) and
  `Department Name` (product area). A linear probe on frozen embeddings vs. a
  **TF-IDF bag-of-words baseline**, plus a nearest-neighbor retrieval proxy
  against random chance.
- **Embeddings are not assumed to win.** The point of a baseline is that it can
  win. On short-text binary sentiment, a bag-of-words model is genuinely strong —
  and if it beats the embeddings, we say so.

## The acceptance bar (fixed before any numbers)

The embedding-based system is "good enough to build on" if **both** hold:

1. **Semantic structure.** Under a linear probe, embeddings at least *match* the
   TF-IDF baseline on `Department Name` (a semantic label), within a small margin.
2. **Useful retrieval.** Nearest-neighbor retrieval recovers same-department
   reviews **well above random chance** (target: ≥ 2× the chance rate at k=3).

Note what is deliberately *not* in the bar: beating TF-IDF on sentiment. We did
not pre-commit to that, because bag-of-words is a strong sentiment baseline on
short text — and, as the results show, it wins there.

## The data

[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
(Kaggle, by nicapotato) — 23,486 anonymized real reviews, 10 columns. Downloaded
on first use from a public mirror and cached locally; not committed. Only
`Review Text` is the input; the other columns are evaluation labels, never
features for the product itself.
