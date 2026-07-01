# Requirement — "Car-ing is sharing" Review Intelligence Assistant

> The kickoff contract. Written before any numbers existed, so the
> acceptance bar can't drift to meet whatever the models happen to score.

## Scenario

**Car-ing is sharing** is an auto dealership (sales + rental). Customer
reviews arrive faster than the support team can read them, and a growing
share of customers are Spanish-speaking. The CTO wants to prototype an
assistant with two audiences:

- **Support agents** — need to triage a flood of reviews (who's unhappy,
  right now?), digest long ones, and answer "what did this customer
  actually say about X?" without reading every paragraph.
- **Customers** — increasingly Spanish-speaking; need reviews and replies
  served in their language.

The prototype is **one assistant** that reads a free-text message and
routes it to the right specialized skill. The routing *is* the product;
the skills are the capabilities behind it.

## Personas

The requirement reflects agreements between three roles, not one wish list.

| Persona | Owns | Cares most about |
|---|---|---|
| **Support Lead** | What agents need day-to-day | Triage that surfaces unhappy customers; digests that save reading time |
| **Localization/CX Stakeholder** | The Spanish-speaking customer experience | Translations good enough to publish, measured not vibed |
| **AI Engineer** | Skill design, evaluation rigor, the data contract | Defensible metrics, honest baselines, reproducibility, no API keys in a public repo |

## The four skills

Each skill is a pre-trained Hugging Face pipeline wrapped behind the data
contract. Here each is a tool the routing agent can call, and each is held
to a baseline and a metric fixed in this document.

| Skill | Job | Model (default) | Baseline to beat | Honest metric |
|---|---|---|---|---|
| **Triage** | sentiment of an incoming review | `distilbert-base-uncased-finetuned-sst-2-english` | VADER lexicon + majority-class | accuracy + **macro-F1** on rating-derived labels |
| **Translate** | EN→ES for customers | `Helsinki-NLP/opus-mt-en-es` | — (no naive MT baseline worth shipping) | **BLEU + chrF** on a real reference set |
| **Answer** | extractive QA grounded in one review | `deepset/minilm-uncased-squad2` | — | exact-match / token-F1 on a small hand-labeled set |
| **Digest** | summarize a long review (~50–55 tokens) | `cnicu/t5-small-booksum` | lead-3 / first-N-chars truncation | ROUGE-1/2/L vs reference summaries (indicative) |

## The acceptance bar (fixed at kickoff)

- **Triage** must beat the VADER baseline on **macro-F1** on the held-out
  evaluation split. Macro-F1, not accuracy, because the rating-derived
  labels are imbalanced (most reviews are positive) and we care about
  catching the negative minority — that's the whole point of triage.
- **Digest** must beat lead-3 on **ROUGE-L** on the reference-summary set,
  *or* we ship lead-3 and say so. A transformer that can't beat
  truncation isn't worth the latency.
- **Translate / Answer** have no baseline to beat — they are reported with
  their metrics and honest caveats about evaluation-set size, not sold as
  "state of the art."

## The honesty boundary

This project's signature, carried over from the classifier project: every
number travels with what it can and cannot prove.

- **Sentiment labels are a proxy.** We derive them from the star
  `Rating`: **≥ 4 → positive, ≤ 2 → negative, the middle (3) is dropped**
  as genuinely ambiguous. This measures agreement with *rating-implied*
  sentiment, **not** human-annotated sentiment. A 5-star review that says
  "great car, terrible dealership" is noise the proxy can't resolve, and
  we say so.
- **QA and summarization eval sets are tiny.** They are hand-built on a
  handful of reviews to show the skill works and is wired correctly — they
  are illustrative, not a benchmark. Stated wherever the numbers appear.
- **BLEU on one sentence is noise.** Translation is evaluated on a real
  reference set with multiple references where possible, and chrF is
  reported alongside BLEU because BLEU is brittle on short text.

## Results against the bar (first run)

First evaluation (`python -m src.run_all`, transformers 5.12, CPU; full
per-skill artifacts in `artifacts/*.json`):

| Skill | Acceptance metric | Baseline | Transformer | Verdict |
|---|---|---|---|---|
| **Triage** | macro-F1 (n=500) | VADER **0.68** | 0.63 | **bar NOT cleared** |
| **Digest** | ROUGE-L (n=3) | lead-3 **0.181** | 0.181 | tie (lead-3 wins ROUGE-1/2) |
| **Translate** | BLEU / chrF (n=4) | — | 54.9 / 75.4 | indicative |
| **Answer** | token-F1 / EM (n=4) | — | 0.75 / 0.75 | indicative |

Read against the bar this document fixed *before* the numbers existed:

- **Triage did not clear the bar.** `distilbert-sst2` (movie-domain) scores
  macro-F1 0.63 against VADER's 0.68 on the rating-proxy labels, and lower
  accuracy (0.78 vs 0.88) — it over-predicts "negative" on the
  majority-positive set. Per this contract that is a **do-not-ship** for the
  transformer as-is; the recommended path is a car-domain / fine-tuned model,
  not shipping it because it is a transformer.
- **Digest ties lead-3** on ROUGE-L (0.181 vs 0.181) and loses ROUGE-1/2. The
  bar explicitly allowed "ship lead-3 and say so" — on this evidence lead-3
  stays a live option.
- **Translate / Answer** have no baseline; reported with their honest n=4
  caveat, not as benchmarks.

This is the acceptance bar working as designed: it turned two "use a
transformer" defaults into documented, defensible non-wins rather than
headline claims.

## Data

[`florentgbelidji/edmunds-car-ratings`](https://huggingface.co/datasets/florentgbelidji/edmunds-car-ratings)
— Edmunds consumer car reviews on the Hugging Face Hub. Loads through the
`datasets` library, no auth. Columns: `Review_Date`, `Author_Name`,
`Vehicle_Title`, `Review_Title`, `Review`, `Rating` (1–5 float). ~1K–10K
rows.

## What "done" looks like for the prototype

1. The four skills run from pre-trained models, each returning a record
   that passes the data contract.
2. The routing agent maps a free-text message to the right skill (or asks
   for clarification) using a **local** Ollama model — nothing that needs
   an API key in a public repo.
3. Each skill has its metric computed on a real (or honestly-labeled-small)
   set, written to `artifacts/`, with the baseline comparison where one
   exists.
4. A Reflex chat app demonstrates the assistant end to end.
5. `pytest` covers the contract, the router's dispatch logic, and a smoke
   test per skill.
