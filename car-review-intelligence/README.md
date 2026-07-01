# Car-ing is sharing — Review Intelligence Assistant

[![CI](https://github.com/buzinarov/ia-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/buzinarov/ia-projects/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

A multi-skill **LLM assistant** for an auto dealership, built from pre-trained
Hugging Face models behind a **local routing agent**. A support agent or
customer sends one free-text message; a local Ollama model reads it and
dispatches to the right specialized skill — **triage** a review's sentiment,
**translate** it to Spanish, **answer** a question grounded in a review, or
**digest** a long one. No API keys: every model runs locally, so nothing can
leak from a public repo.

>The full kickoff contract is in [`docs/requirement.md`](docs/requirement.md).

## What it does

| Skill | The job | Model (default) | Baseline to beat | Honest metric |
|---|---|---|---|---|
| **Triage** | sentiment of an incoming review | `distilbert-base-uncased-finetuned-sst-2-english` | VADER + majority-class | accuracy + **macro-F1** |
| **Translate** | EN→ES for customers | `Helsinki-NLP/opus-mt-en-es` | — | BLEU + chrF |
| **Answer** | extractive QA from one review | `deepset/minilm-uncased-squad2` | — | exact-match / token-F1 |
| **Digest** | summarize a long review | `cnicu/t5-small-booksum` | lead-3 truncation | ROUGE-1/2/L |

The **routing agent** ([`src/agent.py`](src/agent.py)) is the glue: a local
Ollama model with the four skills as tools, picking one per message and
extracting its arguments from the text. When no Ollama server is reachable it
falls back to a deliberately dumb keyword router, so the app and the test suite
still demonstrate dispatch offline.

## The honesty boundary

This project's signature, carried over from its sibling
[product-category-classifier](../product-category-classifier/): every number
travels with what it can and cannot prove.

- **Sentiment labels are a rating proxy.** Triage is evaluated against labels
  derived from the star `Rating` (**≥4 → positive, ≤2 → negative, 3 dropped as
  ambiguous**) — agreement with *rating-implied* sentiment, **not** human
  annotation. A 5-star "great car, terrible dealership" review is noise the
  proxy can't resolve, and the artifact says so.
- **Class skew is why the bar is macro-F1, not accuracy.** Most reviews are
  positive; a model that always says "positive" would post high accuracy and be
  useless for triage. Macro-F1 keeps the negative minority in scope.
- **QA / Translate / Digest eval sets are tiny and hand-built.** They show the
  skill is wired correctly and produce indicative numbers — not a benchmark.
  Every artifact carries that caveat in a `caveat` field.

## The data

[`florentgbelidji/edmunds-car-ratings`](https://huggingface.co/datasets/florentgbelidji/edmunds-car-ratings)
— Edmunds consumer car reviews on the Hugging Face Hub. Loads through the
`datasets` library, no auth. `Review` text + a 1–5 `Rating`, ~1K–10K rows.


## Architecture

```
src/
  skills.py     the four skills (HF pipeline / Auto* models) wrapped behind the contract
  agent.py      the routing agent (Ollama tool-calling + keyword fallback) + dispatch
  contract.py   the output data contract: one validated record shape per skill
  baselines.py  VADER / majority-class (triage) and lead-3 (digest) — the bars to beat
  data.py       Edmunds loader, rating→sentiment proxy, seeded stratified split, caching
  evaluate.py   per-skill metrics vs the baselines -> artifacts/*.json
  run_all.py    run every skill's evaluation in one command
notebooks/
  01_eda.ipynb        the dataset and the label proxy (Descriptive)
  02_case_study.ipynb the four skills evaluated end to end
tests/          pytest: contract, router dispatch, baselines/proxy, skill smoke tests
data/eval/      small hand-built reference sets (translation, QA, summaries)
artifacts/      per-skill metric summaries (written by run_all)
app/app/
  app.py               the Reflex chat UI
  assistant_service.py in-process bridge to the routing agent
docs/           requirement.md (personas, baselines, acceptance bar, honesty boundary)
```

A **data contract** ([`src/contract.py`](src/contract.py)) pins one validated
record shape per skill; a malformed result raises instead of leaking a
half-formed dict into the app or an artifact.

## Running it

```bash
pip install -r requirements.txt

# evaluate every skill against its baseline -> artifacts/*.json
python -m src.run_all                 # first run downloads the models, then caches
python -m src.evaluate triage --n 1000  # a single skill, larger triage slice

pytest                                # contract + routing + baseline tests always run
RUN_MODEL_TESTS=1 pytest              # also run the skill smoke tests (downloads models)

ollama pull llama3.1:8b               # the routing agent's local model (optional)
cd app && reflex run                  # the chat assistant at http://localhost:3000
```

The app and the agent run with **no Ollama** too — they fall back to keyword
routing — so you can demo dispatch before pulling a model.

## Results

Real numbers from `python -m src.run_all` (transformers 5.12, CPU), written to
`artifacts/*.json` with each skill's `n_eval` and honesty caveat. **The
headline is itself honest: the off-the-shelf transformers do not uniformly beat
the dumb baselines** — which is precisely what the acceptance bar exists to
surface.

| Skill | Metric | Baseline | Transformer | Clears the bar? |
|---|---|---|---|---|
| **Triage** | macro-F1 (n=500) | VADER **0.68** | 0.63 | ❌ no |
| **Translate** | BLEU / chrF (n=4) | — | 54.9 / 75.4 | n/a |
| **Answer** | token-F1 / EM (n=4) | — | 0.75 / 0.75 | n/a |
| **Digest** | ROUGE-L (n=3) | lead-3 **0.181** | 0.181 | ~tie |

One honest read per skill:

- **Triage — the transformer loses to VADER, and that's reported, not buried.**
  `distilbert-sst2` is trained on movie reviews; on car reviews scored against
  the rating proxy it posts macro-F1 **0.63 vs VADER's 0.68** (and lower
  accuracy, 0.78 vs 0.88 — it over-calls "negative" on mixed-but-positive
  reviews). The bar was **not** cleared. The honest next step is a
  car-domain or fine-tuned sentiment model — not shipping distilbert just
  because it is a transformer.
- **Digest — essentially a tie with lead-3.** ROUGE-L is 0.181 vs 0.181, and
  lead-3 actually wins ROUGE-1/2. On this tiny set (n=3) the abstractive model
  doesn't justify its latency over truncation, so the documented call is to
  keep lead-3 as a live option — exactly what the requirement allowed for.
- **Translate / Answer — indicative and honestly small.** BLEU 54.9 / chrF 75.4
  and F1/EM 0.75 look strong, but n=4 each: they show the skills are wired
  correctly and produce sensible output, not a benchmark result.

The negative results are the point. The acceptance bar — fixed before any
numbers existed — did its job: it stopped two "just use a transformer" defaults
from being sold as wins.

## Limitations & next steps

- **Proxy labels, not annotations.** The headline caveat: Triage is scored
  against rating-implied sentiment. A small human-annotated sentiment set would
  turn the proxy result into a real one.
- **Tiny eval sets for three skills.** Translate/Answer/Digest are illustrative.
  Native-speaker reference translations and a larger labeled QA set are the
  honest path to defensible numbers.
- **Routing isn't measured yet.** The agent routes; a labeled set of
  message→skill pairs would let it be scored like the skills are. That's the
  next contract to write.
