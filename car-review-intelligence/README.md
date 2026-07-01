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

> **License caveat:** the dataset card states no license. Treated here as
> *demonstration use, provenance to verify* before any redistribution. The
> original five-review `car_reviews.csv` can be dropped into `data/` as a fixed
> demo set for the QA/translation examples.

## Architecture

```
src/
  skills.py     the four skills, each a HF pipeline wrapped behind the contract
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

Populated by `python -m src.run_all` into `artifacts/`. The table below is the
shape of the report; the numbers are filled by a run on your machine (they
depend on the model versions you pull and are not hardcoded here).

| Skill | Metric | Baseline | Transformer | Clears bar? |
|---|---|---|---|---|
| Triage | macro-F1 | VADER: _run_ | _run_ | _run_ |
| Translate | BLEU / chrF | — | _run_ | n/a |
| Answer | token-F1 / EM | — | _run_ | n/a |
| Digest | ROUGE-L | lead-3: _run_ | _run_ | _run_ |

Each artifact also records `n_eval` and the honesty caveat for that skill.

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
