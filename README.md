# IA Projects

Hey, I'm Vitor — a data professional (econ background, MS in Data Science) currently moving deeper into applied AI engineering. This repo is where I keep hands-on projects I'm building along the way: real problems, real trade-offs.

Background: a few years in analytics and data modeling (segmentation, churn, RFM), most recently building and evaluating an LLM-based recommendation engine in production at a large CPG company. I'm now expanding into the engineering side — training and shipping models, in addition to analyzing data.

## What's in here

Each project gets its own folder with a short write-up: the problem, what I built, why I made the choices I made, and what I'd improve with more time. No filler — just what I'd actually tell you if you asked "walk me through this."

- **[Product Category Classifier](product-category-classifier/)** — multi-modal computer vision (product photo + structured attributes) benchmarked honestly against an image-only baseline across multiple training runs, with a local LLM agent (tool-calling + RAG), a data contract with automated tests, and a Reflex app to demo and monitor it.
- **[Car Review Intelligence Assistant](car-review-intelligence/)** — a multi-skill LLM assistant for an auto dealership: a local routing agent (Ollama, no API keys) dispatches each message to one of four pre-trained Hugging Face skills — sentiment triage, EN→ES translation, extractive QA, and summarization — each held to a baseline and evaluated honestly on a real car-reviews dataset, with a data contract, pytest, and a Reflex chat app.
- **[Customer-Feedback Intelligence](customer-feedback-intelligence/)** — text embeddings over 22.6k real clothing reviews for zero-shot theme triage, sentiment, and similar-review retrieval, evaluated honestly with a linear probe against a TF-IDF baseline (embeddings win on meaning and retrieval, lose on short-text sentiment — and the write-up says so), surfaced through a Reflex support console. Local, no API keys.

## Why this exists

I'm looking for remote AI Analyst / AI Engineer roles. This is the portfolio I point people to when "tell me about a project" comes up.
