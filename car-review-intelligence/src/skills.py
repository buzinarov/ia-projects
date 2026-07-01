"""The four skills behind the assistant, each a thin wrapper over a
pre-trained Hugging Face model that returns a validated contract record.

Models are loaded lazily and cached per checkpoint, so importing this module
is free (no downloads) and a process that only ever triages reviews never
pulls the translation or summarization weights. The routing agent
(src/agent.py) calls these; the app and evaluation call them too.

Only the triage skill uses the high-level `pipeline()` API: transformers 5.x
removed the seq2seq and extractive-QA pipeline tasks ("translation",
"summarization", "question-answering"), so translate/digest/answer are wired
directly through the Auto* model classes + `generate()`. That's also the
version-robust path and mirrors the original exercise's QA code.

Defaults match docs/requirement.md:
    triage     distilbert-base-uncased-finetuned-sst-2-english
    translate  Helsinki-NLP/opus-mt-en-es
    answer     deepset/minilm-uncased-squad2
    digest     cnicu/t5-small-booksum
"""
from functools import lru_cache

from .contract import build_record

TRIAGE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-en-es"
ANSWER_MODEL = "deepset/minilm-uncased-squad2"
DIGEST_MODEL = "cnicu/t5-small-booksum"


@lru_cache(maxsize=None)
def _pipeline(task, model):
    """Build (once) and cache a Hugging Face pipeline for (task, model)."""
    from transformers import pipeline

    return pipeline(task, model=model)


@lru_cache(maxsize=None)
def _qa_model(model):
    """Build (once) and cache an extractive-QA (tokenizer, model) pair.

    Extractive QA is wired through AutoModelForQuestionAnswering + manual
    span extraction rather than a pipeline: the plain "question-answering"
    pipeline task was removed in transformers 5.x, and the explicit path is
    version-robust (it's also what the original exercise used).
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    return AutoTokenizer.from_pretrained(model), AutoModelForQuestionAnswering.from_pretrained(model)


@lru_cache(maxsize=None)
def _seq2seq_model(model):
    """Build (once) and cache a seq2seq (tokenizer, model) pair, used by
    translate and digest now that the seq2seq pipeline tasks are gone."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    return AutoTokenizer.from_pretrained(model), AutoModelForSeq2SeqLM.from_pretrained(model)


def _generate(model, text, max_length, min_length=None, no_repeat_ngram_size=None):
    """Run a seq2seq model and decode the first sequence."""
    tokenizer, seq2seq = _seq2seq_model(model)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    gen_kwargs = {"max_length": max_length}
    if min_length is not None:
        gen_kwargs["min_length"] = min_length
    if no_repeat_ngram_size is not None:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    output_ids = seq2seq.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# --- Triage: sentiment of an incoming review -----------------------------

def triage(review, model=TRIAGE_MODEL):
    """Classify a review's sentiment. Returns a 'triage' contract record."""
    clf = _pipeline("sentiment-analysis", model)
    out = clf(review, truncation=True)[0]
    return build_record(
        "triage",
        {"sentiment": out["label"], "confidence": float(out["score"])},
        model_name=model,
    )


# --- Translate: EN -> ES for customers -----------------------------------

def translate(text, target_lang="es", model=TRANSLATE_MODEL, max_length=400):
    """Translate English text to Spanish. Returns a 'translate' record."""
    if target_lang != "es":
        raise ValueError("Only EN->ES is wired up in this prototype")
    translated = _generate(model, text, max_length=max_length)
    return build_record(
        "translate",
        {"source_text": text, "translated_text": translated, "target_lang": target_lang},
        model_name=model,
    )


# --- Answer: extractive QA grounded in one review ------------------------

def answer(question, context, model=ANSWER_MODEL):
    """Answer a question from a context passage (extractive QA).

    Returns an 'answer' record. The best span landing on the [CLS] token is
    the SQuAD2 "no answer" signal; that and an empty span are surfaced as an
    explicit "No answer found" with score 0.0, so the contract's
    non-empty-answer rule holds and the app can say so honestly.
    """
    import torch

    tokenizer, qa_model = _qa_model(model)
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_probs = outputs.start_logits.softmax(dim=-1)[0]
    end_probs = outputs.end_logits.softmax(dim=-1)[0]
    start_idx = int(start_probs.argmax())
    end_idx = int(end_probs.argmax())
    score = float(start_probs[start_idx] * end_probs[end_idx])

    span_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
    text = tokenizer.decode(span_ids, skip_special_tokens=True).strip()
    # start_idx == 0 (the [CLS] slot) or an inverted/empty span = "no answer"
    if start_idx == 0 or end_idx < start_idx or not text:
        text, score = "No answer found in this review.", 0.0
    return build_record(
        "answer",
        {"question": question, "context": context, "answer": text, "score": score},
        model_name=model,
    )


# --- Digest: summarize a long review -------------------------------------

def digest(text, model=DIGEST_MODEL, max_length=60, min_length=20):
    """Summarize a review to ~50-55 tokens. Returns a 'digest' record.

    T5-family models expect a task prefix; the summarization pipeline used to
    add it, so we add it explicitly now that we call generate() directly.
    `no_repeat_ngram_size` blocks the small model from padding to min_length
    by repeating a sentence (a tic visible at higher min_length values).
    """
    summary = _generate(
        model, f"summarize: {text}",
        max_length=max_length, min_length=min_length, no_repeat_ngram_size=3,
    )
    return build_record(
        "digest",
        {"source_text": text, "summary": summary},
        model_name=model,
    )


SKILL_FUNCS = {
    "triage": triage,
    "translate": translate,
    "answer": answer,
    "digest": digest,
}
