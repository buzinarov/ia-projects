"""Per-skill evaluation, each against the baseline and metric fixed in
docs/requirement.md, writing a JSON summary to artifacts/.

Every function carries the honesty boundary into its output: the triage
labels are a rating proxy, and the translate/answer/digest sets are tiny and
hand-built. Those caveats are written into the artifact, not just the README,
so a number can't travel without them.

Run a single skill (e.g. `python -m src.evaluate triage --n 500`) or all of
them via `python -m src.run_all`.
"""
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EVAL_DIR = ROOT_DIR / "data" / "eval"


def _write_artifact(name, summary):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / name
    path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {path}")
    return summary


# --- Triage: accuracy + macro-F1 vs VADER and majority-class -------------

def evaluate_triage(n=500, seed=42):
    """Score the transformer triage skill against VADER and majority-class
    on a held-out, rating-derived-label slice of the Edmunds data."""
    from sklearn.metrics import accuracy_score, f1_score

    from .data import label_balance, load_labeled_reviews, train_eval_split
    from .baselines import majority_class_predictor, vader_sentiment
    from .skills import triage

    df = load_labeled_reviews()
    train_df, eval_df = train_eval_split(df, seed=seed)
    if n and n < len(eval_df):
        eval_df = eval_df.sample(n=n, random_state=seed).reset_index(drop=True)

    reviews = eval_df["review"].tolist()
    gold = eval_df["label"].tolist()

    model_preds = [triage(r)["payload"]["sentiment"] for r in reviews]
    vader_preds = [vader_sentiment(r) for r in reviews]
    majority = majority_class_predictor(train_df["label"].tolist())
    majority_preds = [majority(r) for r in reviews]

    def scores(preds):
        return {
            "accuracy": accuracy_score(gold, preds),
            "macro_f1": f1_score(gold, preds, average="macro"),
        }

    transformer = scores(model_preds)
    vader = scores(vader_preds)
    baseline_majority = scores(majority_preds)
    beat_bar = transformer["macro_f1"] > vader["macro_f1"]

    return _write_artifact("triage_metrics.json", {
        "skill": "triage",
        "n_eval": len(eval_df),
        "label_proxy": "Rating>=4 POSITIVE, <=2 NEGATIVE, 3 dropped (NOT human-annotated sentiment)",
        "eval_label_balance": label_balance(eval_df),
        "transformer": transformer,
        "baseline_vader": vader,
        "baseline_majority_class": baseline_majority,
        "acceptance_metric": "macro_f1",
        "beats_baseline": bool(beat_bar),
    })


# --- Translate: BLEU + chrF on the reference set -------------------------

def evaluate_translate():
    """Score EN->ES translation with BLEU and chrF against the small
    reference set. Tiny set: reported as indicative, not a benchmark."""
    import sacrebleu

    from .skills import translate

    data = json.loads((EVAL_DIR / "translation_references.json").read_text())
    pairs = data["pairs"]

    hyps = [translate(p["source"])["payload"]["translated_text"] for p in pairs]
    # sacrebleu wants references grouped by position: refs[j][i] = ref j of sentence i
    max_refs = max(len(p["references"]) for p in pairs)
    refs = [[p["references"][j] if j < len(p["references"]) else "" for p in pairs] for j in range(max_refs)]

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    chrf = sacrebleu.corpus_chrf(hyps, refs)

    return _write_artifact("translate_metrics.json", {
        "skill": "translate",
        "n_eval": len(pairs),
        "caveat": "Tiny hand-built reference set; BLEU is brittle on short text, chrF reported alongside. Indicative only.",
        "bleu": bleu.score,
        "chrf": chrf.score,
        "examples": [{"source": p["source"], "hypothesis": h} for p, h in zip(pairs, hyps)],
    })


# --- Answer: exact-match / token-F1 on the QA set ------------------------

def _normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(pred, gold):
    pred_toks, gold_toks = _normalize(pred).split(), _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer():
    """Score extractive QA with exact-match and token-F1, taking the best
    over each example's acceptable gold answers."""
    from .skills import answer

    data = json.loads((EVAL_DIR / "qa_examples.json").read_text())
    examples = data["examples"]

    ems, f1s, rows = [], [], []
    for ex in examples:
        pred = answer(ex["question"], ex["context"])["payload"]["answer"]
        em = max(float(_normalize(pred) == _normalize(g)) for g in ex["answers"])
        f1 = max(_token_f1(pred, g) for g in ex["answers"])
        ems.append(em)
        f1s.append(f1)
        rows.append({"question": ex["question"], "prediction": pred, "gold": ex["answers"], "f1": f1})

    return _write_artifact("answer_metrics.json", {
        "skill": "answer",
        "n_eval": len(examples),
        "caveat": "Tiny hand-labeled set; illustrative that the skill is wired correctly, not a benchmark.",
        "exact_match": sum(ems) / len(ems),
        "token_f1": sum(f1s) / len(f1s),
        "examples": rows,
    })


# --- Digest: ROUGE vs lead-3 baseline ------------------------------------

def evaluate_digest():
    """Score the transformer summarizer against the lead-3 baseline with
    ROUGE-1/2/L on the reference-summary set."""
    from rouge_score import rouge_scorer

    from .baselines import lead_n_summary
    from .skills import digest

    data = json.loads((EVAL_DIR / "summary_references.json").read_text())
    examples = data["examples"]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def mean_rouge(summaries):
        agg = {k: 0.0 for k in ("rouge1", "rouge2", "rougeL")}
        for summ, ex in zip(summaries, examples):
            s = scorer.score(ex["reference"], summ)
            for k in agg:
                agg[k] += s[k].fmeasure
        return {k: v / len(examples) for k, v in agg.items()}

    model_summaries = [digest(ex["review"])["payload"]["summary"] for ex in examples]
    lead3_summaries = [lead_n_summary(ex["review"], n=3) for ex in examples]

    transformer = mean_rouge(model_summaries)
    baseline = mean_rouge(lead3_summaries)

    return _write_artifact("digest_metrics.json", {
        "skill": "digest",
        "n_eval": len(examples),
        "caveat": "Tiny reference-summary set; ROUGE is indicative. lead-3 is a famously strong baseline.",
        "transformer": transformer,
        "baseline_lead3": baseline,
        "acceptance_metric": "rougeL",
        "beats_baseline": bool(transformer["rougeL"] > baseline["rougeL"]),
    })


_EVALUATORS = {
    "triage": evaluate_triage,
    "translate": evaluate_translate,
    "answer": evaluate_answer,
    "digest": evaluate_digest,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one skill.")
    parser.add_argument("skill", choices=_EVALUATORS.keys())
    parser.add_argument("--n", type=int, default=500, help="triage eval-set cap")
    args = parser.parse_args()

    if args.skill == "triage":
        summary = evaluate_triage(n=args.n)
    else:
        summary = _EVALUATORS[args.skill]()
    print(json.dumps(summary, indent=2, default=str))
