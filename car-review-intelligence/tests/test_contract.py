"""Schema/bounds/enum tests for the skill output contract. No model needed
-- these always run, even on a bare clone."""
import pytest

from src.contract import build_record, validate_record


def test_triage_record_round_trips():
    rec = build_record("triage", {"sentiment": "POSITIVE", "confidence": 0.91}, "distilbert")
    assert validate_record(rec) is True
    assert rec["skill"] == "triage"
    assert rec["record_id"] and rec["created_at"]


def test_unknown_skill_rejected():
    with pytest.raises(ValueError):
        build_record("vibe-check", {"x": 1}, "m")


def test_triage_rejects_bad_sentiment():
    with pytest.raises(ValueError):
        build_record("triage", {"sentiment": "MEH", "confidence": 0.5}, "m")


def test_triage_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        build_record("triage", {"sentiment": "POSITIVE", "confidence": 1.4}, "m")


def test_answer_rejects_empty_answer():
    with pytest.raises(ValueError):
        build_record("answer", {"question": "q", "context": "c", "answer": "  ", "score": 0.5}, "m")


def test_translate_rejects_unsupported_lang():
    with pytest.raises(ValueError):
        build_record(
            "translate",
            {"source_text": "hi", "translated_text": "hola", "target_lang": "fr"},
            "m",
        )


def test_digest_rejects_empty_summary():
    with pytest.raises(ValueError):
        build_record("digest", {"source_text": "long review", "summary": ""}, "m")


def test_validate_rejects_missing_payload_key():
    rec = build_record("triage", {"sentiment": "POSITIVE", "confidence": 0.5}, "m")
    del rec["payload"]["confidence"]
    with pytest.raises(ValueError):
        validate_record(rec)


def test_validate_rejects_malformed_timestamp():
    rec = build_record("digest", {"source_text": "x", "summary": "y"}, "m")
    rec["created_at"] = "not-a-timestamp"
    with pytest.raises(ValueError):
        validate_record(rec)
