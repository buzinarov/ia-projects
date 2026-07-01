"""Smoke tests that each skill loads its model and returns a contract-valid
record. Opt-in (RUN_MODEL_TESTS=1) because they download from the Hub."""
from conftest import run_model_tests

from src.contract import validate_record


@run_model_tests
def test_triage_smoke():
    from src.skills import triage

    rec = triage("This is the best car I have ever owned. Absolutely love it.")
    assert validate_record(rec)
    assert rec["payload"]["sentiment"] in ("POSITIVE", "NEGATIVE")


@run_model_tests
def test_translate_smoke():
    from src.skills import translate

    rec = translate("I love this car.")
    assert validate_record(rec)
    assert rec["payload"]["target_lang"] == "es"


@run_model_tests
def test_answer_smoke():
    from src.skills import answer

    rec = answer(
        "Why did the customer choose the brand?",
        "I chose this brand because of its reputation for reliability.",
    )
    assert validate_record(rec)
    assert rec["payload"]["answer"].strip()


@run_model_tests
def test_digest_smoke():
    from src.skills import digest

    long_review = (
        "I have owned this SUV for two years and it has been excellent for our "
        "family. The third row is usable, the cargo space is huge, and we have "
        "had no mechanical issues at all. The only downside is the dated infotainment."
    )
    rec = digest(long_review)
    assert validate_record(rec)
    assert rec["payload"]["summary"].strip()
