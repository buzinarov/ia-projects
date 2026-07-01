"""Pure-function tests for the rating->label proxy and the baselines. No
dataset download or model needed."""
import math

import pytest

from src.baselines import lead_n_summary, majority_class_predictor
from src.data import rating_to_label
from src.evaluate import _normalize, _token_f1


@pytest.mark.parametrize("rating,expected", [
    (5.0, "POSITIVE"),
    (4.0, "POSITIVE"),
    (4.125, "POSITIVE"),
    (3.0, None),     # ambiguous middle is dropped, not bucketed
    (3.5, None),
    (2.0, "NEGATIVE"),
    (1.0, "NEGATIVE"),
    (None, None),
    (float("nan"), None),
])
def test_rating_to_label_proxy(rating, expected):
    assert rating_to_label(rating) == expected


def test_lead_n_summary_takes_first_sentences():
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    out = lead_n_summary(text, n=2)
    assert out == "First sentence. Second sentence!"


def test_lead_n_summary_handles_short_text():
    assert lead_n_summary("Only one.", n=3) == "Only one."


def test_majority_class_predictor():
    predict = majority_class_predictor(["POSITIVE", "POSITIVE", "NEGATIVE"])
    assert predict("anything") == "POSITIVE"


def test_token_f1_exact_match_is_one():
    assert math.isclose(_token_f1("the brand reputation", "brand reputation"), 1.0)


def test_token_f1_no_overlap_is_zero():
    assert _token_f1("fuel economy", "leather seats") == 0.0


def test_normalize_strips_articles_and_punctuation():
    assert _normalize("The Infotainment System!") == "infotainment system"
