"""Regression gate for the recommender's image signal: fails if a retrain
drops the vision model's quality below an explicit floor. Floors are set
from the first real `run_all` execution (roughly mean - 1.5*std), not
guessed. Skips entirely if aggregated metrics aren't present locally
(gitignored checkpoints mean a bare clone has none until `run_all` runs).
"""
import json
from pathlib import Path

import pytest

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Calibrated from the real run (3 seeds, 25 epochs, all 4 attributes):
# baseline accuracy 0.931 +/- 0.0003, macro_f1 0.854 +/- 0.002.
# proposed accuracy 0.883 +/- 0.009, macro_f1 0.803 +/- 0.007.
# Floors = mean - 1.5*std, rounded down for margin (baseline's std is
# tiny across only 3 seeds, so its floor is rounded down more generously
# than the formula alone would give). The image-only baseline is the
# signal the recommender ships; the proposed variant trails it (see the
# README appendix). These gates protect each model from regressing
# further; they don't claim the proposed variant is the better choice.
BASELINE_ACCURACY_FLOOR = 0.92
BASELINE_MACRO_F1_FLOOR = 0.84
PROPOSED_ACCURACY_FLOOR = 0.86
PROPOSED_MACRO_F1_FLOOR = 0.79


def _load_summary(model_name):
    path = ARTIFACTS_DIR / f"metrics_{model_name}_summary.json"
    if not path.exists():
        pytest.skip("No aggregated metrics locally; run `python -m src.run_all` first.")
    return json.loads(path.read_text())


@pytest.fixture
def baseline_summary():
    return _load_summary("baseline")


@pytest.fixture
def proposed_summary():
    return _load_summary("proposed")


def test_baseline_accuracy_above_floor(baseline_summary):
    assert baseline_summary["accuracy"]["mean"] >= BASELINE_ACCURACY_FLOOR


def test_baseline_macro_f1_above_floor(baseline_summary):
    assert baseline_summary["macro_f1"]["mean"] >= BASELINE_MACRO_F1_FLOOR


def test_proposed_accuracy_above_floor(proposed_summary):
    assert proposed_summary["accuracy"]["mean"] >= PROPOSED_ACCURACY_FLOOR


def test_proposed_macro_f1_above_floor(proposed_summary):
    assert proposed_summary["macro_f1"]["mean"] >= PROPOSED_MACRO_F1_FLOOR
