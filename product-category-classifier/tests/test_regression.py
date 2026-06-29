"""Regression gate for the recommender's image signal: fails if a retrain
drops the classifier's quality below an explicit floor. Floors are set
from the first real `run_all` execution (roughly mean - 1.5*std), not
guessed. Skips entirely if the aggregated metrics aren't present locally
(gitignored checkpoints mean a bare clone has none until `run_all` runs).
"""
import json
from pathlib import Path

import pytest

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_NAME = "image_classifier"

# Calibrated from the real run (3 seeds, 25 epochs):
# accuracy 0.931 +/- 0.0003, macro_f1 0.854 +/- 0.002.
# Floors = mean - 1.5*std, rounded down for margin (the std is tiny across
# only 3 seeds, so the floor is rounded down more generously than the
# formula alone would give).
ACCURACY_FLOOR = 0.92
MACRO_F1_FLOOR = 0.84


def _load_summary():
    path = ARTIFACTS_DIR / f"metrics_{MODEL_NAME}_summary.json"
    if not path.exists():
        pytest.skip("No aggregated metrics locally; run `python -m src.run_all` first.")
    return json.loads(path.read_text())


@pytest.fixture
def summary():
    return _load_summary()


def test_accuracy_above_floor(summary):
    assert summary["accuracy"]["mean"] >= ACCURACY_FLOOR


def test_macro_f1_above_floor(summary):
    assert summary["macro_f1"]["mean"] >= MACRO_F1_FLOOR
