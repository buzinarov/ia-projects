import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_label_maps
from src.inference import load_model


@pytest.fixture(scope="session")
def label_maps():
    try:
        return load_label_maps()
    except FileNotFoundError:
        pytest.skip("No label_maps.json locally; run `python -m src.run_all` first.")


@pytest.fixture(scope="session")
def proposed_model_bundle():
    try:
        return load_model("proposed", seed=0)
    except FileNotFoundError:
        pytest.skip("No trained checkpoint locally (gitignored); run `python -m src.run_all` first.")
