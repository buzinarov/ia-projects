import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Skill smoke tests download pre-trained models from the Hub, so they are
# opt-in: set RUN_MODEL_TESTS=1 to run them locally. CI runs everything else.
run_model_tests = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to run tests that download Hugging Face models",
)
