"""Bridges the Reflex app to the project's src/ package and artifacts/.
Single place that knows the on-disk layout, so pages don't each
reimplement path resolution or model caching.
"""
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SAMPLES_DIR = PROJECT_ROOT / "app" / "assets" / "sample_images"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_label_maps  # noqa: E402
from src.inference import load_model, predict  # noqa: E402

_cache = {}


def get_label_maps():
    if "maps" not in _cache:
        _cache["maps"] = load_label_maps()
    return _cache["maps"]


def get_vision_models(seed=0):
    key = f"models_seed{seed}"
    if key not in _cache:
        _cache[key] = {
            "baseline": load_model("baseline", seed=seed),
            "proposed": load_model("proposed", seed=seed),
        }
    return _cache[key]


def run_prediction(model_name, pil_image, attrs, seed=0):
    model, maps, device = get_vision_models(seed=seed)[model_name]
    return predict(model, maps, device, pil_image, attrs)


def load_metrics_summary(model_name):
    path = ARTIFACTS_DIR / f"metrics_{model_name}_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_history(model_name, seed=0):
    path = ARTIFACTS_DIR / f"history_{model_name}_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def image_data_uri(path: Path) -> str:
    """Embeds a PNG (e.g. a confusion matrix plot from artifacts/) as a
    base64 data URI, so it can be shown via rx.image without needing
    the file to live inside Reflex's assets/ static folder."""
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_sample_manifest():
    path = SAMPLES_DIR / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def generate_verdict(baseline_summary, proposed_summary):
    """Builds the one-line plain-language verdict from the actual
    aggregated numbers -- never hardcoded, so it can't go stale on a
    rerun with different results."""
    acc_delta = proposed_summary["accuracy"]["mean"] - baseline_summary["accuracy"]["mean"]
    macro_delta = proposed_summary["macro_f1"]["mean"] - baseline_summary["macro_f1"]["mean"]

    class_names = proposed_summary["class_names"]
    f1_deltas = {
        c: proposed_summary["per_class_f1"][c]["mean"] - baseline_summary["per_class_f1"][c]["mean"]
        for c in class_names
    }
    best_class = max(f1_deltas, key=f1_deltas.get)
    worst_class = min(f1_deltas, key=f1_deltas.get)

    seeds = proposed_summary["seeds"]
    macro_direction = "ahead of" if macro_delta >= 0 else "behind"
    acc_direction = "ahead of" if acc_delta >= 0 else "behind"

    return (
        f"Across {len(seeds)} seeds, the proposed model is {macro_direction} the baseline on macro-F1 "
        f"by {abs(macro_delta):.3f} ({proposed_summary['macro_f1']['mean']:.3f} vs. "
        f"{baseline_summary['macro_f1']['mean']:.3f}), and {acc_direction} "
        f"on accuracy by {abs(acc_delta):.1%}. "
        f"Clearest win: {best_class} ({f1_deltas[best_class]:+.3f} F1). "
        f"Clearest regression: {worst_class} ({f1_deltas[worst_class]:+.3f} F1)."
    )
