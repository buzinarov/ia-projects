"""The output data contract this classifier promises downstream
consumers (other ML models, executive/operational reports). Single
source of truth for prediction record shape -- inference.py and
agent.py build records through here, not ad hoc dicts.
"""
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

MODEL_VERSION = "subcategory-1.0"  # bump when target/architecture changes meaningfully


@dataclass
class PredictionRecord:
    product_id: str
    predicted_subcategory: str
    confidence: float
    model_name: str  # "baseline" | "proposed"
    model_version: str
    predicted_at: str  # ISO 8601 UTC

    def to_dict(self):
        return asdict(self)


def build_prediction_record(predicted_class, confidence, model_name, product_id=None):
    return PredictionRecord(
        product_id=product_id or f"upload-{uuid.uuid4().hex[:8]}",
        predicted_subcategory=predicted_class,
        confidence=float(confidence),
        model_name=model_name,
        model_version=MODEL_VERSION,
        predicted_at=datetime.now(timezone.utc).isoformat(),
    ).to_dict()


def validate_prediction_record(record, valid_classes):
    """Raises ValueError on a malformed record -- a record that feeds
    downstream models/reports should fail loudly, not get silently
    clamped or passed through."""
    required = ("product_id", "predicted_subcategory", "confidence", "model_name", "model_version", "predicted_at")
    missing = [f for f in required if f not in record]
    if missing:
        raise ValueError(f"Prediction record missing required fields: {missing}")

    if record["predicted_subcategory"] not in valid_classes:
        raise ValueError(
            f"predicted_subcategory {record['predicted_subcategory']!r} is not a valid class"
        )

    confidence = record["confidence"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be a number in [0, 1], got {confidence!r}")

    if record["model_name"] not in ("baseline", "proposed"):
        raise ValueError(f"model_name must be 'baseline' or 'proposed', got {record['model_name']!r}")

    try:
        datetime.fromisoformat(record["predicted_at"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"predicted_at is not a valid ISO 8601 timestamp: {record['predicted_at']!r}") from exc

    return True
