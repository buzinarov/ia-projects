"""The output data contract for the four skills.

Every skill returns a record built through this module, not an ad-hoc dict,
so the routing agent and the app consume one stable shape per skill and a
malformed result fails loudly instead of leaking a half-formed dict into the
UI. Single source of truth for skill-output shape.
"""
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

CONTRACT_VERSION = "skills-1.0"

SKILLS = ("triage", "translate", "answer", "digest")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _record_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class SkillRecord:
    """Common envelope for every skill result.

    `skill` discriminates the union; `payload` holds the skill-specific
    fields (validated per-skill below). `model_name` is the HF checkpoint
    that produced it, for traceability in the app and artifacts.
    """
    skill: str
    payload: dict
    model_name: str
    record_id: str = field(default_factory=lambda: _record_id("rec"))
    contract_version: str = CONTRACT_VERSION
    created_at: str = field(default_factory=_now)

    def to_dict(self):
        return asdict(self)


# --- payload schemas, one required-key set per skill ---------------------

_PAYLOAD_SCHEMA = {
    "triage": ("sentiment", "confidence"),          # sentiment in {POSITIVE, NEGATIVE}
    "translate": ("source_text", "translated_text", "target_lang"),
    "answer": ("question", "context", "answer", "score"),
    "digest": ("source_text", "summary"),
}


def build_record(skill, payload, model_name):
    """Build and validate a SkillRecord. Raises on a malformed payload."""
    if skill not in SKILLS:
        raise ValueError(f"Unknown skill {skill!r}; expected one of {SKILLS}")
    record = SkillRecord(skill=skill, payload=dict(payload), model_name=model_name)
    validate_record(record.to_dict())
    return record.to_dict()


def validate_record(record):
    """Raise ValueError on a malformed record. A result that feeds the app
    or an artifact should fail here, not get silently passed through."""
    required = ("skill", "payload", "model_name", "record_id", "contract_version", "created_at")
    missing = [f for f in required if f not in record]
    if missing:
        raise ValueError(f"SkillRecord missing required fields: {missing}")

    skill = record["skill"]
    if skill not in SKILLS:
        raise ValueError(f"skill {skill!r} is not one of {SKILLS}")

    payload = record["payload"]
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    payload_missing = [k for k in _PAYLOAD_SCHEMA[skill] if k not in payload]
    if payload_missing:
        raise ValueError(f"{skill} payload missing keys: {payload_missing}")

    _validate_payload_values(skill, payload)

    try:
        datetime.fromisoformat(record["created_at"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"created_at is not ISO 8601: {record['created_at']!r}") from exc

    return True


def _validate_payload_values(skill, payload):
    if skill == "triage":
        if payload["sentiment"] not in ("POSITIVE", "NEGATIVE"):
            raise ValueError(f"triage sentiment must be POSITIVE/NEGATIVE, got {payload['sentiment']!r}")
        _require_unit_interval("confidence", payload["confidence"])
    elif skill == "answer":
        _require_unit_interval("score", payload["score"])
        if not str(payload["answer"]).strip():
            raise ValueError("answer must be non-empty")
    elif skill == "translate":
        if payload["target_lang"] not in ("es", "en"):
            raise ValueError(f"target_lang {payload['target_lang']!r} not supported")
        if not str(payload["translated_text"]).strip():
            raise ValueError("translated_text must be non-empty")
    elif skill == "digest":
        if not str(payload["summary"]).strip():
            raise ValueError("summary must be non-empty")


def _require_unit_interval(name, value):
    if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
