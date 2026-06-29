"""Schema/bounds/enum tests for the output data contract. No model or
trained checkpoint needed -- these always run, even on a bare clone."""
import pytest

from src.contract import build_prediction_record, validate_prediction_record

REQUIRED_FIELDS = ("product_id", "predicted_subcategory", "confidence", "model_name", "model_version", "predicted_at")


def test_build_record_has_required_fields():
    record = build_prediction_record("Topwear", 0.9, "image_classifier", product_id="p1")
    for field in REQUIRED_FIELDS:
        assert field in record


def test_build_record_defaults_product_id_when_missing():
    record = build_prediction_record("Topwear", 0.9, "image_classifier")
    assert record["product_id"].startswith("upload-")


def test_validate_accepts_well_formed_record(label_maps):
    record = build_prediction_record(label_maps["target_classes"][0], 0.9, "image_classifier", product_id="p1")
    assert validate_prediction_record(record, label_maps["target_classes"]) is True


def test_validate_rejects_unknown_class(label_maps):
    record = build_prediction_record("NotARealSubCategory", 0.9, "image_classifier", product_id="p1")
    with pytest.raises(ValueError):
        validate_prediction_record(record, label_maps["target_classes"])


def test_validate_rejects_out_of_bounds_confidence(label_maps):
    record = build_prediction_record(label_maps["target_classes"][0], 1.5, "image_classifier", product_id="p1")
    with pytest.raises(ValueError):
        validate_prediction_record(record, label_maps["target_classes"])


def test_validate_rejects_missing_field(label_maps):
    record = build_prediction_record(label_maps["target_classes"][0], 0.9, "image_classifier", product_id="p1")
    del record["confidence"]
    with pytest.raises(ValueError):
        validate_prediction_record(record, label_maps["target_classes"])


def test_validate_rejects_empty_model_name(label_maps):
    record = build_prediction_record(label_maps["target_classes"][0], 0.9, "", product_id="p1")
    with pytest.raises(ValueError):
        validate_prediction_record(record, label_maps["target_classes"])


def test_validate_rejects_malformed_timestamp(label_maps):
    record = build_prediction_record(label_maps["target_classes"][0], 0.9, "image_classifier", product_id="p1")
    record["predicted_at"] = "not-a-timestamp"
    with pytest.raises(ValueError):
        validate_prediction_record(record, label_maps["target_classes"])
