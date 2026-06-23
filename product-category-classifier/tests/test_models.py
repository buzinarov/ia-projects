"""Model sanity tests: forward-pass shape and no-NaN on a dummy batch.
No data download, no checkpoint, no GPU required -- always run, fast."""
import torch

from src.models import BaselineImageModel, MultiModalProductClassifier, build_model

NUM_CLASSES = 27
ATTR_DIM = 62
IMG_SIZE = 80


def test_baseline_forward_shape_and_no_nan():
    model = BaselineImageModel(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    x_img = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    out = model(x_img)
    assert out.shape == (4, NUM_CLASSES)
    assert not torch.isnan(out).any()


def test_baseline_ignores_attr_arg():
    model = BaselineImageModel(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model.eval()
    x_img = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    x_attr = torch.randn(2, ATTR_DIM)
    with torch.no_grad():
        out_without = model(x_img)
        out_with = model(x_img, x_attr)
    assert torch.allclose(out_without, out_with)


def test_proposed_forward_shape_and_no_nan():
    model = MultiModalProductClassifier(num_classes=NUM_CLASSES, attr_dim=ATTR_DIM, img_size=IMG_SIZE)
    x_img = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    x_attr = torch.randn(4, ATTR_DIM)
    out = model(x_img, x_attr)
    assert out.shape == (4, NUM_CLASSES)
    assert not torch.isnan(out).any()


def test_build_model_dispatch():
    baseline = build_model("baseline", num_classes=NUM_CLASSES, attr_dim=ATTR_DIM)
    proposed = build_model("proposed", num_classes=NUM_CLASSES, attr_dim=ATTR_DIM)
    assert isinstance(baseline, BaselineImageModel)
    assert isinstance(proposed, MultiModalProductClassifier)
