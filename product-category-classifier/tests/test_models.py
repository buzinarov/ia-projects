"""Model sanity tests: forward-pass shape and no-NaN on a dummy batch.
No data download, no checkpoint, no GPU required -- always run, fast."""
import torch

from src.models import ImageClassifier, build_model

NUM_CLASSES = 27
ATTR_DIM = 62
IMG_SIZE = 80


def test_forward_shape_and_no_nan():
    model = ImageClassifier(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    x_img = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    out = model(x_img)
    assert out.shape == (4, NUM_CLASSES)
    assert not torch.isnan(out).any()


def test_ignores_attr_arg():
    model = ImageClassifier(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model.eval()
    x_img = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    x_attr = torch.randn(2, ATTR_DIM)
    with torch.no_grad():
        out_without = model(x_img)
        out_with = model(x_img, x_attr)
    assert torch.allclose(out_without, out_with)


def test_build_model_returns_image_classifier():
    model = build_model(num_classes=NUM_CLASSES)
    assert isinstance(model, ImageClassifier)
