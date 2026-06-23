"""Model architectures for the product category classifier.

BaselineImageModel: image-only CNN.
MultiModalProductClassifier: image + structured attributes (gender,
baseColour, season, usage), a near 1:1 translation of the multi-input
pattern (image_layer + type_layer -> concat -> classifier) from the
original cert exercise this case study is based on, with a deeper
shared CNN trunk and a wider attribute branch to match the richer
inputs and finer-grained target used in v2.
"""
import torch
import torch.nn as nn


def _build_image_trunk(img_size):
    """Shared CNN trunk -- identical for both models, so the only real
    difference between baseline and proposed is whether the attribute
    branch exists, not a stronger image branch on one side."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
    )


def _trunk_output_dim(img_size):
    return 64 * (img_size // 8) ** 2


class BaselineImageModel(nn.Module):
    def __init__(self, num_classes, img_size=80):
        super().__init__()
        self.image_layer = _build_image_trunk(img_size)
        flat_dim = _trunk_output_dim(img_size)
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_image, x_attr=None):
        # x_attr accepted (and ignored) so train/eval loops can call both
        # models with the same signature.
        return self.classifier(self.image_layer(x_image))


class MultiModalProductClassifier(nn.Module):
    def __init__(self, num_classes, attr_dim, img_size=80):
        super().__init__()
        self.image_layer = _build_image_trunk(img_size)
        flat_dim = _trunk_output_dim(img_size)
        self.attr_layer = nn.Sequential(
            nn.Linear(attr_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_image, x_attr):
        x_image = self.image_layer(x_image)
        x_attr = self.attr_layer(x_attr)
        x = torch.cat((x_image, x_attr), dim=1)
        return self.classifier(x)


def build_model(name, num_classes, attr_dim, img_size=80):
    if name == "baseline":
        return BaselineImageModel(num_classes=num_classes, img_size=img_size)
    if name == "proposed":
        return MultiModalProductClassifier(num_classes=num_classes, attr_dim=attr_dim, img_size=img_size)
    raise ValueError(f"Unknown model name: {name}")
