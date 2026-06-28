"""Model architectures for the recommender's image signal.

BaselineImageModel: image-only CNN -- the variant that ships, whose
predicted subcategory feeds the recommender.
MultiModalProductClassifier: the same image trunk plus a small branch
for the structured attributes (gender, baseColour, season, usage),
concatenated before the classifier head -- kept for the classification
appendix.

Both share one image trunk by construction, so the only difference
between them is whether the attribute branch exists.
"""
import torch
import torch.nn as nn


def _build_image_trunk():
    """Shared CNN trunk, used unchanged by both models so a fair
    comparison is enforced by code rather than by convention."""
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
        self.image_layer = _build_image_trunk()
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
        self.image_layer = _build_image_trunk()
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
