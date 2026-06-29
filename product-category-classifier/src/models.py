"""Model architecture for the recommender's image signal.

`ImageClassifier` is a small image-only CNN that predicts a product's
subcategory from its 80x80 photo. That prediction is the category signal
the recommender uses to keep suggestions on-category.
"""
import torch.nn as nn


def _build_image_trunk():
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


class ImageClassifier(nn.Module):
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
        # x_attr is accepted (and ignored) so callers that still pass the
        # catalog's structured attributes work without a special case.
        return self.classifier(self.image_layer(x_image))


def build_model(num_classes, img_size=80):
    return ImageClassifier(num_classes=num_classes, img_size=img_size)
