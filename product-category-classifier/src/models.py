"""Model architectures for the product category classifier.

BaselineImageModel: image-only CNN.
MultiModalProductClassifier: image + gender, a near 1:1 translation of the
multi-input pattern (image_layer + type_layer -> concat -> classifier) from
the original cert exercise this case study is based on.
"""
import torch
import torch.nn as nn


class BaselineImageModel(nn.Module):
    def __init__(self, num_classes, img_size=80):
        super().__init__()
        self.image_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat_dim = 32 * (img_size // 4) * (img_size // 4)
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_image, x_gender=None):
        # x_gender accepted (and ignored) so train/eval loops can call both
        # models with the same signature.
        return self.classifier(self.image_layer(x_image))


class MultiModalProductClassifier(nn.Module):
    def __init__(self, num_classes, num_genders, img_size=80):
        super().__init__()
        self.image_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (img_size // 4) * (img_size // 4), 128),
        )
        self.gender_layer = nn.Sequential(
            nn.Linear(num_genders, 10),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 10, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_image, x_gender):
        x_image = self.image_layer(x_image)
        x_gender = self.gender_layer(x_gender)
        x = torch.cat((x_image, x_gender), dim=1)
        return self.classifier(x)


def build_model(name, num_classes, num_genders, img_size=80):
    if name == "baseline":
        return BaselineImageModel(num_classes=num_classes, img_size=img_size)
    if name == "proposed":
        return MultiModalProductClassifier(
            num_classes=num_classes, num_genders=num_genders, img_size=img_size
        )
    raise ValueError(f"Unknown model name: {name}")
