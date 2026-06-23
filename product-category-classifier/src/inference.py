"""Shared inference helpers, used by the notebook, evaluate.py, the
Streamlit app, and the tool-calling agent. Single source of truth for
"how do we turn an image + gender into a prediction."
"""
from pathlib import Path

import numpy as np
import torch

from .data import IMG_SIZE, load_label_maps
from .models import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "artifacts" / "checkpoints"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name, device=None):
    device = device or get_device()
    maps = load_label_maps()
    num_classes = len(maps["masterCategory_classes"])
    num_genders = len(maps["gender_classes"])
    model = build_model(name, num_classes=num_classes, num_genders=num_genders, img_size=IMG_SIZE)
    state = torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, maps, device


def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def gender_to_onehot(gender, gender_classes):
    idx = gender_classes.index(gender)
    onehot = torch.zeros(1, len(gender_classes), dtype=torch.float32)
    onehot[0, idx] = 1.0
    return onehot


@torch.no_grad()
def predict(model, maps, device, pil_image, gender):
    image_tensor = preprocess_image(pil_image).to(device)
    gender_tensor = gender_to_onehot(gender, maps["gender_classes"]).to(device)
    logits = model(image_tensor, gender_tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = maps["masterCategory_classes"]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }
