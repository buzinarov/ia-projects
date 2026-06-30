"""Shared inference helpers, used by evaluate.py, the Reflex app, the
notebooks, and the tool-calling agent. Single source of truth for how
we turn an image + structured attributes into a prediction.
"""
from pathlib import Path

import numpy as np
import torch

from .contract import build_prediction_record
from .data import IMG_SIZE, load_label_maps
from .models import build_model

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "artifacts" / "checkpoints"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name, seed=0, device=None):
    device = device or get_device()
    maps = load_label_maps()
    num_classes = len(maps["target_classes"])
    model = build_model(num_classes=num_classes, img_size=IMG_SIZE)
    state = torch.load(CHECKPOINT_DIR / f"{name}_seed{seed}.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, maps, device


def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def attrs_to_onehot(attrs, maps):
    """attrs: dict mapping each of maps['attribute_columns'] to a value.
    Builds the same concatenated one-hot vector, in the same column
    order, that FashionProductDataset builds at training time."""
    onehots = []
    for col in maps["attribute_columns"]:
        classes = maps["attribute_classes"][col]
        value = attrs[col]
        if value not in classes:
            raise ValueError(f"Unknown value {value!r} for attribute {col!r}; expected one of {classes}")
        onehot = torch.zeros(1, len(classes), dtype=torch.float32)
        onehot[0, classes.index(value)] = 1.0
        onehots.append(onehot)
    return torch.cat(onehots, dim=1)


@torch.no_grad()
def predict(model, maps, device, pil_image, attrs=None):
    """attrs is optional: the image classifier ignores the attribute
    vector entirely (see models.ImageClassifier.forward), so a caller
    with only a photo -- no known catalog attributes -- still gets a
    prediction. When attrs is provided it is still validated and
    encoded, so callers that do have catalog metadata keep it attached
    to the contract record."""
    image_tensor = preprocess_image(pil_image).to(device)
    if attrs:
        attr_tensor = attrs_to_onehot(attrs, maps).to(device)
    else:
        attr_tensor = torch.zeros(1, maps["attribute_dim"], device=device)
    logits = model(image_tensor, attr_tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = maps["target_classes"]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }


def predict_with_contract(model, maps, device, pil_image, *, attrs=None, model_name, product_id=None):
    """Same as predict(), but returns a contract-compliant record
    (src/contract.py) instead of the raw probabilities dict -- this is
    what should feed any downstream consumer (the agent's tool output,
    a future batch-scoring pipeline), since 'probabilities' isn't part
    of the data contract. attrs is optional, same as predict()."""
    result = predict(model, maps, device, pil_image, attrs)
    return build_prediction_record(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        model_name=model_name,
        product_id=product_id,
    )
