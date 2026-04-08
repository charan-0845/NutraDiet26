from __future__ import annotations

from functools import lru_cache

import cv2
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLIP_MODEL_PATH
from modules.yolo_segment import get_class_names


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_clip_model():
    checkpoint = torch.load(str(CLIP_MODEL_PATH), map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    class_names = get_class_names()

    model, _, preprocess = open_clip.create_model_and_transforms(
        config["model"],
        pretrained=config["pretrained"],
    )
    tokenizer = open_clip.get_tokenizer(config["model"])

    model.load_state_dict(checkpoint["model_state"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    text_tokens = tokenizer([f"a photo of {name}" for name in class_names]).to(DEVICE)
    with torch.no_grad():
        text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

    return model, preprocess, class_names, text_features


def _to_pil(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


@torch.no_grad()
def predict_food(image):
    model, preprocess, class_names, text_features = load_clip_model()
    image_tensor = preprocess(_to_pil(image)).unsqueeze(0).to(DEVICE)
    image_features = F.normalize(model.encode_image(image_tensor), dim=-1)
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    confidence, index = similarities[0].max(dim=0)

    return class_names[int(index.item())], float(confidence.item())
