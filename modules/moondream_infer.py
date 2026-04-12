"""
modules/moondream_infer.py

Queries Moondream2 and returns its raw text output.
All food parsing is handled downstream by the BERT parser (llm/parser.py).
"""

from __future__ import annotations

import threading
import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("USE_TF", "0")

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model loading — lazy, thread-safe, loaded once per process
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_model: Optional[AutoModelForCausalLM] = None
_tokenizer = None

_MODEL_ID = "vikhyatk/moondream2"

_PROMPT = (
    "Look at the image and list every distinct food item you can see. "
    "For each food item, write exactly one sentence in this format: "
    "'I had [quantity + unit] of [food name].' "
    "Use realistic portion estimates like: '2 pieces of', '1 small bowl of', "
    "'300g of', '1 cup of', '1 full plate of'. "
    "Base your quantity estimate on the size of the container or visual portion size. "
    "Do not repeat any food item. "
    "Do not include garnishes, herbs, or spices as separate items. "
    "Do not add any explanation, intro, or extra words — only the sentences."
)


def _load_model() -> AutoModelForCausalLM:
    global _model, _tokenizer
    with _lock:
        if _model is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            _tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_ID,
                trust_remote_code=True,
                local_files_only=True,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                _MODEL_ID,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=True,
            )
            _model.eval()
            if torch.cuda.is_available():
                _model.cuda()
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_moondream(image: np.ndarray) -> str:
    """
    Run Moondream2 on a BGR numpy image and return its raw text output.

    Parameters
    ----------
    image : np.ndarray
        BGR image from cv2 / preprocess_image.

    Returns
    -------
    str
        Raw Moondream output, e.g.:
        "I had 1 bowl of dal makhani. I had 2 pieces of naan."
        Empty string if the model returns nothing.
    """
    model = _load_model()
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        output = model.query(pil_image, _PROMPT)

    if isinstance(output, dict):
        output = output.get("answer", "") or ""

    return output.strip()
