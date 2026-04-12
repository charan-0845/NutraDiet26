from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

from modules.moondream_infer import query_moondream
from modules.nutrition import get_nutrition
from modules.preprocess import preprocess_image
from modules.yolo_segment import segment_food

# BERT-based parser — merges user text + moondream text internally
from llm.parser import parse_food_items

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/nutrition")
async def nutrition_lookup(food: str = Form(...), weight: float = Form(...)):
    """Re-fetch nutrition when the user edits a food name or weight."""
    try:
        nutrients = get_nutrition(food, weight)
        return {"food": food, "weight": weight, "nutrients": nutrients}
    except Exception as e:
        return {"error": str(e)}


def _build_results(food_items: list[dict], default_weight: float) -> list[dict]:
    """Turn parsed food items into final result records with nutrition data."""
    results = []
    for item in food_items:
        label = item["name"]
        item_weight = (
            item["quantity_grams"]
            if item["quantity_grams"] is not None
            else default_weight
        )
        nutrients = get_nutrition(label, item_weight)
        results.append(
            {
                "food": label,
                "weight": item_weight,
                "nutrients": nutrients,
                "source": item.get("source", "unknown"),
            }
        )
    return results


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form(...),
    weight: float = Form(None),
    text: str = Form(None),
):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            return {"results": [], "error": "Invalid image"}

        image = preprocess_image(image)
        default_weight = weight if weight is not None else 100.0

        # ------------------------------------------------------------------
        # IMAGE PIPELINE — always runs regardless of whether text is present
        # ------------------------------------------------------------------
        moondream_text = ""

        if mode == "single":
            # Query the whole image once
            moondream_text = query_moondream(image)

        elif mode == "mixed":
            crops, boxes, _labels, _confidences = segment_food(image)

            if not crops:
                # No segments — fall back to whole-image query
                moondream_text = query_moondream(image)
            else:
                # Query each YOLO crop and concatenate the outputs.
                # The merged parser handles deduplication, so repeats are fine.
                parts = []
                for crop in crops:
                    crop_text = query_moondream(crop)
                    if crop_text:
                        parts.append(crop_text)
                moondream_text = " ".join(parts)

        else:
            return {"results": [], "error": "mode must be 'single' or 'mixed'"}

        # ------------------------------------------------------------------
        # MERGE — BERT parser combines user text + moondream output.
        # Text portions take priority; moondream fills in what text missed.
        # ------------------------------------------------------------------
        food_items = parse_food_items(
            text=text or None,
            moondream_text=moondream_text or None,
        )

        if not food_items:
            return {"results": [], "error": "Could not identify any food items."}

        return {"results": _build_results(food_items, default_weight)}

    except Exception as e:
        print("SERVER ERROR:", e)
        return {"results": [], "error": str(e)}
