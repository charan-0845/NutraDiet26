from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

from modules.clip_infer import predict_food
from modules.nutrition import get_nutrition
from modules.preprocess import preprocess_image
from modules.yolo_segment import segment_food

# ✅ NEW: import parser
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

        # TEXT PARSING
        text_items = []
        if text:
            try:
                text_items = parse_food_items(text)
            except Exception as e:
                print("LLM ERROR:", e)
                text_items = []

        results = []

        # TEXT PRIORITY
        if text_items:
            for item in text_items:
                label = item["name"]

                item_weight = (
                    item["quantity_grams"]
                    if item["quantity_grams"] is not None
                    else weight if weight is not None
                    else 100
                )

                nutrients = get_nutrition(label, item_weight)

                results.append(
                    {
                        "food": label,
                        "confidence": 1.0,
                        "weight": item_weight,
                        "nutrients": nutrients,
                        "source": "text",
                    }
                )

            return {"results": results}

        # IMAGE PIPELINE
        if mode == "single":
            label, conf = predict_food(image)

            item_weight = weight if weight is not None else 100

            nutrients = get_nutrition(label, item_weight)

            results.append(
                {
                    "food": label,
                    "confidence": conf,
                    "weight": item_weight,
                    "nutrients": nutrients,
                    "source": "image",
                }
            )

        elif mode == "mixed":
            crops, boxes, _labels, _confidences = segment_food(image)

            if not crops:
                label, conf = predict_food(image)

                item_weight = weight if weight is not None else 100

                nutrients = get_nutrition(label, item_weight)

                results.append(
                    {
                        "food": label,
                        "confidence": conf,
                        "weight": item_weight,
                        "nutrients": nutrients,
                        "source": "image",
                    }
                )
            else:
                areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
                total_area = sum(areas)

                for i, _crop in enumerate(crops):
                    label, conf = predict_food(crops[i])

                    ratio = (areas[i] / total_area) if total_area else (1.0 / len(crops))
                    item_weight = (weight if weight is not None else 100) * ratio

                    nutrients = get_nutrition(label, item_weight)

                    results.append(
                        {
                            "food": label,
                            "confidence": conf,
                            "weight": item_weight,
                            "nutrients": nutrients,
                            "source": "image",
                        }
                    )

        else:
            return {"results": [], "error": "mode must be 'single' or 'mixed'"}

        return {"results": results}

    except Exception as e:
        print("SERVER ERROR:", e)
        return {"results": [], "error": str(e)}