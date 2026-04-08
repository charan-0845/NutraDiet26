from __future__ import annotations

from functools import lru_cache

from ultralytics import YOLO

from config import YOLO_MODEL_PATH


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    return YOLO(str(YOLO_MODEL_PATH))


def get_class_names() -> list[str]:
    model = get_model()
    return [model.names[index] for index in sorted(model.names)]


def segment_food(image):
    model = get_model()
    results = model(image, verbose=False)
    result = results[0]

    crops = []
    boxes = []
    labels = []
    confidences = []

    if result.boxes is None:
        return crops, boxes, labels, confidences

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        crops.append(image[y1:y2, x1:x2])
        boxes.append((x1, y1, x2, y2))
        labels.append(result.names[class_id])
        confidences.append(confidence)

    return crops, boxes, labels, confidences
