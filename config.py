from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

YOLO_MODEL_PATH = MODELS_DIR / "yolo_best.pt"
CLIP_MODEL_PATH = MODELS_DIR / "clip_best.pt"
FNDDS_PATH = DATA_DIR / "FNDDS.xlsx"
INDB_PATH = DATA_DIR / "INDB.xlsx"

FUZZY_MATCH_THRESHOLD = 70
