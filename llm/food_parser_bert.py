import re
import json
import os
import pandas as pd
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# LOAD LABELS + MAPPING
# ─────────────────────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LABELS_FILE = BASE_DIR / "clean_labels.txt"
MAPPING_FILE = BASE_DIR / "food_mapping.csv"

FOOD_LABELS = [line.strip() for line in LABELS_FILE.read_text(encoding="utf-8").splitlines()]

mapping_df = pd.read_csv(MAPPING_FILE)
MAPPING_DICT = dict(zip(mapping_df["original"], mapping_df["canonical"]))

print(f"Loaded {len(FOOD_LABELS)} labels")

# ─────────────────────────────────────────────
# LOAD BERT
# ─────────────────────────────────────────────

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
LABEL_EMBEDDINGS = MODEL.encode(FOOD_LABELS, convert_to_tensor=True)

# ─────────────────────────────────────────────
# WORD → NUMBER (FROM YOUR CODE)
# ─────────────────────────────────────────────

def word_to_number(text):
    mapping = {
        "half": 0.5,
        "quarter": 0.25,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5
    }

    for word, num in mapping.items():
        text = text.replace(word, str(num))

    text = text.replace("1 and 0.5", "1.5")
    return text

# ─────────────────────────────────────────────
# CLEAN FOOD NAME (FROM YOUR CODE)
# ─────────────────────────────────────────────

def clean_food_name(name):
    if not name:
        return None

    name = name.lower().strip()

    remove_words = ["spicy", "tasty", "delicious", "hot"]
    for word in remove_words:
        name = name.replace(word, "")

    return name.strip()

# ─────────────────────────────────────────────
# NORMALIZE QUANTITY (FROM YOUR CODE)
# ─────────────────────────────────────────────

def normalize_quantity(quantity, unit):
    if quantity is None:
        return None

    unit = unit.lower() if unit else ""

    if unit in ["g", "gram", "grams"]:
        return quantity
    elif unit in ["kg"]:
        return quantity * 1000

    elif unit in ["ml"]:
        return quantity
    elif unit in ["liter", "litre"]:
        return quantity * 1000

    elif "cup" in unit:
        return quantity * 200
    elif "bowl" in unit:
        return quantity * 250
    elif "plate" in unit:
        return quantity * 350

    elif unit in ["tablespoon"]:
        return quantity * 15
    elif unit in ["teaspoon"]:
        return quantity * 5

    elif unit in ["piece", "pieces"]:
        return quantity * 50

    return quantity

# ─────────────────────────────────────────────
# FIX QUANTITY BUG (FROM YOUR CODE)
# ─────────────────────────────────────────────

def fix_quantities(items):
    items_with_qty = [i for i in items if i["quantity_grams"] is not None]

    if len(items) > 1 and len(items_with_qty) == 1:
        qty = items_with_qty[0]["quantity_grams"]

        for item in items:
            item["quantity_grams"] = None

        items[-1]["quantity_grams"] = qty

    return items

# ─────────────────────────────────────────────
# SPLIT ITEMS
# ─────────────────────────────────────────────

SPLIT_RE = re.compile(r"\s+(and|with|,)\s+")

def split_items(text):
    return [t.strip() for t in SPLIT_RE.split(text) if t not in {"and", "with", ","}]

# ─────────────────────────────────────────────
# EXTRACT QUANTITY + UNIT (REGEX)
# ─────────────────────────────────────────────

QTY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(grams?|kg|ml|cups?|bowls?|plates?|pieces?)")

def extract_quantity(text):
    match = QTY_RE.search(text)
    if match:
        qty = float(match.group(1))
        unit = match.group(2).rstrip("s")
        clean = text.replace(match.group(0), "")
        return qty, unit, clean
    return None, None, text

# ─────────────────────────────────────────────
# COUNT-BASED FOODS
# ─────────────────────────────────────────────

COUNT_FOODS = {"chapati", "roti", "idli", "egg"}

def detect_count(text):
    match = re.search(r"(\d+(?:\.\d+)?)\s+([a-zA-Z]+)", text)
    if match:
        qty = float(match.group(1))
        food = match.group(2).lower()

        # normalize plural → singular
        if food.endswith("s"):
            food = food[:-1]

        return qty, "piece", food

    return None, None, text
# ─────────────────────────────────────────────
# BERT MATCH
# ─────────────────────────────────────────────

def match_label(text, threshold=0.6):
    emb = MODEL.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(emb, LABEL_EMBEDDINGS)[0]

    idx = int(scores.argmax())
    score = float(scores[idx])

    if score < threshold:
        return text, score

    return FOOD_LABELS[idx], score

# ─────────────────────────────────────────────
# NORMALIZE LABEL
# ─────────────────────────────────────────────

def normalize_label(label):
    return MAPPING_DICT.get(label, label)

# ─────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────

def parse_food(text):

    text = word_to_number(text.lower())
    segments = split_items(text)

    results = []

    for seg in segments:

        qty, unit, food_text = extract_quantity(seg)

        if qty is None:
            qty, unit, food_text = detect_count(seg)

        cleaned = clean_food_name(food_text)

        if not cleaned:
            continue

        matched, conf = match_label(cleaned)
        normalized = normalize_label(matched)
        DEFAULT_GRAMS = {
            "chapati": 50,
            "roti": 50,
            "idli": 60,
            "egg": 50
        }
        
        if qty is None:
            grams = 100
        elif unit == "piece":
            base = DEFAULT_GRAMS.get(normalized, 50)
            grams = qty * base
        else:
            grams = normalize_quantity(qty, unit)
        results.append({
            "name": normalized,
            "raw_input": seg,
            "quantity": qty,
            "unit": unit,
            "quantity_grams": grams,
            "confidence": round(conf, 3)
        })

    results = fix_quantities(results)

    return results

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n🔥 FINAL OFFLINE FOOD PARSER\n")

    while True:
        user = input("Enter food: ").strip()
        if user.lower() in ("exit", "quit"):
            break

        output = parse_food(user)

        print("\nParsed Output:")
        print(json.dumps(output, indent=2))
        print()
