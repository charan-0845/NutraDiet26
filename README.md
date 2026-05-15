# NutraDiet26

AI-powered meal nutrition analyzer built for Indian cuisine. Upload a photo of your meal or describe what you ate — the system detects every food item, estimates portion sizes, and returns a breakdown of 27 nutrients per item.

---

## How It Works

NutraDiet26 runs a four-stage pipeline on every request:

1. **Vision (Moondream2)** — The uploaded image is decoded and resized via OpenCV, then passed to Moondream2, a lightweight vision language model that runs fully locally. It generates structured per-item descriptions with portion estimates (e.g. "2 pieces of idli", "1 small bowl of sambar").

2. **BERT Semantic Matching** — Each food token from text input and/or Moondream output is encoded with `all-MiniLM-L6-v2` and matched against 452+ canonical food labels using cosine similarity. A curated `food_mapping.csv` with 1,000+ entries resolves regional aliases, Hindi transliterations, and variant names to canonical labels.

3. **Text-Image Fusion** — Both sources are merged with a priority rule: user-typed descriptions always win for portion sizes. If a food appears in text without a weight, Moondream's portion estimate is adopted. Items detected only by vision are appended to the result. Duplicates are removed by canonical label.

4. **Nutrition Retrieval** — Matched labels are looked up in INDB first (500+ Indian dishes), falling back to FNDDS (5,600+ USDA items). Retrieval runs three strategies in order: exact name match → token-subset match with SequenceMatcher ranking → fuzzy difflib match. All 27 nutrient values are scaled proportionally to the estimated portion weight.

---

## Features

- Offline inference — no API calls, no per-query cost
- Supports image input, text input, or both simultaneously
- 27 nutrients per item: energy, protein, carbs, fat, fiber, sugars, saturated/mono/polyunsaturated fat, cholesterol, calcium, phosphorus, magnesium, sodium, potassium, iron, copper, selenium, zinc, vitamins A/C/E, thiamin, riboflavin, niacin, B6, folate
- Interactive UI — edit food names (triggers live re-lookup), adjust weights (proportional macro scaling), remove items
- Human-in-the-loop correction for any detection errors
- Source tagging on each item: `text`, `image`, or `text+image`

---

## Project Structure

```
nutradiet26/
├── app.py                      # FastAPI app — /predict, /nutrition endpoints
├── config.py                   # Paths and constants
├── requirements.txt
│
├── modules/
│   ├── moondream_infer.py      # Moondream2 wrapper (lazy load, thread-safe)
│   ├── nutrition.py            # Nutrition lookup — INDB + FNDDS, 27 nutrients
│   └── preprocess.py           # OpenCV image resize
│
├── llm/
│   ├── parser.py               # Entry point: parse_food_items()
│   ├── merged_food_parser.py   # Text-image fusion logic
│   ├── food_parser_bert.py     # BERT matching, quantity extraction, unit normalization
│   ├── clean_labels.txt        # 452+ canonical food labels
│   └── food_mapping.csv        # 1,000+ alias → canonical mappings
│
├── data/
│   ├── INDB.xlsx               # Indian Nutrition Database (primary)
│   └── FNDDS.xlsx              # USDA FNDDS 2019–2020 (fallback)
│
├── models/
│   └── (model weights — gitignored)
│
├── templates/
│   └── index.html              # Single-file web UI
│
├── static/
│   ├── app.js
│   └── styles.css
│
├── test_end_to_end.py          # Full pipeline test suite (20 scenarios)
└── test_nutrition_api.py       # Nutrient coverage verification
```

---

## Setup

**Requirements:** Python 3.10+, pip

```bash
# Clone and install
git clone https://github.com/yourname/nutradiet26.git
cd nutradiet26
pip install -r requirements.txt
```

**Download models:**

Moondream2 must be downloaded from HuggingFace and placed in the `models/` directory (or cached via `transformers`):

```bash
# Using huggingface-cli
huggingface-cli download vikhyatk/moondream2 --local-dir models/moondream2
```

**Add data files:**

Place `INDB.xlsx` and `FNDDS.xlsx` in the `data/` directory. These are not included in the repo.

**Run:**

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

---

## API

### `POST /predict`
Accepts a multipart form with an image file and optional text description. Returns detected food items with full nutrition data.

| Field | Type | Required |
|-------|------|----------|
| `file` | image (JPG/PNG/WEBP) | Yes |
| `text` | string | No |
| `weight` | float (default 100g) | No |

**Response:**
```json
{
  "results": [
    {
      "food": "idli",
      "weight": 120.0,
      "source": "text+image",
      "nutrients": {
        "matched_name": "idli",
        "source": "INDB",
        "energy_kcal": 134.4,
        "protein": 4.1,
        "carbs": 26.2,
        "fat": 0.7,
        "calories": 134.4,
        ...
      }
    }
  ]
}
```

### `POST /nutrition`
Re-fetches nutrition for a single food item by name and weight. Used by the UI when a user edits a food name.

| Field | Type |
|-------|------|
| `food` | string |
| `weight` | float |

---

## Running Tests

```bash
# Full test suite (20 scenarios across text, vision, merged, and edge cases)
python test_end_to_end.py

# Quick mode (text-only + edge cases)
python test_end_to_end.py --quick

# Verbose output (show all detected items and DB matches)
python test_end_to_end.py --verbose

# Run a specific category
python test_end_to_end.py --category merged

# Verify 27-nutrient coverage
python test_nutrition_api.py
```

Test results are saved to `e2e_report.json` after each run.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Backend | FastAPI, Uvicorn |
| Vision model | Moondream2 (vikhyatk/moondream2) |
| Semantic matching | Sentence-Transformers (all-MiniLM-L6-v2) |
| Image processing | OpenCV, Pillow |
| Data | Pandas, openpyxl |
| ML runtime | PyTorch |
| Frontend | Vanilla JS, HTML/CSS |

---

## Notes

- `*.pt`, `*.bin`, `*.pkl`, and `*.h5` files are gitignored — model weights must be downloaded separately.
- INDB stores saturated/mono/polyunsaturated fat values in mg; the nutrition module converts these to grams automatically for consistent output.
- Moondream2 is loaded once per process with a thread-safe lock and runs in `float16` on CUDA or `float32` on CPU.
- The BERT label embeddings are precomputed and cached at startup to keep inference fast.
