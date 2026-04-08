import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"


# =========================
# SAFE JSON PARSER
# =========================
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except:
                pass

    print("⚠️ JSON parsing failed. Raw output:")
    print(text)
    return {"food_items": []}


# =========================
# WORD → NUMBER
# =========================
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


# =========================
# CLEAN FOOD NAME
# =========================
def clean_food_name(name):
    if not name:
        return None

    name = name.lower().strip()

    remove_words = ["spicy", "tasty", "delicious", "hot"]
    for word in remove_words:
        name = name.replace(word, "").strip()

    return name


# =========================
# SINGULAR NORMALIZATION
# =========================
def normalize_food_name(name):
    if not name:
        return None

    if name.endswith("s") and len(name) > 3:
        if name not in ["peas"]:
            name = name[:-1]

    return name


# =========================
# UNIT NORMALIZATION
# =========================
def normalize_quantity(quantity, unit):
    if quantity is None:
        return None

    unit = unit.lower() if unit else ""

    # weight
    if unit in ["g", "gram", "grams"]:
        return quantity

    elif unit in ["kg", "kilogram"]:
        return quantity * 1000

    # liquids
    elif unit in ["ml"]:
        return quantity

    elif unit in ["liter", "litre"]:
        return quantity * 1000

    # cups
    elif "cup" in unit:
        if "small" in unit:
            return quantity * 150
        elif "large" in unit:
            return quantity * 250
        return quantity * 200

    # bowls
    elif "bowl" in unit:
        if "small" in unit:
            return quantity * 150
        elif "large" in unit:
            return quantity * 300
        return quantity * 250

    # plates
    elif "plate" in unit:
        return quantity * 350

    # spoons
    elif unit in ["tablespoon"]:
        return quantity * 15
    elif unit in ["teaspoon"]:
        return quantity * 5

    # pieces
    elif unit in ["piece", "pieces"]:
        return quantity * 50

    return quantity


# =========================
# FIX QUANTITY ASSIGNMENT
# =========================
def fix_quantities(items):
    if not items:
        return items

    items_with_qty = [i for i in items if i["quantity_grams"] is not None]

    if len(items) > 1 and len(items_with_qty) == 1:
        qty = items_with_qty[0]["quantity_grams"]

        for item in items:
            item["quantity_grams"] = None

        items[-1]["quantity_grams"] = qty

    return items


# =========================
# LLM CALL
# =========================
def extract_food_data_local(text: str):
    prompt = f"""
Extract food items and their quantities from the text.

STRICT RULES:
- Return ONLY valid JSON
- DO NOT generalize food names (keep full names like "lemon rice", "veg biryani")
- Remove only adjectives like spicy, tasty
- If multiple foods, return all

QUANTITY RULES:
- Convert words to numbers:
    half → 0.5
    quarter → 0.25
    one → 1
    two → 2
    three → 3
    one and half → 1.5
- If no quantity → null

UNIT RULES:
Recognize:
- grams, g, kg
- ml, liter
- cup, small cup, large cup
- bowl, small bowl, large bowl
- plate
- piece, pieces
- tablespoon, teaspoon

FORMAT:
{{
  "food_items": [
    {{
      "name": "string",
      "quantity": number or null,
      "unit": "string or null"
    }}
  ]
}}

STRICT:
- No explanation
- No extra text
- Only JSON

Text:
{text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m"
        }
    )

    result = response.json()["response"]

    return safe_json_parse(result)


# =========================
# MAIN FUNCTION (EXPORT)
# =========================
def parse_food_items_local(text: str):
    if not text or not text.strip():
        return []

    text = word_to_number(text.lower())

    data = extract_food_data_local(text)

    parsed_items = []

    for item in data.get("food_items", []):
        raw_name = item.get("name")
        quantity = item.get("quantity")
        unit = item.get("unit")

        name = normalize_food_name(clean_food_name(raw_name))
        normalized_qty = normalize_quantity(quantity, unit)

        parsed_items.append({
            "name": name,
            "quantity_grams": normalized_qty
        })

    parsed_items = fix_quantities(parsed_items)

    return parsed_items
def parse_food_items(text: str):
    return parse_food_items_local(text)   # ✅ correct