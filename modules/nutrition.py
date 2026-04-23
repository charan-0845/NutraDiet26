from __future__ import annotations

import difflib
import re
from functools import lru_cache

import pandas as pd

from config import FNDDS_PATH, INDB_PATH


def _normalize_name(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def _tokenize(value: str) -> set[str]:
    return set(_normalize_name(value).split())


# Mapping of FNDDS columns to canonical nutrient names
NUTRIENT_COLUMNS = {
    "energy_kcal": {
        "fndds": "Energy (kcal)",
        "indb": "energy_kcal",
        "unit": "kcal",
        "display_name": "Energy (kcal)"
    },
    "protein": {
        "fndds": "Protein (g)",
        "indb": "protein_g",
        "unit": "g",
        "display_name": "Protein (g)"
    },
    "carbs": {
        "fndds": "Carbohydrate (g)",
        "indb": "carb_g",
        "unit": "g",
        "display_name": "Carbohydrate (g)"
    },
    "fat": {
        "fndds": "Total Fat (g)",
        "indb": "fat_g",
        "unit": "g",
        "display_name": "Total Fat (g)"
    },
    "fiber": {
        "fndds": "Fiber, total dietary (g)",
        "indb": "fibre_g",
        "unit": "g",
        "display_name": "Fiber (g)"
    },
    "sugars": {
        "fndds": "Sugars, total\n(g)",
        "indb": "freesugar_g",
        "unit": "g",
        "display_name": "Sugars (g)"
    },
    "saturated_fat": {
        "fndds": "Fatty acids, total saturated (g)",
        "indb": "sfa_mg",
        "unit": "g",  # INDB stores as mg
        "display_name": "Saturated Fat (g)"
    },
    "monounsaturated_fat": {
        "fndds": "Fatty acids, total monounsaturated (g)",
        "indb": "mufa_mg",
        "unit": "g",
        "display_name": "Monounsaturated Fat (g)"
    },
    "polyunsaturated_fat": {
        "fndds": "Fatty acids, total polyunsaturated (g)",
        "indb": "pufa_mg",
        "unit": "g",
        "display_name": "Polyunsaturated Fat (g)"
    },
    "cholesterol": {
        "fndds": "Cholesterol (mg)",
        "indb": "cholesterol_mg",
        "unit": "mg",
        "display_name": "Cholesterol (mg)"
    },
    "calcium": {
        "fndds": "Calcium (mg)",
        "indb": "calcium_mg",
        "unit": "mg",
        "display_name": "Calcium (mg)"
    },
    "phosphorus": {
        "fndds": "Phosphorus (mg)",
        "indb": "phosphorus_mg",
        "unit": "mg",
        "display_name": "Phosphorus (mg)"
    },
    "magnesium": {
        "fndds": "Magnesium (mg)",
        "indb": "magnesium_mg",
        "unit": "mg",
        "display_name": "Magnesium (mg)"
    },
    "sodium": {
        "fndds": "Sodium (mg)",
        "indb": "sodium_mg",
        "unit": "mg",
        "display_name": "Sodium (mg)"
    },
    "potassium": {
        "fndds": "Potassium (mg)",
        "indb": "potassium_mg",
        "unit": "mg",
        "display_name": "Potassium (mg)"
    },
    "iron": {
        "fndds": "Iron\n(mg)",
        "indb": "iron_mg",
        "unit": "mg",
        "display_name": "Iron (mg)"
    },
    "copper": {
        "fndds": "Copper (mg)",
        "indb": "copper_mg",
        "unit": "mg",
        "display_name": "Copper (mg)"
    },
    "selenium": {
        "fndds": "Selenium (mcg)",
        "indb": "selenium_ug",
        "unit": "mcg",
        "display_name": "Selenium (mcg)"
    },
    "zinc": {
        "fndds": "Zinc\n(mg)",
        "indb": "zinc_mg",
        "unit": "mg",
        "display_name": "Zinc (mg)"
    },
    "vitamin_c": {
        "fndds": "Vitamin C (mg)",
        "indb": "vitc_mg",
        "unit": "mg",
        "display_name": "Vitamin C (mg)"
    },
    "vitamin_e": {
        "fndds": "Vitamin E (alpha-tocopherol) (mg)",
        "indb": "vite_mg",
        "unit": "mg",
        "display_name": "Vitamin E (mg)"
    },
    "thiamin": {
        "fndds": "Thiamin (mg)",
        "indb": "vitb1_mg",
        "unit": "mg",
        "display_name": "Thiamin/B1 (mg)"
    },
    "riboflavin": {
        "fndds": "Riboflavin (mg)",
        "indb": "vitb2_mg",
        "unit": "mg",
        "display_name": "Riboflavin/B2 (mg)"
    },
    "niacin": {
        "fndds": "Niacin (mg)",
        "indb": "vitb3_mg",
        "unit": "mg",
        "display_name": "Niacin/B3 (mg)"
    },
    "vitamin_b6": {
        "fndds": "Vitamin B-6 (mg)",
        "indb": "vitb6_mg",
        "unit": "mg",
        "display_name": "Vitamin B6 (mg)"
    },
    "folate": {
        "fndds": "Folate, total (mcg)",
        "indb": "folate_ug",
        "unit": "mcg",
        "display_name": "Folate (mcg)"
    },
    "vitamin_a": {
        "fndds": "Vitamin A, RAE (mcg_RAE)",
        "indb": "vita_ug",
        "unit": "mcg",
        "display_name": "Vitamin A (mcg)"
    },
}


def _load_fndds() -> pd.DataFrame:
    raw = pd.read_excel(FNDDS_PATH, header=1)
    
    # Select all nutrient columns
    columns_to_keep = ["Main food description"] + [config["fndds"] for config in NUTRIENT_COLUMNS.values()]
    columns_available = [col for col in columns_to_keep if col in raw.columns]
    data = raw[columns_available].copy()
    
    # Rename to canonical names
    rename_map = {"Main food description": "name"}
    for nutrient_key, config in NUTRIENT_COLUMNS.items():
        if config["fndds"] in data.columns:
            rename_map[config["fndds"]] = nutrient_key
    
    data = data.rename(columns=rename_map)
    data = data.dropna(subset=["name"])
    data["source"] = "FNDDS"
    return data


def _load_indb() -> pd.DataFrame:
    raw = pd.read_excel(INDB_PATH)
    
    # Select all nutrient columns
    columns_to_keep = ["food_name"] + [config["indb"] for config in NUTRIENT_COLUMNS.values()]
    columns_available = [col for col in columns_to_keep if col in raw.columns]
    data = raw[columns_available].copy()
    
    # Rename to canonical names
    rename_map = {"food_name": "name"}
    for nutrient_key, config in NUTRIENT_COLUMNS.items():
        if config["indb"] in data.columns:
            rename_map[config["indb"]] = nutrient_key
    
    data = data.rename(columns=rename_map)
    data = data.dropna(subset=["name"])
    data["source"] = "INDB"
    return data


@lru_cache(maxsize=1)
def _nutrition_df() -> pd.DataFrame:
    combined = pd.concat([_load_indb(), _load_fndds()], ignore_index=True)
    combined["lookup_name"] = combined["name"].map(_normalize_name)
    combined["lookup_tokens"] = combined["lookup_name"].map(_tokenize)
    combined = combined.drop_duplicates(subset=["lookup_name"], keep="first")
    return combined


def _find_food(food_name: str):
    dataset = _nutrition_df()
    query = _normalize_name(food_name)
    query_tokens = _tokenize(food_name)

    exact = dataset.loc[dataset["lookup_name"] == query]
    if not exact.empty:
        return exact.iloc[0], 100.0

    token_matches = dataset.loc[dataset["lookup_tokens"].map(lambda tokens: bool(query_tokens) and query_tokens.issubset(tokens))]
    if not token_matches.empty:
        ranked = token_matches.assign(
            score=token_matches["lookup_name"].map(
                lambda candidate: difflib.SequenceMatcher(None, query, candidate).ratio() * 100
            ),
            length_gap=token_matches["lookup_name"].map(lambda candidate: abs(len(candidate) - len(query))),
        ).sort_values(by=["score", "length_gap"], ascending=[False, True])
        row = ranked.iloc[0]
        return row, float(row["score"])

    matches = difflib.get_close_matches(query, dataset["lookup_name"].tolist(), n=1, cutoff=0.6)
    if not matches:
        return None

    matched_name = matches[0]
    score = difflib.SequenceMatcher(None, query, matched_name).ratio() * 100
    row = dataset.loc[dataset["lookup_name"] == matched_name].iloc[0]
    return row, score


def _normalize_nutrient_units(row, source):
    """
    Normalize nutrient values to consistent units.
    INDB stores saturated/mono/polyunsaturated fats in mg, FNDDS in g.
    """
    result = {}
    
    for nutrient_key, config in NUTRIENT_COLUMNS.items():
        if nutrient_key in row and pd.notna(row[nutrient_key]):
            value = float(row[nutrient_key])
            
            # Convert INDB fat values from mg to g
            if source == "INDB" and nutrient_key in ["saturated_fat", "monounsaturated_fat", "polyunsaturated_fat"]:
                value = value / 1000.0  # mg to g
            
            result[nutrient_key] = value
        else:
            result[nutrient_key] = None
    
    return result


def get_nutrition(food: str, weight: float):
    """
    Get nutrition data for a food item.
    Returns all 27 common nutrients available in both databases.
    Also includes backward-compatible 'calories' key.
    """
    match = _find_food(food)

    if match is None:
        base_response = {
            "matched_name": None,
            "match_score": 0.0,
            "source": None,
            "calories": None,  # Backward compatibility
        }
        # Add all nutrients as None
        for nutrient_key in NUTRIENT_COLUMNS.keys():
            base_response[nutrient_key] = None
        return base_response

    row, score = match
    factor = max(weight, 0.0) / 100.0

    # Get normalized nutrients
    nutrients = _normalize_nutrient_units(row, row["source"])

    response = {
        "matched_name": row["name"],
        "match_score": float(score),
        "source": row["source"],
    }
    
    # Scale nutrients by weight
    for nutrient_key, value in nutrients.items():
        if value is not None:
            response[nutrient_key] = round(value * factor, 2)
        else:
            response[nutrient_key] = None

    # Backward compatibility: add 'calories' as alias for 'energy_kcal'
    response["calories"] = response.get("energy_kcal")

    return response
