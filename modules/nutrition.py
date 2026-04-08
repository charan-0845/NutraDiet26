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


def _load_fndds() -> pd.DataFrame:
    raw = pd.read_excel(FNDDS_PATH, header=1)
    data = raw.rename(
        columns={
            "Main food description": "name",
            "Energy (kcal)": "calories",
            "Protein (g)": "protein",
            "Carbohydrate (g)": "carbs",
            "Total Fat (g)": "fat",
        }
    )
    data = data[["name", "calories", "protein", "carbs", "fat"]].dropna(subset=["name"])
    data["source"] = "FNDDS"
    return data


def _load_indb() -> pd.DataFrame:
    raw = pd.read_excel(INDB_PATH)
    data = raw.rename(
        columns={
            "food_name": "name",
            "energy_kcal": "calories",
            "protein_g": "protein",
            "carb_g": "carbs",
            "fat_g": "fat",
        }
    )
    data = data[["name", "calories", "protein", "carbs", "fat"]].dropna(subset=["name"])
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


def get_nutrition(food: str, weight: float):
    match = _find_food(food)

    if match is None:
        return {
            "matched_name": None,
            "match_score": 0.0,
            "source": None,
            "calories": None,
            "protein": None,
            "carbs": None,
            "fat": None,
        }

    row, score = match
    factor = max(weight, 0.0) / 100.0

    return {
        "matched_name": row["name"],
        "match_score": float(score),
        "source": row["source"],
        "calories": float(row["calories"]) * factor,
        "protein": float(row["protein"]) * factor,
        "carbs": float(row["carbs"]) * factor,
        "fat": float(row["fat"]) * factor,
    }
