"""
merged_food_parser.py
─────────────────────────────────────────────────────────────────
Merges structured food info from TWO sources:
  1. User text input   (e.g. "dal makhani with extra butter")
  2. Moondream output  (e.g. "1 small bowl of chutney, 300g of rice, 1 cup of curry")
 
Rules:
  • Text input has PRIORITY for portion sizes.
  • Moondream portions are used ONLY when the food is not mentioned in text.
  • Duplicate food items are removed (matched by canonical label name).
  • Low-confidence matches (< 0.5) are dropped.
"""
 
import re
import json
from typing import Optional
from llm.food_parser_bert import parse_food   # your existing BERT parser
 
# ─────────────────────────────────────────────────────────────────
# MOONDREAM TEXT CLEANUP
# ─────────────────────────────────────────────────────────────────
 
_MOONDREAM_FILLER = re.compile(
    r"^(i (can see|see|had|notice|observe|think i see)|"
    r"there (is|are|appears? to be)|"
    r"it looks? like|"
    r"the image (shows?|contains?|has)|"
    r"this (is|looks like)|"
    r"in (the|this) (image|photo|picture)[,.]?\s*)",
    re.IGNORECASE,
)
 
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
 
# "Xg/ml/kg of Y"  →  "Xg Y"  (keep numeric unit, drop "of")
_GRAMS_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*(g|grams?|kg|ml)\s+of\s+",
    re.IGNORECASE,
)
 
# "N [size] unit(s) of X"  →  "N unit X"
_NUM_UNIT_OF = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+"
    r"(?:small\s+|large\s+|medium\s+)?"
    r"(bowls?|plates?|cups?|pieces?|glasses?|servings?|tablespoons?|teaspoons?)"
    r"\s+of\s+",
    re.IGNORECASE,
)
 
# "a/an [size] unit of X"  →  "1 unit X"
_ART_UNIT_OF = re.compile(
    r"\ban?\s+(?:small\s+|large\s+|medium\s+)?"
    r"(bowl|plate|cup|piece|glass|serving|tablespoon|teaspoon)"
    r"\s+of\s+",
    re.IGNORECASE,
)
 
_SOME         = re.compile(r"\bsome\s+", re.IGNORECASE)
_BARE_ARTICLE = re.compile(r"\ban?\s+",  re.IGNORECASE)
_STRAY_OF     = re.compile(r"\bof\s+",   re.IGNORECASE)
_SIZE_WORDS   = re.compile(r"\b(small|large|medium|big|little|tiny|huge)\b", re.IGNORECASE)
 
 
def _normalise_chunk(chunk: str) -> str:
    """
    Normalise a single food chunk (no commas) so extract_quantity() can parse it.
 
    Examples
    --------
    "1 small bowl of chutney"  →  "1 bowl chutney"
    "300g of rice"             →  "300g rice"
    "1 cup of curry"           →  "1 cup curry"
    "2 pieces of naan"         →  "2 piece naan"
    "a bowl of dal"            →  "1 bowl dal"
    "some rice"                →  "rice"
    """
    # "Xg of Y" → "Xg Y"
    chunk = _GRAMS_OF.sub(lambda m: f"{m.group(1)}{m.group(2)} ", chunk)
 
    # "N unit(s) of X" → "N unit X"
    chunk = _NUM_UNIT_OF.sub(
        lambda m: f"{m.group(1)} {m.group(2).rstrip('sS')} ", chunk
    )
 
    # "a/an unit of X" → "1 unit X"
    chunk = _ART_UNIT_OF.sub(
        lambda m: f"1 {m.group(1).lower()} ", chunk
    )
 
    chunk = _SOME.sub("", chunk)
    chunk = _BARE_ARTICLE.sub("", chunk)
    chunk = _STRAY_OF.sub("", chunk)
    chunk = _SIZE_WORDS.sub("", chunk)
 
    return chunk.strip()
 
 
def _clean_moondream(text: str) -> str:
    """
    Convert raw Moondream output into a clean comma-joined string
    that parse_food() can handle.
 
    Key fix: split on commas FIRST (per-chunk), then normalise each
    chunk independently.  Previously the whole sentence was normalised
    as one string, so "1 small bowl of chutney, 300g of rice, 1 cup of
    curry" was treated as a single food item.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    cleaned_chunks = []
 
    for sent in sentences:
        # Strip leading conversational filler
        sent = _MOONDREAM_FILLER.sub("", sent).strip(" ,.")
        if not sent:
            continue
 
        # ── CORE FIX ─────────────────────────────────────────────
        # Split on commas so each food item is handled on its own,
        # then normalise each chunk independently.
        # ─────────────────────────────────────────────────────────
        for chunk in sent.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            normalised = _normalise_chunk(chunk).strip(" ,.")
            if normalised:
                cleaned_chunks.append(normalised)
 
    return ", ".join(cleaned_chunks)
 
 
# ─────────────────────────────────────────────────────────────────
# DEDUPLICATION HELPER
# ─────────────────────────────────────────────────────────────────
 
def _same_food(name_a: str, name_b: str) -> bool:
    return name_a.strip().lower() == name_b.strip().lower()
 
 
# ─────────────────────────────────────────────────────────────────
# CORE MERGE LOGIC
# ─────────────────────────────────────────────────────────────────
 
def merge_food_sources(
    user_text: Optional[str] = None,
    moondream_text: Optional[str] = None,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """
    Parse user text and/or Moondream output, then merge them.
 
    Parameters
    ----------
    user_text : str | None
    moondream_text : str | None
    confidence_threshold : float  – items below this BERT score are dropped
 
    Returns
    -------
    list of dicts: name, quantity_grams, source, confidence
    """
 
    # ── 1. Parse text items ───────────────────────────────────────
    text_items: list[dict] = []
    if user_text and user_text.strip():
        raw = parse_food(user_text)
        text_items = [
            {**item, "source": "text"}
            for item in raw
            if item.get("confidence", 1.0) >= confidence_threshold
        ]
 
    # ── 2. Parse Moondream items ──────────────────────────────────
    image_items: list[dict] = []
    if moondream_text and moondream_text.strip():
        cleaned = _clean_moondream(moondream_text)
        raw = parse_food(cleaned)
        image_items = [
            {**item, "source": "image"}
            for item in raw
            if item.get("confidence", 1.0) >= confidence_threshold
        ]
 
    # ── 3. Merge: text has priority ───────────────────────────────
    merged: list[dict] = list(text_items)
    name_index: dict[str, int] = {
        item["name"]: idx for idx, item in enumerate(merged)
    }
 
    for img_item in image_items:
        img_name = img_item["name"]
        if img_name in name_index:
            idx = name_index[img_name]
            existing = merged[idx]
            if existing.get("quantity_grams") is None and img_item.get("quantity_grams") is not None:
                merged[idx] = {
                    **existing,
                    "quantity_grams": img_item["quantity_grams"],
                    "source": "text+image",
                }
        else:
            merged.append(img_item)
            name_index[img_name] = len(merged) - 1
 
    # ── 4. Final clean-up ─────────────────────────────────────────
    return [
        {
            "name":           item["name"],
            "quantity_grams": item.get("quantity_grams"),
            "source":         item.get("source", "unknown"),
            "confidence":     round(item.get("confidence", 0.0), 3),
        }
        for item in merged
    ]
 
 
# ─────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────
 
def parse_food_items_merged(
    user_text: Optional[str] = None,
    moondream_text: Optional[str] = None,
) -> list[dict]:
    results = merge_food_sources(user_text=user_text, moondream_text=moondream_text)
    return [
        {"name": r["name"], "quantity_grams": r["quantity_grams"], "source": r["source"]}
        for r in results
    ]
 
 
# ─────────────────────────────────────────────────────────────────
# CLI — quick manual testing
# ─────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n🍛  MERGED FOOD PARSER  (text + Moondream)\n")
 
    EXAMPLES = [
        {
            "label":    "Moondream comma list — the reported bug",
            "text":     None,
            "moondream":"1 small bowl of chutney, 300g of rice, 1 cup of curry",
        },
        {
            "label":    "Mixed: text + Moondream comma list",
            "text":     "dal makhani and naan",
            "moondream":"1 cup of dal makhani, 2 pieces of naan, 300g of rice",
        },
        {
            "label":    "Text no portion → image portion used",
            "text":     "idli and sambar",
            "moondream":"3 pieces of idli, 1 bowl of sambar",
        },
        {
            "label":    "Filler sentence + comma list",
            "text":     None,
            "moondream":"I can see a plate of biryani, some raita, and 2 pieces of naan.",
        },
        {
            "label":    "Only text",
            "text":     "two chapati and dal",
            "moondream":None,
        },
    ]
 
    for ex in EXAMPLES:
        print(f"{'─'*60}")
        print(f"🔖  {ex['label']}")
        if ex["text"]:
            print(f"   Text input : {ex['text']}")
        if ex["moondream"]:
            print(f"   Moondream  : {ex['moondream']}")
            print(f"   Cleaned    : {_clean_moondream(ex['moondream'])}")
        result = parse_food_items_merged(user_text=ex["text"], moondream_text=ex["moondream"])
        print("\n   Merged output:")
        print(json.dumps(result, indent=4))
        print()
 
    print(f"{'─'*60}")
    print("🎤  Interactive mode  (type 'exit' to quit)\n")
    while True:
        u = input("User text   (blank = none): ").strip() or None
        m = input("Moondream   (blank = none): ").strip() or None
        if (u and u.lower() in ("exit", "quit")) or (m and m.lower() in ("exit", "quit")):
            break
        if m:
            print(f"   Cleaned moondream: {_clean_moondream(m)}")
        out = parse_food_items_merged(user_text=u, moondream_text=m)
        print("\nMerged Output:")
        print(json.dumps(out, indent=2))
        print()
 