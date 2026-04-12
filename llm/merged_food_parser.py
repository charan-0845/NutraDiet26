"""
merged_food_parser.py
─────────────────────────────────────────────────────────────────
Merges structured food info from TWO sources:
  1. User text input   (e.g. "dal makhani with extra butter")
  2. Moondream output  (e.g. "I had 2 pieces of dosa.")

Rules:
  • Text input has PRIORITY for portion sizes.
  • Moondream portions are used ONLY when the food is not mentioned in text.
  • Duplicate food items are removed (matched by canonical label name).
  • Low-confidence matches (< 0.5) are dropped.

Usage:
    from merged_food_parser import merge_food_sources

    items = merge_food_sources(
        user_text="dal makhani and naan",
        moondream_text="I can see 2 pieces of naan and a bowl of dal makhani and some rice."
    )
─────────────────────────────────────────────────────────────────
"""

import re
import json
from typing import Optional
from llm.food_parser_bert import parse_food   # your existing BERT parser

# ─────────────────────────────────────────────────────────────────
# MOONDREAM TEXT CLEANUP
# Strip conversational filler before feeding into the BERT parser
# e.g. "I can see ...", "There appears to be ...", "It looks like ..."
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


def _clean_moondream(text: str) -> str:
    """
    Strip conversational filler from each sentence in Moondream output
    and return a plain food-description string the BERT parser can handle.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    cleaned = []
    for sent in sentences:
        sent = _MOONDREAM_FILLER.sub("", sent).strip(" ,.")
        if sent:
            cleaned.append(sent)
    return ", ".join(cleaned)


# ─────────────────────────────────────────────────────────────────
# DEDUPLICATION HELPER
# Two items are "the same" if their canonical names match exactly.
# (The BERT parser already canonicalises via MAPPING_DICT, so a
#  simple string equality check is sufficient here.)
# ─────────────────────────────────────────────────────────────────

def _same_food(name_a: str, name_b: str) -> bool:
    """
    Return True if two canonical food names refer to the same item.
    Exact match on the canonical label is enough because the BERT
    parser + mapping dict has already normalised both strings.
    """
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
        Free-text food description typed by the user.
    moondream_text : str | None
        Natural-language output from Moondream (or any VLM).
    confidence_threshold : float
        Items with BERT confidence below this are discarded.

    Returns
    -------
    list of dicts, each with keys:
        name            – canonical food label
        quantity_grams  – estimated weight in grams
        source          – "text", "image", or "text+image"
        confidence      – BERT match score
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
    #
    # Strategy:
    #   • Start with all text items (they always make it in).
    #   • For each image item:
    #       - If already present in text → skip (text portion wins).
    #         But if text item has no portion (None), borrow from image.
    #       - If NOT present → add it with image portion.
    #

    merged: list[dict] = list(text_items)   # copy so we can mutate

    # Build a quick lookup: canonical name → index in `merged`
    name_index: dict[str, int] = {
        item["name"]: idx for idx, item in enumerate(merged)
    }

    for img_item in image_items:
        img_name = img_item["name"]

        if img_name in name_index:
            # Duplicate found — text item already exists
            idx = name_index[img_name]
            existing = merged[idx]

            # Borrow portion from image ONLY if text gave us nothing
            if existing.get("quantity_grams") is None and img_item.get("quantity_grams") is not None:
                merged[idx] = {
                    **existing,
                    "quantity_grams": img_item["quantity_grams"],
                    "source": "text+image",   # portion came from image
                }
            # else: keep text portion as-is

        else:
            # New food only seen by Moondream → add it
            merged.append(img_item)
            name_index[img_name] = len(merged) - 1

    # ── 4. Final clean-up ─────────────────────────────────────────
    output = []
    for item in merged:
        output.append({
            "name":           item["name"],
            "quantity_grams": item.get("quantity_grams"),
            "source":         item.get("source", "unknown"),
            "confidence":     round(item.get("confidence", 0.0), 3),
        })

    return output


# ─────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER  (mirrors your existing parse_food_items API)
# ─────────────────────────────────────────────────────────────────

def parse_food_items_merged(
    user_text: Optional[str] = None,
    moondream_text: Optional[str] = None,
) -> list[dict]:
    """
    Drop-in replacement for your existing parse_food_items() that also
    accepts a moondream_text argument.

    Returns only {name, quantity_grams} like the original, but also
    includes 'source' for debugging.
    """
    results = merge_food_sources(user_text=user_text, moondream_text=moondream_text)
    return [
        {
            "name":           r["name"],
            "quantity_grams": r["quantity_grams"],
            "source":         r["source"],
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────
# CLI — quick manual testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🍛  MERGED FOOD PARSER  (text + Moondream)\n")

    EXAMPLES = [
        {
            "label":    "Text mentions naan, image adds rice",
            "text":     "dal makhani and naan",
            "moondream":"I had 2 pieces of naan and a bowl of dal makhani and some rice.",
        },
        {
            "label":    "Text has portions, image has different portions → text wins",
            "text":     "1 cup dosa",
            "moondream":"I had 2 pieces of dosa.",
        },
        {
            "label":    "Text mentions food but no portion → image portion used",
            "text":     "idli and sambar",
            "moondream":"I can see 3 idlis and a bowl of sambar.",
        },
        {
            "label":    "Only Moondream, no text",
            "text":     None,
            "moondream":"There appears to be a plate of biryani and some raita.",
        },
        {
            "label":    "Only text, no Moondream",
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
        result = parse_food_items_merged(
            user_text=ex["text"],
            moondream_text=ex["moondream"],
        )
        print("\n   Merged output:")
        print(json.dumps(result, indent=4))
        print()

    # Interactive mode
    print(f"{'─'*60}")
    print("🎤  Interactive mode  (type 'exit' to quit)\n")
    while True:
        u = input("User text   (blank = none): ").strip() or None
        m = input("Moondream   (blank = none): ").strip() or None
        if u and u.lower() in ("exit", "quit"):
            break
        if m and m.lower() in ("exit", "quit"):
            break
        out = parse_food_items_merged(user_text=u, moondream_text=m)
        print("\nMerged Output:")
        print(json.dumps(out, indent=2))
        print()
