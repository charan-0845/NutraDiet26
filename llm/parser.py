# =========================
# IMPORTS
# =========================
from llm.merged_food_parser import merge_food_sources


# =========================
# MAIN FUNCTION (BERT BASED)
# =========================
def parse_food_items(text: str = None, moondream_text: str = None):
    if not text and not moondream_text:
        return []

    results = merge_food_sources(
        user_text=text,
        moondream_text=moondream_text,
        confidence_threshold=0.5,
    )

    return [
        {
            "name": item["name"],
            "quantity_grams": item["quantity_grams"],
            "source": item.get("source", "unknown"),
            "confidence": item.get("confidence", 1.0),
        }
        for item in results
    ]


# =========================
# TEST
# =========================
if __name__ == "__main__":
    while True:
        text = input("User text   (blank = none): ").strip() or None
        moon = input("Moondream   (blank = none): ").strip() or None

        if text in ["exit", "quit"] or moon in ["exit", "quit"]:
            break

        output = parse_food_items(text=text, moondream_text=moon)

        print("\nParsed Output:")
        print(output)
        print()
