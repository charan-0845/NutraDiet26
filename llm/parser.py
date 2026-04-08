# =========================
# IMPORTS
# =========================
from llm.food_parser_bert import parse_food


# =========================
# MAIN FUNCTION (BERT BASED)
# =========================
def parse_food_items(text: str):
    if not text or not text.strip():
        return []

    results = parse_food(text)

    parsed_items = []

    for item in results:
        # Optional: filter low confidence
        if item.get("confidence", 1) < 0.5:
            continue

        parsed_items.append({
            "name": item["name"],
            "quantity_grams": item["quantity_grams"]
        })

    return parsed_items


# =========================
# TEST
# =========================
if __name__ == "__main__":
    while True:
        user = input("Enter food: ")
        if user.lower() in ["exit", "quit"]:
            break

        output = parse_food_items(user)

        print("\nParsed Output:")
        print(output)
        print()