from rapidfuzz import process

def find_food(food_name, dataset):
    names = dataset["name"].tolist()

    match, score, idx = process.extractOne(food_name, names)

    if score < 70:
        return None

    return dataset.iloc[idx]


def get_nutrition(food_name, fndds, indb):

    result = find_food(food_name, indb)

    if result is None:
        result = find_food(food_name, fndds)

    if result is None:
        return {"error": "Food not found"}

    return {
        "name": result["name"],
        "energy": result["energy"],
        "protein": result["protein"],
        "carbs": result["carbs"],
        "fat": result["fat"]
    }