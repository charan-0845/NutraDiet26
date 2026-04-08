def preprocess_datasets(fndds, indb):

    fndds = fndds.rename(columns={
        "Food Description": "name",
        "Energy (kcal)": "energy",
        "Protein (g)": "protein",
        "Carbohydrate (g)": "carbs",
        "Fat (g)": "fat"
    })

    indb = indb.rename(columns={
        "Food Name": "name",
        "Energy_kcal": "energy",
        "Protein_g": "protein",
        "Carbohydrate_g": "carbs",
        "Fat_g": "fat"
    })

    cols = ["name", "energy", "protein", "carbs", "fat"]

    fndds = fndds[cols].dropna()
    indb  = indb[cols].dropna()

    return fndds, indb