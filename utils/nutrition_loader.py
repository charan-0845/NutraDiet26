import pandas as pd

def load_datasets():
    fndds = pd.read_excel("data/FNDDS.xlsx")
    indb  = pd.read_excel("data/INDB.xlsx")

    print("FNDDS columns:", fndds.columns.tolist())
    print("INDB columns:", indb.columns.tolist())

    return fndds, indb