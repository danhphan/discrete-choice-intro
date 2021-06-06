import numpy as np
import pandas as pd

def accuracy(actual, pred):
    res = np.sum(np.array(actual) == np.array(pred)) / len(actual)
    return res


def add_utilities(db, V):
    df = db.data
    for idx, v in V.items():
        df[f'V_{idx}'] = db.valuesFromDatabase(v)
    Vs = [f'V_{idx}' for idx in V.keys()]
    df["PRED"] = np.argmax(df[Vs].values, axis=1) + 1
    return df