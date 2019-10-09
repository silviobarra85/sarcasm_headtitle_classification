import pandas as pd


def loadDataFromPath(path):
    df = pd.read_json(path_or_buf=path, lines=True, encoding='utf-8')

    return df
