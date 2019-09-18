import pandas as pd
import numpy as np


def load_data(file):
    return pd.read_csv(file)


def save_features(dataframes=None, names=None, columns=None, index_column: str = None, path='./pickles', suffix=""):
    for i, df in enumerate(dataframes):
        name = names[i]
        slice = df[[index_column] + columns]
        slice.to_pickle(f"{path}/{name}_{suffix}.pkl")


def load_features(dataframes=None, names=None, index_column: str = None, path='./pickles', suffix=""):
    result = []
    for i, df in enumerate(dataframes):
        name = names[i]
        slice = pd.read_pickle(f"{path}/{name}_{suffix}.pkl")
        df = pd.merge(df, slice, on=index_column, how='left')
        result.append(df)
    return result


def save_features2(df=None, name=None, columns=None, path='./pickles', suffix=""):
    slice = df[columns]
    slice.to_pickle(f"{path}/{name}_{suffix}.pkl")


def load_features2(df, name=None, path='./pickles', suffix=""):
    slice = pd.read_pickle(f"{path}/{name}_{suffix}.pkl")
    df = pd.merge(df, slice, on=index_column, how='left')
    return df


def insert_features(df, name=None, path='./pickles', suffix=""):
    slice = pd.read_pickle(f"{path}/{name}_{suffix}.pkl")
    if np.all(slice.index.values == df.index.values):
        print("coherent")
        for col in slice.columns:
            df[col] = slice[col]
    else:
        print('incoherent')
        raise ValueError


#     return df
#     df = pd.merge(df, slice, on=index_column, how='left')
#     return df


