import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.linalg import norm


class CorrelatorCPU(object):
    index = None

    def __init__(self, df: pd.DataFrame, size=5000, normalize=True):
        self.index = df.index
        self.size = size
        self.data = np.zeros((len(df.index), size), dtype=np.float32)
        if normalize:
            self.names = []
            for col in tqdm(df.columns):
                self.add_vector(df[col])
        else:
            self.names = list(df.columns)
            self.data[:, :len(self.names)] = df.values

    def add_vector(self, v: pd.Series):
        vector = np.ascontiguousarray(v.replace([np.inf, -np.inf], [0, 0]).fillna(0).values)
        n = norm(vector)
        if n == 0:
            print("Warning, not adding empty array")
            return
        pos = len(self.names) + 1
        if pos >= self.size:
            raise MemoryError("Can't allocate more")
        self.names.append(v.name)
        vector = (vector / n).astype(np.float32)
        self.data[:, pos - 1:pos] = vector.reshape(-1, 1)

    def get_max_corr(self, v: pd.Series):
        vector = np.ascontiguousarray(v.replace([np.inf, -np.inf], 0).fillna(0).values.reshape(-1, 1))
        n = norm(vector)
        if n == 0:
            print("Warning, empty array")
            return 0
        vector = (vector / n).astype(np.float32)
        maxcorr = np.amax(np.dot(vector.T, self.data)[:len(self.names)])
        return maxcorr
