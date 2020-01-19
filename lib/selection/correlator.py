import numpy as np
import pandas as pd
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
from scipy.linalg import norm
from tqdm import tqdm
import gc

class Correlator(object):
    index = None

    @property
    def data(self):
        return self.gpudata.get()


    def __init__(self, df: pd.DataFrame, size=3000):
        linalg.init()

        self.index = df.index
        self.size = size
        self.gpudata = gpuarray.zeros((len(df.index), size), dtype=np.float32)
        self.names = []
        for col in tqdm(df.columns):
            self.add_vector(df[col])

    def add_vector(self, v: pd.Series):
        vector = np.ascontiguousarray(v.replace([np.inf, -np.inf], 0).fillna(0).values)
        n = norm(vector)
        if n == 0:
            print("Warning, not adding empty array")
            return
        pos = len(self.names) + 1
        if pos >= self.size:
            raise MemoryError("Can't allocate more")
        self.names.append(v.name)
        vector = (vector / n).astype(np.float32)
        #         if self.gpudata is None:
        #             self.gpudata = gpuarray.to_gpu(vector.reshape(-1,1))
        #         else:
        #         sh = self.gpudata.shape
        #         newdata = gpuarray.zeros((sh[0],sh[1]+1),dtype=np.float32)
        #         newdata[:,:-1] = self.gpudata
        #         newdata[:,-1:] =  gpuarray.to_gpu(vector.reshape(-1,1))
        #         self.gpudata=newdata
        self.gpudata[:, pos - 1:pos] = gpuarray.to_gpu(vector.reshape(-1, 1))
        gc.collect()

    def get_max_corr(self, v: pd.Series):
        vector = np.ascontiguousarray(v.replace([np.inf, -np.inf], 0).fillna(0).values.reshape(-1, 1))
        n = norm(vector)
        if n == 0:
            print("Warning, empty array")
            return 0
        vector = (vector / n).astype(np.float32)
        gpu_vector = gpuarray.to_gpu(vector)
        maxcorr = gpuarray.max(linalg.dot(self.gpudata, gpu_vector, transa='T')[:len(self.names)]).get()
        return maxcorr
