from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LabelEncoderNan(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = LabelEncoder()
        self.nan_value = -128

    def fit(self,x,y=None):
        #Fill missing values with the string 'NaN'
        a = [q for q in x if q != 'nan']
        self.le.fit(a)
        return self

    def transform(self,x,y=None):
        #Fill missing values with the string 'NaN'
        a = [q for q in x if q != 'nan']
        #Store an ndarray of the current column
        b = np.array(x)
        #Replace the elements in the ndarray that are not 'NaN'
        #using the transformer
        b[b!='nan'] = self.le.transform(a)
        b[b=='nan'] = self.nan_value
        #Overwrite the column in the DataFrame
        x=b.astype(np.int32)
        #return the transformed DataFrame
        return x