from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderPopularity(BaseEstimator, TransformerMixin):
    def __init__(self, convert_nan=False, negate_nan=True):
        self.le = LabelEncoder()
        self.convert_nan = convert_nan
        self.negate_nan = negate_nan
        self.replacement_dict = {}

    def fit(self, x, y=None):
        value_counts = x.value_counts()
        stats = []

        for v, c in value_counts.iteritems():
            #             print(v,c)
            stats.append((v, c))

        if self.convert_nan and not self.negate_nan:
            stats.append((np.NaN, sum(x.isna())))
            stats = sorted(stats, key=lambda q: q[1], reverse=True)

        for i, s in enumerate(stats):
            self.replacement_dict[s[0]] = i

        if self.convert_nan and self.negate_nan:
            self.replacement_dict[np.NaN] = -1
        #         print(self.replacement_dict)
        #         #Fill missing values with the string 'NaN'
        #         a = [q for q in x if q != 'nan']
        #         self.le.fit(a)
        return self

    def transform(self, x, y=None):
        tr = x.map(self.replacement_dict)  # .astype(np.int32)
        return tr

#         #Fill missing values with the string 'NaN'
#         a = [q for q in x if q != 'nan']
#         #Store an ndarray of the current column
#         b = np.array(x)
#         #Replace the elements in the ndarray that are not 'NaN'
#         #using the transformer
#         b[b!='nan'] = self.le.transform(a)
#         b[b=='nan'] = self.nan_value
#         #Overwrite the column in the DataFrame
#         x=b.astype(np.int32)
#         #return the transformed DataFrame
#         return x
# L = LabelEncoderPopularity(convert_nan=True)
# L.fit(train_transaction['card3'])
# t = L.transform(train_transaction['card3'])
# t.head()