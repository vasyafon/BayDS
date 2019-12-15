from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from .node import Node
from .pipeline import *
from ..io import *
import numpy as np
import os
from sklearn import preprocessing
from ..encoding import LabelEncoderPopularity
from ..aggregations.temporal import aggregate_with_time_local, aggregate_transaction_frequencies
import multiprocess as mp


class AddAggregatesTotalNode(Node):
    '''
    For every numerical feature makes to_mean and to_std
    '''
    params = {
        'features': [],
        'group_by': ''
    }

    def _run(self):
        data = self.input
        group_by_feature = self.params['group_by']

        for fname in self.params['features']:
            data[f'{fname}_to_mean_{group_by_feature}'] = data[fname] / data.groupby([group_by_feature])[
                fname].transform('mean').replace(-np.inf, np.nan).replace(np.inf, np.nan).astype(np.float32)
            data[f'{fname}_to_std_{group_by_feature}'] = data[fname] / data.groupby([group_by_feature])[
                fname].transform('std').replace(-np.inf, np.nan).replace(np.inf, np.nan).astype(np.float32)
            # num_cols.extend([f'{fname}_to_mean_{group_by_feature}', f'{fname}_to_std_{group_by_feature}'])

        self.output = data