from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from .node import Node
from .pipeline import *
from ..io import *
from ..utils import *
from .ieee_fraud_nodes import *
import os
import yaml
import json


class LoaderNode(Node):
    params = {
        'input_directory': None,
        'file': Union[str, Tuple[str, str]],
    }

    def _run(self):
        folder_path = self.params['input_directory']
        data = None

        filename = self.params['file']
        if isinstance(filename, str):
            ext = os.path.splitext(filename)[1]
            if ext == '.csv':
                data = pd.read_csv(f'{folder_path}/{filename}')
            elif ext == '.pkl':
                data = pd.read_pickle(f'{folder_path}/{filename}')
            elif ext == '.fth':
                data = pd.read_feather(f'{folder_path}/{filename}')
            elif ext == '.h5':
                print('You should set "file" to tuple (filename, key)')
                raise RuntimeError
            elif ext in ('.yaml', '.yml'):
                data = yaml.load(open(f'{folder_path}/{filename}', 'r'), Loader=yaml.FullLoader)
            elif ext == 'json':
                data = json.load(open(f'{folder_path}/{filename}', 'r'))
            else:
                print('unsupported file extension')
                raise RuntimeError
        elif isinstance(filename, tuple):
            data = pd.read_hdf(f'{folder_path}/{filename[0]}', filename[1])

        self.output = data


class FilterFeaturesNode(Node):
    params = {
        'remain': List[str]
    }

    def _run(self):
        cols_to_drop = [col for col in self.input.columns if col not in self.params['remain']]
        self.output = self.input.drop(cols_to_drop, axis=1)


class DropFeaturesNode(Node):
    params = {
        'drop': List[str]
    }

    def _run(self):
        data = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]

        cols_to_drop = [col for col in self.params['drop'] if col in data.columns]
        for col in self.params['drop']:
            if col in num_cols:
                num_cols.remove(col)
            if col in cat_cols:
                cat_cols.remove(col)


        self.output = (data.drop(cols_to_drop, axis=1), num_cols,cat_cols)


class JoinNode(Node):
    params = {
        'on': None,
        'type': 'merge'
    }

    def _run(self):
        data = None
        if self.params['type'] == 'merge':
            data = pd.merge(self.input[0], self.input[1], on=self.params['on'], how='left')
        self.output = data


class FunctionNode(Node):
    params = {
        'function': lambda q: q
    }

    def _run(self):
        self.output = self.params['function'](self.input)


class ReduceMemoryUsageNode(Node):
    params = {
        'verbose': False
    }

    def _run(self):
        self.output = reduce_mem_usage_sd(self.input, verbose=self.params['verbose'])


class CatBoostEncoderNode(Node):
    params = {

    }

    def _run(self):
        from category_encoders.cat_boost import CatBoostEncoder

        data = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]

        train = data[data['isFraud'] != -1]

        X = train.drop('isFraud', axis=1)
        y = train['isFraud'].astype(np.uint8)

        del train

        encoder = CatBoostEncoder(verbose=1, cols=cat_cols)
        encoder.fit(X, y)

        cat_data: pd.DataFrame = data.drop('isFraud', axis=1)
        cat_data = encoder.transform(cat_data)
        cat_data = cat_data.join(data['isFraud'])
        self.output = cat_data


class LabelEncoderNode(Node):
    params = {
    }

    def _run(self):
        from category_encoders.cat_boost import CatBoostEncoder

        data = self.input[0]
        num_cols = self.input[1]
        cat_cols = self.input[2]

        le = LabelEncoderPopularity(convert_nan=True)
        for feature in cat_cols:
            le.fit(data[feature].astype(str))
            data[feature] = le.transform(data[feature].astype(str)).astype(np.int32)

        self.output = data


class DownsamplingTrainNode(Node):
    params = {
        'random_state': 12,
        'strategy': 'equalize',
        'desired_size': None
    }

    def _run(self):
        data = self.input

        train_pos = data[data['isFraud'] == 1]
        train_neg = data[data['isFraud'] == 0]

        if self.params['strategy'] == 'equalize':
            desired_size = train_pos.shape[0]
        elif self.params['strategy'] == 'desired_size':
            desired_size = self.params['desired_size']
        elif self.params['strategy'] == 'reduction_factor':
            desired_size = int(train_neg.shape[0] * self.params['reduction_factor'])
        train_neg = train_neg.sample(desired_size, random_state=self.params['random_state'])

        self.output = pd.concat([train_pos, train_neg]).sort_index()
