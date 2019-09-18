from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from lib.pipeline.node import Node
from lib.pipeline.pipeline import *
from lib.io import *
import os


class IEEEFraudTransactionLoaderNode(Node):
    params = {
        'input_directory': None
    }

    def _run(self):
        folder_path = self.params['input_directory']
        files = [f'{folder_path}/train_transaction.csv',
                 f'{folder_path}/test_transaction.csv']
        train_transaction, test_transaction = map(load_data, files)
        test_transaction['isFraud'] = -1
        data = pd.concat([train_transaction, test_transaction], axis=0, sort=False)
        data.set_index('TransactionID', inplace=True)
        self.output = data


class IEEEFraudIdentityLoaderNode(Node):
    params = {
        'input_directory': None
    }

    def _run(self):
        folder_path = self.params['input_directory']
        files = [f'{folder_path}/train_identity.csv',
                 f'{folder_path}/test_identity.csv']
        train_identity, test_identity = map(load_data, files)
        data_id = pd.concat([train_identity, test_identity], axis=0, sort=False)
        data_id.set_index('TransactionID', inplace=True)
        self.output = data_id


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
        cols_to_drop = [col for col in self.params['drop'] if col in self.input.columns]
        self.output = self.input.drop(cols_to_drop, axis=1)


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


class AddNaNCountNode(Node):
    params = {
        'name': None,
        'inplace': True
    }

    def _run(self):
        nan_count = self.input.isna().sum(axis=1)
        if self.params['inplace']:
            self.input[self.params['name']] = nan_count
            self.output = self.input
        else:
            self.output = pd.DataFrame(index=self.input.index)
            self.output[self.params['name']] = nan_count


class FunctionNode(Node):
    params = {
        'function': lambda q: q
    }

    def _run(self):
        self.output = self.params['function'](self.input)


if __name__ == '__main__':
    p = Pipeline(working_folder=r'f:\my\Prog\kaggle\Baydin\Temp\1')

    p.add_node(IEEEFraudTransactionLoaderNode, None, 'transactions',
               params={
                   'input_directory': r'f:\my\Prog\kaggle\Fraud\data'
               })

    p.add_node(IEEEFraudIdentityLoaderNode, None, 'identity',
               params={
                   'input_directory': r'f:\my\Prog\kaggle\Fraud\data'
               })

    p.add_node(AddNaNCountNode, 'transactions', 'transactions',
               params={
                   'name': 'NanTransactionCount'
               })
    p.add_node(AddNaNCountNode, 'identity', 'identity',
               params={
                   'name': 'NanIdentityCount'
               })
    p.add_node(JoinNode,
               ('transactions', 'identity'),
               'data',
               params={
                   'on': 'TransactionID'
               })

    p.add_node(EraserNode, params={
        'remove_keys': ['transactions', 'identity']
    })

    # p.add_node(FunctionNode,)

    # p.add_node(LoaderNode, None, 'raw_data',
    #            params={
    #                'input_directory': r'f:\my\Prog\kaggle\Baydin\Temp\1',
    #                'file': 'data.pkl'
    #            })

    p.save()
    p.run(verbose=True)
    print(p.data)
    p.save_data('pickle')
    # p.save_data('hdf')
