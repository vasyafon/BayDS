from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from lib.pipeline.node import Node
import gc
import pandas as pd
from pandas.core.frame import DataFrame
import json
import time
import datetime
from collections.abc import Iterable


class EraserNode(Node):
    params = {
        'remove_keys': []
    }


class Pipeline(object):
    working_folder: str = ""
    nodes: List[Tuple[Type[Node], Union[str, Tuple[str, ...]], Union[str, Tuple[str, ...]], Any]] = []
    data: Dict[str, Any] = {}
    running_cursor: int = 0
    name: str = 'pipeline'

    def __init__(self, working_folder, name=None):
        self.working_folder = working_folder
        if name is not None:
            self.name = name

    def add_node(self, node: Type[Node], input_key: Union[str, Tuple[str, ...]] = None,
                 output_key: Union[str, Tuple[str, ...]] = None, params=None):
        self.nodes.append((node, input_key, output_key, params))

    def reset(self):
        self.running_cursor = 0
        self.data = {}
        gc.collect()

    def run(self, verbose=True):
        while (self.running_cursor < len(self.nodes)):
            self.run_step(verbose=verbose)

    def run_step(self, verbose=True):
        start_date = datetime.datetime.now()
        (node, input_key, output_key, params) = self.nodes[self.running_cursor]
        if verbose:
            print('---------------------------')
            print(f'{self.running_cursor}: {node.__name__} [{start_date.strftime("%Y-%m-%d %H:%M:%S")}]')
            print('params:\n', str(params))
        n: Node = node(params=params)
        if input_key is not None:
            if isinstance(input_key, str):
                n.input = self.data[input_key]
            elif isinstance(input_key, Iterable):
                n.input = [self.data[k] for k in input_key]
        n.start()
        if output_key is not None:
            assert n.output is not None
            if isinstance(output_key, str):
                self.data[output_key] = n.output
            elif isinstance(output_key, Iterable):
                for i, k in enumerate(output_key):
                    self.data[k] = n.output[i]

        self.running_cursor += 1

    def add_node_and_run(self, node: Type[Node], input_key: Union[str, Tuple[str, ...]] = None,
                         output_key: Union[str, Tuple[str, ...]] = None,
                         params=None, verbose=True):
        assert self.running_cursor == len(self.nodes)
        self.add_node(node, input_key, output_key, params)
        self.run_step(verbose=verbose)

    def save_data(self, format='pickle'):
        for k, df in self.data.items():
            if not isinstance(df, (pd.DataFrame, DataFrame)):
                print(f"Error, {k} is not DataFrame, but {type(df)}")
                # continue
            df: pd.DataFrame
            if format == 'pickle':
                df.to_pickle(f'{self.working_folder}/{k}.pkl')
            elif format == 'csv':
                df.to_csv(f'{self.working_folder}/{k}.csv')
            if format == 'feather':
                df.to_feather(f'{self.working_folder}/{k}.fth')
            if format == 'hdf':
                df.to_hdf(f'{self.working_folder}/{self.name}.h5', k)

    def save(self, format='json'):
        node_description = []
        for (node, input_key, output_key, params) in self.nodes:
            node_description.append({
                'Node': node.__name__,
                'input_key': str(input_key),
                'output_key': str(output_key),
                'params': params
            })
        json.dump(node_description, open(f'{self.working_folder}/{self.name}.json', 'w'), indent=2)