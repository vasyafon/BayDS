from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from lib.pipeline.node import Node
from lib.pipeline.pipeline import *
from lib.io import *
from lib.pipeline.ieee_fraud_nodes import *
from lib.pipeline.main_nodes import *

import os

if __name__ == '__main__':
    # main_dir = r'd:\Documents\Private\Kaggle\Baydin'
    main_dir = r'f:\my\Prog\kaggle\Baydin'
    data_dir = f'{main_dir}/Data'

    p = Pipeline(working_folder=f'{main_dir}/Snapshots/1')

    p.add_node(IEEEFraudTransactionLoaderNode, None, 'transactions',
               params={
                   'input_directory': data_dir
               })

    p.add_node(IEEEFraudIdentityLoaderNode, None, 'identity',
               params={
                   'input_directory': data_dir
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
    p.add_node(ReduceMemoryUsageNode,
               'data', 'data',
               params={
                   'verbose': True
               })
    p.run(verbose=True)
    print(p.data['data'].columns)
    p.save_data('pickle')

    # p.add_node(LoaderNode, None, 'data',
    #            params={
    #                'input_directory': r'f:\my\Prog\kaggle\Baydin\Temp\1',
    #                'file': 'data.pkl'
    #            })

    p.add_node(TimeTransformNode, 'data')
    p.add_node(SomeAggregatesFromAnyaNode, 'data')
    # todo: check LatestBrowser generation
    p.add_node(EmailTransformNode, 'data')
    p.save()

    # p.save_data('hdf')
