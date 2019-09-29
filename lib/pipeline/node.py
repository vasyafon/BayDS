from typing import List, Set, Dict, Optional, Any, Union
import enum
import pandas as pd
import abc
import gc


class Node(object):
    # has_input_df: bool = False
    # has_output_df: bool = False
    generated_files: Set[str] = set()
    input: Union[pd.DataFrame, dict, list] = None
    output: Union[pd.DataFrame, dict, list] = None
    save_output: bool = True
    params: Dict[str, Any] = {}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if params is not None:
            for k, v in params.items():
                self.params[k] = v

    @abc.abstractmethod
    def _run(self):
        pass

    def start(self):
        # if self.has_input_df:
        #     assert self.input is not None

        # save params if needed

        self._run()
        gc.collect()
