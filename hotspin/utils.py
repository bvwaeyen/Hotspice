import datetime
import getpass
import inspect
import io
import json
import math
import os
import subprocess
import sys
import warnings

import cupy as cp
import cupy.lib.stride_tricks as striding
import numpy as np
import pandas as pd

from IPython.terminal.embed import InteractiveShellEmbed
from typing import Callable, Iterable, TypeVar


def mirror4(arr, /, *, negativex=False, negativey=False):
    ''' Mirrors the 2D CuPy array <arr> along some of the edges, in such a manner that the
        original element at [0,0] ends up in the middle of the returned array.
        Hence, if <arr> has shape (a, b), the returned array has shape (2*a - 1, 2*b - 1).
    '''
    ny, nx = arr.shape
    arr4 = cp.zeros((2*ny-1, 2*nx-1))
    xp = -1 if negativex else 1
    yp = -1 if negativey else 1
    arr4[ny-1:, nx-1:] = arr
    arr4[ny-1:, nx-1::-1] = xp*arr
    arr4[ny-1::-1, nx-1:] = yp*arr
    arr4[ny-1::-1, nx-1::-1] = xp*yp*arr
    return arr4


def filter_kwargs(kwargs: dict, func: Callable):
    return {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}


Field = TypeVar("Field", int, float, np.ndarray, cp.ndarray) # Every type that can be parsed by as_cupy_array()
def as_cupy_array(value: Field, shape: tuple) -> cp.ndarray:
    ''' Converts <value> to a CuPy array of shape <shape>. If <value> is scalar, the returned
        array is constant. If <value> is a CuPy or NumPy array with an equal amount of values
        as fit in <shape>, the returned array is the reshaped version of <value> to fit <shape>.
    '''
    is_scalar = True # Determine if <value> is scalar-like...
    if isinstance(value, (np.ndarray, cp.ndarray)):
        if value.size != 1: is_scalar = False

    if is_scalar: # ... and act accordingly
        return cp.ones(shape)*float(value)
    else:
        if value.size != math.prod(shape):
            raise ValueError(f"Specified value (shape {value.shape}) is incompatible with desired shape {shape}.")
        return cp.asarray(value).reshape(shape)


def is_significant(i: int, N: int, order=1) -> bool:
    ''' Returns True if <i> iterations is an 'important' milestone if there are <N> in total.
        Useful for verbose print statements in long simulations.
        @param i [int]: the index of the current iteration (starting at 0)
        @param N [int]: the total number of iterations
        @param order [int]: an example will help to explain this argument.
            If N=1000 and order=0, True will be returned if i = 0 or 999,
            while for order=1 any of i = 0, 99, 199, 299, ..., 999 yield True.
            Basically, at least <10**order> values of i (equally spaced) will yield True.
    '''
    if (i + 1) % 10**(math.floor(math.log10(N))-order) == 0:
        return True
    if i == 0 or i == N - 1:
        return True
    return False


def strided(a: cp.ndarray, W: int):
    ''' <a> is a 1D CuPy array, which gets expanded into 2D shape (a.size, W) where every row
        is a successively shifted version of the original <a>:
        the first row is [a[0], NaN, NaN, ...] with total length <W>.
        The second row is [a[1], a[0], NaN, ...], and this continues until <a> is exhausted.
        
        NOTE: the returned array is only a view, hence it is quite fast but care has to be
                taken when editing the array; e.g. editing one element directly changes the
                corresponding value in <a>, resulting in a whole diagonal being changed at once.
    '''
    a_ext = cp.concatenate((cp.full(W - 1, cp.nan), a))
    n = a_ext.strides[0]
    return striding.as_strided(a_ext[W - 1:], shape=(a.size, W), strides=(n, -n))

def R_squared(a, b):
    ''' Returns the R² metric between two 1D arrays <a> and <b> as defined in
        "Task Agnostic Metrics for Reservoir Computing" by Love et al.
    '''
    a, b = cp.asarray(a), cp.asarray(b)
    cov = cp.mean((a - cp.mean(a))*(b - cp.mean(b)))
    return cov**2/cp.var(a)/cp.var(b) # Same as cp.corrcoef(a, b)[0,1]**2, but faster


def check_repetition(arr, nx: int, ny: int):
    ''' Checks if <arr> is periodic with period <nx> along axis=1 and period <ny> along axis=0.
        If there are any further axes (axis=2, axis=3 etc.), the array is simply seen as a
        collection of 2D arrays (along axes 0 and 1), and the total result is only True if all
        of these are periodic with period <nx> and <ny>
    '''
    extra_dims = [1] * (len(arr.shape) - 2)
    max_y, max_x = arr.shape[:2]
    i, current_checking = 0, arr[:ny, :nx, ...]
    end_y, end_x = current_checking.shape[:2]
    while end_y < arr.shape[0] or end_x < arr.shape[1]:
        if i % 2: # Extend in x-direction (axis=1)
            current_checking = cp.tile(current_checking, (1, 2, *extra_dims))
            start_x = current_checking.shape[1]//2
            end_y, end_x = current_checking.shape[:2]
            if not cp.allclose(current_checking[:max_y,start_x:max_x, ...], arr[:end_y, start_x:end_x, ...]):
                return False
        else: # Extend in y-direction (axis=0)
            current_checking = cp.tile(current_checking, (2, 1, *extra_dims))
            start_y = current_checking.shape[0]//2
            end_y, end_x = current_checking.shape[:2]
            if not cp.allclose(current_checking[start_y:max_y, :max_x, ...], arr[start_y:end_y, :end_x, ...]):
                return False
        i += 1
    return True


def shell():
    ''' Pauses the program and opens an interactive shell where the user
        can enter statements or expressions to inspect the scope in which
        shell() was called. Write "exit()" to terminate this shell.
        Using Ctrl+C will stop the entire program, not just this function
        (this is due to a bug in the scipy library).
    '''
    try:
        caller = inspect.getframeinfo(inspect.stack()[1][0])

        print('-'*80)
        print(f'Opening an interactive shell in the current scope')
        print(f'(i.e. {caller.filename}:{caller.lineno}-{caller.function}).')
        print(f'Call "exit" to stop this interactive shell.')
        print(f'Warning: Ctrl+C will stop the program entirely, not just this shell, so take care which commands you run.')
    except:
        pass # Just to make sure we don't break when logging
    try:
        InteractiveShellEmbed().mainloop(stack_depth=1)
    except (KeyboardInterrupt, SystemExit, EOFError):
        pass


def free_gpu_memory():
    ''' Garbage-collects unused memory on the currently active CUDA device. '''
    cp.get_default_memory_pool().free_all_blocks()

def get_gpu_memory():
    ''' @return [dict]: keys "free" and "total" memory (in bytes) of the currently active CUDA device. '''
    free, total = cp.cuda.Device().mem_info # mem_info is a tuple: (free, total) memory in bytes
    return {"free": free, "total": total}

def readable_bytes(N):
    if N < 1024: return f"{N:.0f} B"
    i = int(math.floor(math.log(N, 1024)))
    number = N / 1024**i
    return f"{number:.2f} {('B', 'KiB', 'MiB', 'GiB', 'TiB')[i]}"


# TODO: just use a class to do all this, not a bunch of different similarly-named functions smh
def combine_json(df: pd.DataFrame, constants: dict = None, metadata: dict = None):
    ''' Converts the Pandas dataframe <df> to JSON format with appropriate metadata.
        @param df [pandas.DataFrame]: the dataframe to be saved.
        @param constants [dict] ({}): used to store constants such that they needn't be
            repeated in every row of the dataframe, e.g. cell size, temperature...
            These are stored under the "constants" object in the json file. 
        @param metadata [dict] ({}): any elements present in this dictionary will overwrite
            their default values as generated automatically to be put under the name "metadata".
        @return (dict): a dictionary containing the (meta)data.

        The JSON-like dictionary contains the following three objects:
        - "constants", with names and values of parameters which remained constant throughout the simulation.
        - "data", which stores a JSON 'table' representation of the Pandas dataframe containing the actual data.
        - "metadata", with the following values: 
            - "author": name of the author (default: login name of user on the computer)
            - "creator": path of the __main__ module in the session
            - "datetime": a string representing the time in yyyymmddHHMMSS format
            - "description": a (small) description of what the data represents
            - "GPU": an object containing some information about the used NVIDIA GPU, more specifically with
                values "name", " compute_cap", " driver_version", " memory.total [MiB]", " timestamp", " uuid".
            - "savepath": the full pathname where this function saved the JSON object to a file.
            - "simulator": "Hotspin", just for clarity
    ''' # TODO: add information about which function/class generated the data, and SimParams
    df_dict = json.loads(df.to_json(orient='split', index=False))
    if constants is None: constants = {}

    if metadata is None: metadata = {}
    metadata.setdefault('datetime', datetime.datetime.now().strftime(r"%Y%m%d%H%M%S"))
    metadata.setdefault('author', getpass.getuser())
    try: 
        creator_info = os.path.abspath(str(sys.modules['__main__'].__file__))
        if 'hotspin' in creator_info:
            creator_info = '...\\' + creator_info[len(creator_info.split('hotspin')[0]):]
    except:
        creator_info = ''
    metadata.setdefault('creator', creator_info)
    metadata.setdefault('simulator', "Hotspin")
    metadata.setdefault('description', "No custom description available.")
    try:
        gpu_info = json.loads(pd.read_csv(io.StringIO(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name,compute_cap,driver_version,gpu_uuid,memory.total,timestamp", "--format=csv"],
            encoding='utf-8').strip())).to_json(orient='table', index=False))['data']
    except:
        gpu_info = {}
    metadata.setdefault('GPU', gpu_info)

    return {'metadata': metadata, 'constants': constants, 'data': df_dict}


def save_json(total_dict: dict, path: str = None, name: str = None, timestamp=True):
    ''' Saves the dictionary <total_dict> as a JSON file in the directory <path> with name <name>.
        The full path of the saved file is "<path>/<name>_<yyyymmddHHMMSS>.json".
        is generated by `combine_json(df, constants=constants, metadata=metadata)`.
        Parameters for this saving function in particular are:
        @param total_dict [dict]: should be a dictionary as generated by `combine_json()`.
        @param name [str] ('hotspin_simulation'): this text is used as the start of the filename.
            Additionally, a timestamp is added to this if <timestamp> is True.
        @param path [str] ('hotspin_results'): the directory to create the .json file in.
        @return (str): the absolute path where the JSON file was saved.
    '''
    if path is None: path = 'hotspin_results'
    if name is None: name = 'hotspin_simulation'

    if timestamp:
        timestr = '_' + (timestamp if isinstance(timestamp, str) else total_dict['metadata']['datetime'])
    else:
        timestr = ''
    
    filename = name + timestr + '.json'
    fullpath = os.path.abspath(os.path.join(path, filename))
    os.makedirs(path, exist_ok=True)
    with open(fullpath, 'w') as outfile:
        json.dump(total_dict, outfile, indent="\t", cls=_CompactJSONEncoder)
    return fullpath


class _CompactJSONEncoder(json.JSONEncoder):
    """ A JSON encoder for pretty-printing, but with lowest-level lists kept on single lines. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """ Encode JSON object <o> with respect to single line lists. """
        if isinstance(o, (np.ndarray, cp.ndarray)):
            o = o.tolist()
        if isinstance(o, (list, tuple)):
            if self._is_single_line_list(o):
                return "[" + ", ".join(json.dumps(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
        elif isinstance(o, dict):
            if len(o) > 0:
                self.indentation_level += 1
                output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
                self.indentation_level -= 1
                return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            else:
                return "{ }"
        else:
            return json.dumps(o)

    def _is_single_line_list(self, o):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, (list, tuple, dict)) for el in o) # and len(o) <= 2 and len(str(o)) - 2 <= 60

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, str):
            return self.indent * self.indentation_level
        elif isinstance(self.indent, int):
            return " " * self.indentation_level * self.indent
    
    def iterencode(self, o, **kwargs):
        """ Required to also work with `json.dump`. """
        return self.encode(o)


def read_json(JSON):
    """ Reads a JSON-parseable object containing previously generated Hotspin data, and returns its
        contents as a dictionary, with the "data" attribute readily converted to a Pandas dataframe.
        @param JSON: a JSON-parseable object, can be any of: file-like object, JSON string,
            or a string representing a file path.
    """
    if isinstance(JSON, dict):
        total_dict = JSON
    elif isinstance(JSON, str): # EAFP ;)
        try: # See if <JSON> is actually a JSON-format string
            total_dict = json.loads(JSON)
        except json.JSONDecodeError: # Nope, but EAFP ;)
            try: # See if <JSON> is actually a file-like object
                total_dict = json.load(JSON)
            except: # See if <JSON> is actually a string representing a file path
                with open(JSON, 'r') as infile: total_dict = json.load(infile)
    else:
        raise ValueError('Could not parse JSON-like object correctly.')

    df = pd.read_json(json.dumps(total_dict['data']), orient='split')
    total_dict['data'] = df
    return total_dict


def combine_all(container: str | Iterable):
    ''' Combines all the JSON data in <container>, which can either be a:
        - string representing a path to a directory containing many similar .json files
        - iterable containing many dictionaries, each representing a bunch of similar data
    '''
    if isinstance(container, str): # Then it is a path to a directory containing several .json files
        all_jsons = [read_json(os.path.join(container, path)) for path in os.listdir(container) if path.endswith('.json')]
    else: # Then it is an iterable containing many dictionaries representing the data already
        all_jsons = container
    dataframes, constants, metadata = [], {}, {}
    for json_i in all_jsons:
        dataframes.append(json_i["data"])
        constants |= json_i["constants"]
        metadata |= json_i["metadata"]

    df: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
    for column in df.columns: # Remove 'constants' which are actually not constant
        if column in constants.keys():
            del constants[column]

    return combine_json(df, constants=constants, metadata=metadata)


def full_obj_name(obj):
    klass = type(obj)
    if hasattr(klass, "__module__"):
        return f'{klass.__module__}.{klass.__qualname__}'
    else:
        return klass.__qualname__


if __name__ == "__main__":
    full_json = combine_json(pd.DataFrame({"H_range": cp.arange(15).get(), "m": (cp.arange(15)**2).get()}))
    fullpath = save_json(full_json, name='This is just a test. Do not panic.')
    print(read_json(fullpath))
