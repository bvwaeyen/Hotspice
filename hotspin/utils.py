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

def clean_indices(indices2D):
    ''' Converts a generic iterable <indices2D> into a standardized form for representing 2D indices.
        @param <indices2D> [iterable]: may contain at most 2 dimensions of size > 1, at least one of which
            has size == 2. It is this size-2 dimension which will become the primary dim of the returned array.
        @return [tuple(2)cp.array(2xN)]: A tuple containing exactly 2 CuPy arrays of length N. The first array
            represents y-indices, the second represents x-indices.
    '''
    indices2D = cp.atleast_2d(cp.asarray(indices2D).squeeze()) if indices2D is not None else cp.empty((2,0))
    if len(indices2D.shape) > 2: raise ValueError("An array with more than 2 non-empty dimensions can not be used to represent a list of indices.")
    if not cp.any(cp.asarray(indices2D.shape) == 2): raise ValueError("The list of indices has an incorrect shape. At least one dimension should have length 2.")
    match indices2D.shape:
        case (2, _):
            return tuple(indices2D)
        case (_, 2):
            return tuple(indices2D.T)
        case (_, ): # 1D
            return tuple(indices2D.reshape(2, -1))
        case _: # wildcard
            raise ValueError("Structure of attribute <indices2D> could not be recognized: not 2D with at least one size-2 dimension.")

def clean_index(index2D):
    ''' Does the same thing as clean_indices, but optimized for a single 2D index. '''
    return tuple(cp.asarray(index2D).reshape(2))

def J_to_eV(E: float, /):
    return E/1.602176634e-19
def eV_to_J(E: float, /):
    return E*1.602176634e-19

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


def is_significant(i: int, N: int, order: float=1) -> bool:
    ''' Returns True if <i> iterations is an 'important' milestone if there are <N> in total.
        Useful for verbose print statements in long simulations.
        @param i [int]: the index of the current iteration (starting at 0)
        @param N [int]: the total number of iterations
        @param order [float]: strictly speaking this can be any float, but integers work best.
            Basically, approximately <10**order> values of i (equally spaced) will yield True.
            An example can be useful to illustrate the behavior of this parameter:
                If N=1000 and order=0, True will be returned if i = 0 or 999,
                while for order=1 any of i = 0, 99, 199, 299, ..., 999 yield True.
                A float example: for order=0.6 only i = 0, 251, 502, 753, 999 yield True.
    '''
    if (i + 1) % 10**(math.floor(math.log10(N))-order) < 1:
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


class Data:
    def __init__(self, df: pd.DataFrame, constants: dict = None, metadata: dict = None):
        ''' Stores the Pandas dataframe <df> with appropriate metadata and optional constants.
            @param df [pandas.DataFrame]: the dataframe to be stored.
            @param constants [dict] ({}): used to store constants such that they needn't be
                repeated in every row of the dataframe, e.g. cell size, temperature...
                These constants can also be arrays of JSON-convertable objects, e.g. scalars or strings.
                Both CuPy and NumPy arrays are supported for this purpose.
            @param metadata [dict] ({}): any elements present in this dictionary will overwrite
                their default values as generated automatically to be put under the name "metadata".
                Additional fields can also be provided without issue, as long as they are a scalar or string.

            # When saved as a JSON file, the following three top-level objects are available:
            # - "constants", with names and values of parameters which remained constant throughout the simulation.
            # - "data", which stores a JSON 'table' representation of the Pandas dataframe containing the actual data.
            # - "metadata", with the following values: 
            #     - "author": name of the author (default: login name of user on the computer)
            #     - "creator": path of the __main__ module in the session
            #     - "datetime": a string representing the time in yyyymmddHHMMSS format
            #     - "description": a (small) description of what the data represents
            #     - "GPU": an object containing some information about the used NVIDIA GPU, more specifically with
            #         values "name", " compute_cap", " driver_version", " memory.total [MiB]", " timestamp", " uuid".
            #     - "savepath": the full pathname where this function saved the JSON object to a file.
            #     - "simulator": "Hotspin", just for clarity
        '''
        self.df = df
        self.metadata = metadata
        self.constants = constants
    
    @property
    def df(self): return self._df
    @df.setter
    def df(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            try: # Then assume <value> is a JSON-parseable object
                value = pd.read_json(json.dumps(value), orient='split') 
            except: # So no DataFrame, and not JSON-parseable? Just stop this madness.
                raise ValueError('Could not parse DataFrame-like object correctly.')
        self._df = value
        assert isinstance(self._df, pd.DataFrame)
    
    @property
    def metadata(self): return self._metadata
    @metadata.setter
    def metadata(self, value: dict): # TODO: add information about which function/class generated the data, and SimParams
        if value is None: value = {}
        if not isinstance(value, dict): raise ValueError('Metadata must be provided as a dictionary.')
        
        value.setdefault('datetime', datetime.datetime.now().strftime(r"%Y%m%d%H%M%S"))
        value.setdefault('author', getpass.getuser())
        try:
            creator_info = os.path.abspath(str(sys.modules['__main__'].__file__))
            if 'hotspin' in creator_info:
                creator_info = '...\\' + creator_info[len(creator_info.split('hotspin')[0]):]
        except:
            creator_info = ''
        value.setdefault('creator', creator_info)
        value.setdefault('simulator', "Hotspin")
        value.setdefault('description', "No custom description available.")
        try:
            gpu_info = json.loads(pd.read_csv(io.StringIO(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=gpu_name,compute_cap,driver_version,gpu_uuid,memory.total,timestamp", "--format=csv"],
                encoding='utf-8').strip())).to_json(orient='table', index=False))['data']
        except:
            gpu_info = {}
        value.setdefault('GPU', gpu_info)

        self._metadata = value
        assert isinstance(self._metadata, dict)
    
    @property
    def constants(self): return self._constants
    @constants.setter
    def constants(self, value: dict):
        if value is None: value = {}
        if not isinstance(value, dict): raise ValueError('Constants must be provided as a dictionary.')
        self._constants = value
        assert isinstance(self._constants, dict)
    
    def save(self, dir: str = None, name: str = None, timestamp=True):
        ''' Saves the currently stored data (<self.df>, <self.constants> and <self.metadata>)
            to a JSON file, with path "<dir>/<name>_<yyyymmddHHMMSS>.json". The automatically
            added timestamp in the filename can be disabled by passing <timestamp=False>.

            @param dir [str] ('hotspin_results'): the directory to create the .json file in.
            @param name [str] ('hotspin_simulation'): this text is used as the start of the filename.
            @param timestamp [bool|str] (True): if true, a timestamp is added to the filename. A string
                can be passed to override the auto-generated timestamp. If False, no timestamp is added.
            @return (str): the absolute path where the JSON file was saved.
        '''
        if dir is None: dir = 'hotspin_results'
        if name is None: name = 'hotspin_simulation'

        total_dict = {
            'metadata': self.metadata,
            'constants': self.constants,
            'data': json.loads(self.df.to_json(orient='split', index=False)) # DataFrame -> JSON to save
        }

        if timestamp:
            timestr = '_' + (timestamp if isinstance(timestamp, str) else self.metadata['datetime'])
        else:
            timestr = ''
        
        filename = name + timestr + '.json'
        fullpath = os.path.abspath(os.path.join(dir, filename))
        os.makedirs(dir, exist_ok=True)
        with open(fullpath, 'w') as outfile:
            json.dump(total_dict, outfile, indent="\t", cls=_CompactJSONEncoder)
        return fullpath
    
    @staticmethod
    def load(JSON):
        """ Reads a JSON-parseable object containing previously generated Hotspin data, and returns its
            contents as Data object.
            @param JSON: a parseable object resembling JSON data, can be any of: 
                Data() object, valid dict(), JSON string, file-like object, string representing a file path.
            @return [Data]: a Data object containing the information from <JSON>.
        """
        if isinstance(JSON, Data): return JSON

        if isinstance(JSON, dict):
            if {'data', 'constants', 'metadata'}.issubset(JSON): # Test if those keys exist (can also use <= operator)
                JSONdict = JSON
            else:
                raise ValueError("The dict passed to Data.load() does not contain the required keys.")

        elif isinstance(JSON, str): # EAFP ;)
            try: # See if <JSON> is actually a JSON-format string
                JSONdict = json.loads(JSON)
            except json.JSONDecodeError: # Nope, but EAFP ;)
                try: # See if <JSON> is actually a file-like object
                    JSONdict = json.load(JSON)
                except: # See if <JSON> is actually a string representing a file path
                    with open(JSON, 'r') as infile: JSONdict = json.load(infile)
        else:
            raise ValueError('Could not parse JSON-like object correctly.')

        df = pd.read_json(json.dumps(JSONdict['data']), orient='split') # JSON -> DataFrame to load
        return Data(df, constants=JSONdict['constants'], metadata=JSONdict['metadata'])


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


def combine_all(container: str|Iterable[Data]):
    ''' Combines all the JSON data in <container>, which can either be a:
        - string representing a path to a directory containing many similar .json files
        - iterable containing many Data objects, each representing a bunch of similar data
    '''
    if isinstance(container, str): # Then it is a path to a directory containing several .json files
        all_data = [Data.load(os.path.join(container, path)) for path in os.listdir(container) if path.endswith('.json')]
    elif all(isinstance(i, Data) for i in all_data): # Then it is an iterable containing many Data objects already
        all_data = container
    dataframes, constants, metadata = [], {}, {}
    data: Data
    for data in all_data:
        dataframes.append(data.df)
        constants |= data.constants
        metadata |= data.metadata

    df: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
    for column in df.columns: # Remove 'constants' which are actually not constant
        if column in constants.keys():
            del constants[column]

    return Data(df, constants=constants, metadata=metadata)


def full_obj_name(obj):
    klass = type(obj)
    if hasattr(klass, "__module__"):
        return f'{klass.__module__}.{klass.__qualname__}'
    else:
        return klass.__qualname__


if __name__ == "__main__":
    full_json = Data(pd.DataFrame({"H_range": cp.arange(15).get(), "m": (cp.arange(15)**2).get()}))
    fullpath = full_json.save(name='This is just a test. Do not panic.')
    print(Data.load(fullpath))
