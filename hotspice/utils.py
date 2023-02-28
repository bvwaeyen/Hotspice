import colorama
colorama.init()
import getpass
import inspect
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import threading
import warnings

import numpy as np
import pandas as pd

from datetime import datetime
from IPython.terminal.embed import InteractiveShellEmbed
from textwrap import dedent, indent
from typing import Callable, Iterable, Literal, TypeVar

from . import config
if config.USE_GPU:
    import cupy as cp # Only ever try to import cupy if USE_GPU, otherwise we get premature ImportErrors on non-CUDA devices
    import cupy as xp
    import cupy.lib.stride_tricks as striding
else:
    import numpy as cp # We need cp to be defined, so use numpy to be that placeholder as it is the closest module
    import numpy as xp
    import numpy.lib.stride_tricks as striding


## ARRAY MANIPULATIONS/OPERATIONS
def check_repetition(arr, nx: int, ny: int):
    """ Checks if <arr> is periodic with period <nx> along axis=1 and period <ny> along axis=0.
        If there are any further axes (axis=2, axis=3 etc.), the array is simply seen as a
        collection of 2D arrays (along axes 0 and 1), and the total result is only True if all
        of these are periodic with period <nx> and <ny>
    """
    extra_dims = [1] * (len(arr.shape) - 2)
    max_y, max_x = arr.shape[:2]
    i, current_checking = 0, arr[:ny, :nx, ...]
    end_y, end_x = current_checking.shape[:2]
    while end_y < arr.shape[0] or end_x < arr.shape[1]:
        if i % 2: # Extend in x-direction (axis=1)
            current_checking = xp.tile(current_checking, (1, 2, *extra_dims))
            start_x = current_checking.shape[1]//2
            end_y, end_x = current_checking.shape[:2]
            if not xp.allclose(current_checking[:max_y,start_x:max_x, ...], arr[:end_y, start_x:end_x, ...]):
                return False
        else: # Extend in y-direction (axis=0)
            current_checking = xp.tile(current_checking, (2, 1, *extra_dims))
            start_y = current_checking.shape[0]//2
            end_y, end_x = current_checking.shape[:2]
            if not xp.allclose(current_checking[start_y:max_y, :max_x, ...], arr[start_y:end_y, :end_x, ...]):
                return False
        i += 1
    return True

def mirror4(arr, /, *, negativex=False, negativey=False):
    """ Mirrors the 2D array <arr> along some of the edges, in such a manner that the
        original element at [0,0] ends up in the middle of the returned array.
        Hence, if <arr> has shape (a, b), the returned array has shape (2*a - 1, 2*b - 1).
    """
    ny, nx = arr.shape
    arr4 = xp.zeros((2*ny-1, 2*nx-1))
    x_sign = -1 if negativex else 1
    y_sign = -1 if negativey else 1
    arr4[ny-1:, nx-1:] = arr
    arr4[ny-1:, nx-1::-1] = x_sign*arr
    arr4[ny-1::-1, nx-1:] = y_sign*arr
    arr4[ny-1::-1, nx-1::-1] = x_sign*y_sign*arr
    return arr4

def R_squared(a, b):
    """ Returns the R² metric between two 1D arrays <a> and <b> as defined in
        "Task Agnostic Metrics for Reservoir Computing" by Love et al.
    """
    if (var_a := xp.var(a)) == 0 or (var_b := xp.var(b)) == 0: return 0
    cov = xp.mean((a - xp.mean(a))*(b - xp.mean(b)))
    return cov**2/var_a/var_b # Same as xp.corrcoef(a, b)[0,1]**2, but faster

def strided(a: xp.ndarray, W: int):
    """ <a> is a 1D array, which gets expanded into 2D shape (a.size, W) where every row
        is a successively shifted version of the original <a>:
        the first row is [a[0], NaN, NaN, ...] with total length <W>.
        The second row is [a[1], a[0], NaN, ...], and this continues until <a> is exhausted.
        
        NOTE: the returned array is only a view, hence it is quite fast but care has to be
                taken when editing the array; e.g. editing one element directly changes the
                corresponding value in <a>, resulting in a whole diagonal being changed at once.
    """
    a_ext = xp.concatenate((xp.full(W - 1, xp.nan), a))
    n = a_ext.strides[0]
    return striding.as_strided(a_ext[W - 1:], shape=(a.size, W), strides=(n, -n))

def lower_than(x, y, rtol=1e-10):
    """ Returns False if <x> is very close (within relative tolerance <rtol>) or greater than <y>. """
    return x < y and not xp.isclose(x, y, atol=0, rtol=rtol)


## EASE-OF-USE UTILITIES (e.g. for printing)
def human_sort(text):
    """ To sort a <list> of strings in human order, use <list>.sort(key=hotspice.utils.human_sort).
        Human order means that if there are numbers in the strings, they are treated as numbers,
        such that e.g. 10 will come after 2, which is not the case with a naive sort.
    """
    def atoi(text): return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def is_significant(i: int, N: int, order: float=1) -> bool:
    """ Returns True if <i> iterations is an 'important' milestone if there are <N> in total.
        Useful for verbose print statements in long simulations.
        @param i [int]: the index of the current iteration (starting at 0)
        @param N [int]: the total number of iterations
        @param order [float]: strictly speaking this can be any float, but integers work best.
            Basically, approximately <10**order> values of i (equally spaced) will yield True.
            An example can be useful to illustrate the behavior of this parameter:
                If N=1000 and order=0, True will be returned if i = 0 or 999,
                while for order=1 any of i = 0, 99, 199, 299, ..., 999 yield True.
                A float example: for order=0.6 only i = 0, 251, 502, 753, 999 yield True.
    """
    if (i + 1) % 10**(math.floor(math.log10(N))-order) < 1:
        return True
    if i == 0 or i == N - 1:
        return True
    return False

def readable_bytes(N: int):
    if not isinstance(N, int): raise ValueError("Number of bytes must be an integer.")
    if N < 1024: return f"{N:.0f} B"
    i = int(math.floor(math.log(N, 1024)))
    number = N/(1024**i)
    return f"{number:.2f} {('B', 'KiB', 'MiB', 'GiB', 'TiB')[i]}"

SIprefix_to_magnitude = {'f': -15, 'p': -12, 'n': -9, 'µ': -6, 'm': -3, 'c': -2, 'd': -1, '': 0, 'da': 1, 'h': 2, 'k': 3, 'M': 6, 'G': 9, 'T': 12}
def SIprefix_to_mul(unit: Literal['f', 'p', 'n', 'µ', 'm', 'c', 'd', '', 'da', 'h', 'k', 'M', 'G', 'T']):
    return 10**SIprefix_to_magnitude[unit]

magnitude_to_SIprefix = {v: k for k, v in SIprefix_to_magnitude.items()}
def appropriate_SIprefix(n: float|np.ndarray|xp.ndarray, unit_prefix: Literal['f', 'p', 'n', 'µ', 'm', 'c', 'd', '', 'da', 'h', 'k', 'M', 'G', 'T']=''):
    """ Converts <n> (which already has SI prefix <unit_prefix> for whatever unit it is in)
        to a reasonable number with a new SI prefix. Returns a tuple with (the new scalar values, the new SI prefix).
    """
    value = n.min() if isinstance(n, (np.ndarray, xp.ndarray)) else n # To avoid excessive commas, we take min()
    nearest_magnitude = round(math.log10(value)) + SIprefix_to_magnitude[unit_prefix]
    used_magnitude = -100 # placeholder
    for supported_magnitude in sorted(magnitude_to_SIprefix.keys()):
        if supported_magnitude <= nearest_magnitude: used_magnitude = supported_magnitude
    return (n/10**used_magnitude, SIprefix_to_magnitude[used_magnitude])

def shell():
    """ Pauses the program and opens an interactive shell where the user
        can enter statements or expressions to inspect the scope in which
        shell() was called. Write "exit()" to terminate this shell.
        Using Ctrl+C will stop the entire program, not just this function
        (this is due to a bug in the scipy library).
    """
    try:
        caller = inspect.getframeinfo(inspect.stack()[1][0])

        print("-"*80)
        print(f"Opening an interactive shell in the current scope")
        print(f"(i.e. {caller.filename}:{caller.lineno}-{caller.function}).")
        print(f"Call 'exit' to stop this interactive shell.")
        print(f"Warning: Ctrl+C will stop the program entirely, not just this shell, so take care which commands you run.")
    except Exception:
        pass # Just to make sure we don't break when logging
    try:
        InteractiveShellEmbed().mainloop(stack_depth=1)
    except (KeyboardInterrupt, SystemExit, EOFError):
        pass


## CONVERSION
Field = TypeVar("Field", int, float, list, np.ndarray, xp.ndarray) # Every type that can be parsed by as_2D_array()
def as_2D_array(value: Field, shape: tuple) -> xp.ndarray:
    """ Converts <value> to a 2D array of shape <shape>. If <value> is scalar, the returned
        array is constant. If <value> is a CuPy or NumPy array with an equal amount of values
        as fit in <shape>, the returned array is the reshaped version of <value> to fit <shape>.
        Either a CuPy or NumPy array is returned, depending on config.USE_GPU.
    """
    is_scalar = True # Determine if <value> is scalar-like...
    if isinstance(value, list):
        try:
            value = xp.asarray(value, dtype=float)
        except ValueError:
            raise ValueError("List of lists could not be converted to rectangular float64 array.")
    if isinstance(value, (np.ndarray, xp.ndarray)):
        if value.size != 1: is_scalar = False

    if is_scalar: # ... and act accordingly
        return xp.ones(shape, dtype=float)*float(value)
    else:
        if value.size != math.prod(shape):
            raise ValueError(f"Incorrect shape: passed array of shape {value.shape} is not of desired shape {shape}.")
        return xp.asarray(value).reshape(shape)

def asnumpy(array: xp.ndarray) -> np.ndarray:
    """ Converts the CuPy/NumPy <array> to a NumPy array, which is necessary for e.g. matplotlib. """
    if not config.USE_GPU: # Not very clean if-else statement but oh well
        return np.asarray(array) # Then CuPy doesn't exist, so no other options than to try np.asarray
    elif isinstance(array, cp.ndarray):
        return array.get()
    else:
        return np.asarray(array)

def eV_to_J(E: float, /):
    return E*1.602176634e-19
def J_to_eV(E: float, /):
    return E/1.602176634e-19

def filter_kwargs(kwargs: dict, func: Callable):
    return {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}


## VARIOUS INFORMATION
def full_obj_name(obj):
    klass = type(obj)
    if hasattr(klass, "__module__"):
        if klass.__module__ != "__main__":
            return f"{klass.__module__}.{klass.__qualname__}"
    return klass.__qualname__

def free_gpu_memory():
    """ Garbage-collects unused memory on the currently active CUDA device. """
    cp.get_default_memory_pool().free_all_blocks()

def get_gpu_memory():
    """ @return [dict]: keys 'free' and 'total' memory (in bytes) of the currently active CUDA device. """
    free, total = cp.cuda.Device().mem_info # mem_info is a tuple: (free, total) memory in bytes
    return {'free': readable_bytes(free), 'total': readable_bytes(total)}

def timestamp():
    """ @return [str]: the current time, in YYYYMMDDhhmmss format. """
    return datetime.utcnow().strftime(r"%Y%m%d%H%M%S")


## MULTI-GPU UTILITIES
def run_script(script_name, args=None):
    """ Runs the script <script_name> located in the hotspice/scripts directory.
        Any arguments for the script can be passed as a list to <args>.
    """
    script_name = script_name.strip()
    if not os.path.splitext(script_name)[1]: script_name += '.py'
    path = pathlib.Path(__file__).parent / "scripts" / script_name
    if not os.path.exists(path): raise FileNotFoundError(f"No script with name '{script_name}' was found.")
    command = ["python", str(path)] + list(args)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        warnings.warn(dedent(f"""
            {colorama.Fore.LIGHTRED_EX}The script {script_name} with arguments {args} could not be run successfully.
            See a possible error message above for more info. The full command used was:
            {colorama.Fore.LIGHTYELLOW_EX}{' '.join(command)}{colorama.Style.RESET_ALL}
        """), stacklevel=2)

def ParallelJobs(sweepscript_path, outdir=None, _ParallelJobs_script_name="ParallelJobs"):
    """ Convenient wrapper around run_script() for the ParallelJobs.py script in particular. """
    args = [sweepscript_path]
    if outdir is not None:
        args = ['-o', outdir] + args
    run_script(_ParallelJobs_script_name, args=args)

def log(message, device_id=None, style: Literal['issue', 'success', 'header'] = None, show_device=True):
    """ Can print <message> to console from subprocesses running on a specific GPU or thread.
        The <device_id> (currently used GPU/CPU core) is printed in front of the message, if <show_device> is True.
        <style> specifies the color of the message (default None=white), and can be any of:
            'issue' (red), 'success' (green), 'header' (blue).
    """
    if device_id is None:
        device_id = 0 if config.USE_GPU else config.DEVICE_ID
    if show_device:
        if config.USE_GPU:
            try:
                device = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")][device_id]
            except KeyError: # So CUDA_VISIBLE_DEVICES was not defined manually, that probably means they are all available
                device = device_id if device_id < cp.cuda.runtime.getDeviceCount() else np.nan
            text_device = f"{colorama.Fore.GREEN}[GPU{device}] "
        else:
            device = str(device_id) if device_id is not None else ''
            text_device = f"{colorama.Fore.GREEN}[CPU{device}] "
    else: text_device = ""

    color = {
        'issue': colorama.Fore.LIGHTRED_EX,
        'success': colorama.Fore.LIGHTGREEN_EX,
        'header': colorama.Fore.LIGHTBLUE_EX
    }.get(style, colorama.Style.RESET_ALL)

    text = text_device + f"{color}{message}{colorama.Style.RESET_ALL}"
    _rlock = threading.RLock()
    with _rlock: # Fix print to work with asynchronous queues on different GPUs, though this might not be entirely necessary
        print(text)


## STANDARDIZED WAY OF HANDLING DATA ACROSS HOTSPICE EXAMPLES
class Data: # TODO: make a get_column() function that returns (one or multiple) df column even if the requested column is actually a constant
    def __init__(self, df: pd.DataFrame, constants: dict = None, metadata: dict = None):
        """ Stores the Pandas dataframe <df> with appropriate metadata and optional constants.
            Constant columns in <df> are automatically moved to <constants>, and keys present
            in <constants> that are also columns in <df> are removed from <constants>.
            @param df [pandas.DataFrame]: the dataframe to be stored.
            @param constants [dict] ({}): used to store constants such that they needn't be
                repeated in every row of the dataframe, e.g. cell size, temperature...
                These constants can be scalars, strings or CuPy/NumPy arrays.
            @param metadata [dict] ({}): used to store additional information about the
                simulation, e.g. description, author, time... Some fields are automatically
                generated if they are not present in the dictionary passed to <metadata>.
                For more details, see the <self.metadata> docstring.
        """
        self.df = df
        self.metadata = metadata
        self.constants = constants

    @property
    def df(self): return self._df
    @df.setter
    def df(self, value: pd.DataFrame):
        """ The Pandas DataFrame containing the bulk of the data. """
        if not isinstance(value, pd.DataFrame):
            try: # Then assume <value> is a JSON-parseable object
                value = pd.read_json(json.dumps(value), orient='split') 
            except Exception: # So no DataFrame, and not JSON-parseable? Just stop this madness.
                raise ValueError("Could not parse DataFrame-like object correctly.")
        self._df = value
        self._check_consistency()
        assert isinstance(self._df, pd.DataFrame)

    @property
    def metadata(self): return self._metadata
    @metadata.setter
    def metadata(self, value: dict):
        """ All keys in <value> are stored without modification, and the following keys are
            automatically added if they are not provided in <value>:
            - 'author': name of the author (default: login name of user on the computer)
            - 'creator': main file responsible for creating the data (default: path of the __main__ module in the session)
            - 'datetime': a string representing the UTC time in "yyyymmddHHMMSS" format
            - 'description': a (small) description of what the data represents
            - 'GPU': a list of dicts representing the NVIDIA GPUs used for the simulation. Each dict contains
                keys 'name', 'compute_cap', 'driver_version', 'memory.total [MiB]', 'timestamp', 'uuid'.
            - 'simulator': "Hotspice", just for clarity. Can include version number when applicable.
        """
        if value is None: value = {}
        if not isinstance(value, dict): raise ValueError("Metadata must be provided as a dictionary.")

        value.setdefault('datetime', timestamp())
        value.setdefault('author', getpass.getuser())
        try:
            creator_info = os.path.abspath(str(sys.modules['__main__'].__file__))
            if 'hotspice' in creator_info:
                creator_info = "...\\" + creator_info[len(creator_info.split('hotspice')[0]):]
        except Exception:
            creator_info = ''
        value.setdefault('creator', creator_info)
        value.setdefault('simulator', "Hotspice")
        value.setdefault('description', "No custom description available.")
        value['description'] = dedent(value['description']).strip()
        try:
            gpu_info = json.loads(pd.read_csv(io.StringIO(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=gpu_name,compute_cap,driver_version,gpu_uuid,memory.total,timestamp", "--format=csv"],
                encoding='utf-8').strip())).to_json(orient='table', index=False))['data']
            gpu_info = [{key.strip(): (value.strip() if isinstance(value, str) else value) for key, value in d.items()} for d in gpu_info] # Remove spaces around key and value text
        except Exception:
            gpu_info = []
        value.setdefault('GPU', gpu_info)
        value.setdefault('hotspice_config', config.get_dict())

        self._metadata = value
        assert isinstance(self._metadata, dict)

    @property
    def constants(self): return self._constants
    @constants.setter
    def constants(self, value: dict):
        """ Names and values of parameters which are constant throughout all entries in <self.df>. """
        if value is None: value = {}
        if not isinstance(value, dict): raise ValueError("Constants must be provided as a dictionary.")
        self._constants = value
        self._check_consistency()
        assert isinstance(self._constants, dict)

    def _check_consistency(self):
        """ Moves constant columns in <self.df> to <self.constants>, and removes keys
            from <self.constants> that are also columns in <self.df>, to prevent ambiguous duplicates.
            Checks if all keys in <self.constants> and <self.metadata> are strings;
        """
        if hasattr(self, 'constants') and hasattr(self, 'df'):
            # Move all constant columns in self.df to self.constants
            for column in self.df:
                if self.df[column].dtype == 'O': continue # We don't meddle with 'constant' objects of general type in the df.
                is_constant = ~(self.df[column] != self.df[column].iloc[0]).any()
                if is_constant:
                    self._constants[column] = self.df[column].iloc[0]
                    self._df = self._df.drop(columns=column)
            # Remove constants from self.constants that are also column labels in self.df
            for column in self.df.columns:
                if column in self.constants.keys():
                    del self._constants[column]

        if hasattr(self, 'constants'):
            for key in self.constants.keys():
                if not isinstance(key, str): raise KeyError("Data.constants keys must be of type string.")
        if hasattr(self, 'metadata'):
            for key in self.metadata.keys():
                if not isinstance(key, str): raise KeyError("Data.metadata keys must be of type string.")

    def save(self, dir: str = None, name: str = None, *, timestamp=True): # TODO: DO THE CONVERSION OF COLUMNS TO CONSTANTS ETC. HERE AND ONLY HERE. EXPAND THEM UPON OPENING SUCH FILES. THIS MEMORY-EFFICIENCY-IMPROVEMENT IS CAUSING TOO MUCH TROUBLE EVERYWHERE IF WE DO IT ON-THE-FLY IN THE PROPERTIES
        # TODO: do we even need this conversion at all? It has only caused more headaches than it has saved memory/storage
        """ Saves the currently stored data (<self.df>, <self.constants> and <self.metadata>)
            to a JSON file, with path "<dir>/<name>_<yyyymmddHHMMSS>.json". The automatically
            added timestamp in the filename can be disabled by passing <timestamp=False>.
            The JSON file contains three top-level objects: 'metadata', 'constants' and 'data',
            where 'data' stores a JSON 'table' representation of the Pandas DataFrame <self.df>.

            @param dir [str] ('hotspice_results'): the directory to create the .json file in.
            @param name [str] ('hotspice_simulation'): this text is used as the start of the filename.
                This should not include an extension or timestamp, as these are generated automatically.
            @param timestamp [bool|str] (True): if true, a timestamp is added to the filename. A string
                can be passed to override the auto-generated timestamp. If False, no timestamp is added.
            @return (str): the absolute path of the saved JSON file.
        """
        if dir is None: dir = "hotspice_results"
        if name is None: name = "hotspice_simulation"

        total_dict = {
            'metadata': self.metadata,
            'constants': self.constants,
            'data': json.loads(self.df.to_json(orient='split', index=False, default_handler=str)) # DataFrame -> JSON to save, with unknown dtypes converted to str
        }

        if timestamp:
            timestr = "_" + (timestamp if isinstance(timestamp, str) else self.metadata['datetime'])
        else:
            timestr = ""
        filename = name + timestr + ".json"
        fullpath = os.path.abspath(os.path.join(dir, filename))
        os.makedirs(dir, exist_ok=True)
        with open(fullpath, 'w') as outfile:
            json.dump(total_dict, outfile, indent="\t", cls=_CompactJSONEncoder)
        return fullpath

    def mimic(self, df): # WARN: this could go wrong due to aliasing, but we will see
        return Data(df, constants=self.constants, metadata=self.metadata)

    @staticmethod
    def load(JSON): # TODO: improve the error handling here so a caller of load() can see what is going on
        """ Reads a JSON-parseable object containing previously generated Hotspice data, and returns its
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
                except Exception: # See if <JSON> is actually a string representing a file path
                    with open(JSON, 'r') as infile: JSONdict = json.load(infile)
        else:
            raise ValueError("Could not parse JSON-like object correctly.")

        df = pd.read_json(json.dumps(JSONdict['data']), orient='split') # JSON -> DataFrame to load
        return Data(df, constants=JSONdict['constants'], metadata=JSONdict['metadata'])

    @staticmethod
    def load_collection(collection: str|Iterable, verbose=False):
        """ Combines all the JSON data in <collection>, which can either be a:
            - string representing a directory path, containing many similar .json files
            - iterable containing many Data objects, each representing a bunch of similar data
            The idea is that these different chunks of data have a certain overlap, e.g. that
            most of their constants/columns are the same, with only a few varying between them.
            NOTE: arrays as constants are ignored in the comparison.
        """
        if isinstance(collection, str): # Then it is a path to a directory containing several .json files
            files = [os.path.join(collection, path) for path in os.listdir(collection) if path.endswith('.json')]
            collection = []
            for i, path in enumerate(files):
                if verbose and is_significant(i, n := len(files)): print(f"Reading data from directory... (file {i+1}/{n})")
                collection.append(Data.load(path))
        elif all(isinstance(i, Data) for i in collection): # Then it is an iterable containing many Data objects already
            collection = collection
        else:
            raise ValueError("Could not recognize <collection> as a bunch of data.")

        if verbose: print(f"Collecting results into a single Data object...")
        nonconstants, constants, metadata = set(), dict(), dict()
        # 1st sweep: to possibly find constants which are not constant throughout <collection>
        for i, data in enumerate(collection):
            nonconstants = nonconstants.union(data.df.columns) # df columns are never considered to be constant
            for key in data.constants:
                if key in constants: # Then this 'constant' was already encountered earlier
                    if constants[key] != data.constants[key]: # And it is not the same as earlier
                        nonconstants.add(key)
            constants |= data.constants # We have checked all <data.constants> now, so merge it with <constants> for the next iteration
            metadata |= data.metadata
        constants = {key: value for key, value in constants.items() if key not in nonconstants}

        # 2nd sweep: now we know which constants are not really constant, so we can properly deal with them
        dataframes = []
        for data in collection:
            df = data.df.copy() # To prevent changing the original data.df
            for const in data.constants:
                if (impostor := const) in nonconstants: # If const is not deemed constant ඞ, move it to the df
                    try:
                        df[impostor] = data.constants[impostor]
                    except ValueError as e:
                        pass # ValueError occurs when data.constants is an iterable
            dataframes.append(df)

        # And now combine everything into one big Data() object
        big_df = pd.concat(dataframes, ignore_index=True) # Concatenate all rows of dataframes
        if verbose: print('Collected data into a single Data object.')
        return Data(big_df, constants=constants, metadata=metadata)
    
    def __str__(self):
        c = "\n".join([f"{k} = {v}" for k, v in self.constants.items()])
        return dedent(
            f"""
            {'='*32}
            Data created by {self.metadata['author']} on {self.metadata['datetime']}.
            {self.metadata['description']}

            Constants:
            {indent(c, ' '*4)}

            Data:
            {indent(self.df.to_string(index=False), ' '*4)}
            {'='*32}
            """
        )

    def __getitem__(self, item: str):
        """ Returns the full column of <param_name>, even if it is in self.constants. Usage: data[item]. """
        if item in self.df:
            return self.df[item]
        if item in self.constants:
            return pd.Series([self.constants[item]]*len(self.df))
        raise ValueError(f"Unknown parameter name '{item}'.")


class _CompactJSONEncoder(json.JSONEncoder):
    """ A JSON encoder for pretty-printing, but with lowest-level lists kept on single lines. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """ Encode JSON object <o> with respect to single line lists. """
        if isinstance(o, (np.ndarray, xp.ndarray)):
            o = o.tolist()
        if isinstance(o, (list, tuple)):
            if self._is_single_line_list(o):
                return "[" + ", ".join(self.encode(el) for el in o) + "]"
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
        elif isinstance(o, (xp.int32, xp.int64)): # json.dumps can not handle numpy ints (floats are ok)
            return f"{int(o)}"
        elif (name := full_obj_name(o)).startswith('hotspice'): # Then it is some Hotspice-defined class, so ...
            return json.dumps(name) # use full obj name (e.g. hotspice.ASI.IP_Pinwheel etc.)
        else:
            try: return json.dumps(o)
            except Exception: return json.dumps(str(o)) # Otherwise just use string representation of whatever kind of object this might be

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


if __name__ == "__main__":
    def test_save():
        full_json = Data(pd.DataFrame({'H_range': np.arange(15), 'm': np.arange(15)**2}))
        fullpath = full_json.save(name="This is just a test. Do not panic.")
        print(Data.load(fullpath))
    test_save()

    def test_load_collection():
        constants1 = {'T': 400, 'nx': 200, 'ny':200}
        constants2 = {'T': 300, 'ny': 200, 'nx': 200}
        df1 = pd.DataFrame({'nx': [200, 250, 300], 'E_b': [6, 7, 8]})
        df2 = pd.DataFrame({'E_b': [3, 4, 5]})
        data1 = Data(df1, constants=constants1)
        data2 = Data(df2, constants=constants2)
        data = Data.load_collection([data1, data2])
        print(data.df)
        print(data.constants)
    test_load_collection()
