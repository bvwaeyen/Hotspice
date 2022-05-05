import datetime
import getpass
import inspect
import io
import json
import os
import subprocess
import warnings

import cupy as cp
import cupy.lib.stride_tricks as striding
import pandas as pd

from IPython.terminal.embed import InteractiveShellEmbed


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


def strided(a, W):
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


def gpu_memory():
    ''' Returns a tuple (free memory, total memory) of the currently active CUDA device. '''
    return cp.cuda.Device().mem_info


def save_json(df: pd.DataFrame, name: str, path: str = None, constants: dict = None, metadata: dict = None):
    ''' Saves the Pandas dataframe <df> as a json file with appropriate metadata,
        in the directory specified by <path> with optional name <name>.
        @param df [pandas.DataFrame]: the dataframe to be saved.
        @param name [str]: this text is used as the start of the filename.
            Additionally, a timestamp is added to this.
        @param path [str] ('hotspin_results'): the directory to create the .json file in.
        @param constants [dict] ({}): used to store constants such that they needn't be
            repeated in every row of the dataframe, e.g. cell size, temperature...
            These are stored under the "constants" object in the json file. 
        @param metadata [dict] ({}): any elements present in this dictionary will overwrite
            their default values as generated automatically to be put under the name "metadata".
        @return (str): the absolute path where the .json file was saved.
    '''
    if path is None: path = 'hotspin_results'
    if constants is None: constants = {}

    df_dict = json.loads(df.to_json(orient='split', index=True))
    if metadata is None: metadata = {}
    metadata.setdefault('datetime', datetime.datetime.now().strftime(r"%Y%m%d%H%M%S"))
    metadata.setdefault('author', getpass.getuser())
    metadata.setdefault('simulator', "Hotspin")
    metadata.setdefault('description', "No further description was provided.")
    try:
        gpu_info = json.loads(pd.read_csv(io.StringIO(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name,compute_cap,driver_version,gpu_uuid,memory.total,timestamp", "--format=csv"],
            encoding='utf-8').strip())).to_json(orient='table', index=False))['data']
    except:
        gpu_info = {}
    metadata.setdefault('GPU', gpu_info)

    total_dict = {'metadata': metadata, 'constants': constants, 'data': df_dict}
    filename = name + '_' + metadata['datetime'] + '.json'
    fullpath = os.path.abspath(os.path.join(path, filename))
    os.makedirs(path, exist_ok=True)
    with open(fullpath, 'w') as outfile:
        json.dump(total_dict, outfile, indent="\t", cls=CompactJSONEncoder)
    return fullpath

    
class CompactJSONEncoder(json.JSONEncoder):
    """ A JSON Encoder to pretty-rpint while keeping lowest-level lists on single lines. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """ Encode JSON object *o* with respect to single line lists. """
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
        """ equired to also work with `json.dump`. """
        return self.encode(o)


def read_json(filename: str):
    """ Reads a .json file located at <filename> and returns its contents as a dictionary. """
    try: # See if what was passed as a filename was actually a json-format string
        total_dict = json.loads(filename)
    except json.JSONDecodeError: # Nope, it was just a filename
        with open(filename, 'r') as infile: total_dict = json.load(infile)

    df = pd.read_json(json.dumps(total_dict['data']), orient='split')
    total_dict['data'] = df
    return total_dict


def save_data(df: pd.DataFrame, save_path: str):
    ''' <save_path> is a full relative pathname, usually something like
        "results/<test_or_experiment_name>/<relevant_params=...>.pdf"
    '''
    # TODO: make this json-only so we can consistently save metadata, add standard procedure to add this metadata etc.

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    name, ext = os.path.splitext(save_path)
    try:
        match ext:
            case '.csv':
                df.to_csv(name + '.csv')
            case '.json':
                df.to_json(name + '.json')
            case _:
                warnings.warn(f'Saved pandas.DataFrame as .csv, while .{ext} was requested. Only csv is supported.')
    except PermissionError:
        warnings.warn(f'Could not save to {save_path}, probably because the file is opened somewhere else.', stacklevel=2)


if __name__ == "__main__":
    fullpath = save_json(pd.DataFrame({"H_range": cp.arange(15).get(), "m": (cp.arange(15)**2).get()}), 'This is just a test. Do not panic.')
    print(read_json(fullpath))
