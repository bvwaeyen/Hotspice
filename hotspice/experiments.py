import math
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from abc import ABC, abstractmethod
from matplotlib import cm, widgets
from scipy.spatial import distance
from textwrap import dedent
from typing import Any, Iterable

from .core import Energy, ExchangeEnergy, Magnets, DipolarEnergy, ZeemanEnergy
from .ASI import OOP_Square
from .io import Inputter, OutputReader, RandomBinaryDatastream, FieldInputter, PerpFieldInputter, RandomUniformDatastream, RegionalOutputReader
from .plottools import close_interactive, init_interactive, init_fonts, save_plot, show_m
from .utils import asnumpy, Data, filter_kwargs, full_obj_name, human_sort, is_significant, log, R_squared, strided
from . import config
if config.USE_GPU:
    import cupy as xp
else:
    import numpy as xp


class Experiment(ABC):
    def __init__(self, inputter: Inputter, outputreader: OutputReader, mm: Magnets):
        self.inputter = inputter
        self.outputreader = outputreader
        self.mm = mm
        self.outputreader.configure_for(self.mm)
        self.results = {} # General-purpose dict to store simulation results in
    
    @abstractmethod
    def run(self) -> None:
        """ Runs the entire experiment and records all the useful data. """
    
    @abstractmethod
    def calculate_all(self) -> None:
        """ (Re)calculates all the metrics in the <self.results> dict. """
    
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """ Creates a Pandas dataframe from the saved results of self.run(). """
    
    @abstractmethod
    def load_dataframe(self, df: pd.DataFrame) -> None:
        """ Loads the data from self.to_dataframe() to the current object.
            Might return the most important columns of data, but this is not required.
        """
    
    @staticmethod
    @abstractmethod
    def get_plot_metrics(self):
        """ Returns a dictionary with as many elements as there should be 2D or 1D plots
            in Sweep().plot(). Keys are names of these metrics, values are functions that
            take one argument (a hotspice.utils.Data object, which for this purpose is
            equivalent to a pandas.DataFrame), and returns a pandas.Series which is either
            a simple column from the Data, or a mathematical combination of several columns.
        """
        return {}


class Sweep(ABC):
    def __init__(self, groups: Iterable[tuple[str]] = None, **kwargs): # kwargs can be anything to specify the sweeping variables and their values.
        """ Sweep quacks like a generator yielding <Experiment>-like objects.
            The purpose of a Sweep is to reliably generate a sequence of <Experiment>s in a consistent order.
            @param groups [iterable[tuple[str]]] (None): a tuple of tuples of strings. Each tuple represents a group:
                groups are sweeped together. The strings inside those tuples represent the names of the
                variables that are being swept together. By sweeping together, it is meant that those
                variables move through their sweeping values simultaneously, instead of forming a
                hypercube through their values in which each pair is visited. We only visit the diagonal.
                Example: groups=(('nx', 'ny'), ('res_x', 'res_y')) will cause the variables <nx> and <ny>
                    to sweep through their values together, i.e. e.g. when <nx> is at its fourth value, also
                    <ny> will be at its fourth value.
            The additional arguments (kwargs) passed to this are interpreted as follows:
            If a kwarg is iterable then it is stored in self.variables as a sweeped variable,
                unless it has length 1, in which case its only element is stored in self.constants.
                NOTE: watch out with multi-dimensional arrays, as their first axis can be interpreted as the 'sweep'!
            If an argument is not iterable, its value is stored in self.constants without converting to a length-1 tuple.
            Subclasses of <Sweep> determine which arguments are used.
            @attr parameters [dict[str, tuple]]: stores all variables and constants with their values in a tuple,
                even if this tuple only has length 1.
            @attr variables [dict[str, tuple]]: stores the sweeped variables and their values in a tuple.
            @attr constants [dict[str, Any]]: stores the non-sweeped variables and their only value.
        """
        self.parameters: dict[str, tuple] = {k: Sweep.variable(v) for k, v in kwargs.items()}
        self.variables = {k: v for k, v in self.parameters.items() if len(v) > 1}
        self.constants = {k: v[0] for k, v in self.parameters.items() if len(v) == 1}

        if groups is None: groups = []
        groups = [tuple([key for key in group if key in self.variables.keys()]) for group in groups] # Remove non-variable groups
        groups = [group for group in groups if len(group)] # Remove now-empty groups
        for k in self.variables.keys(): # Add ungrouped variables to their own lonely group
            if not any(k in group for group in groups): # Then we have a variable that is not in a group
                groups.append((k,))
        self.groups = tuple(groups) # Make immutable
        self.n_per_group = tuple([len(self.variables[group[0]]) for group in self.groups])

        # Check that vars in a single group have the same sweeping length
        for group in self.groups:
            l = [len(self.variables[key]) for key in group]
            if not l.count(l[0]) == len(l): raise ValueError(f"Not all elements of {group} have same sweeping length: {l} respecitvely.")

    @staticmethod
    def variable(iterable):
        """ Converts <iterable> into a tuple, or a length-1 tuple if <iterable> is not iterable."""
        if isinstance(iterable, str): return (iterable,) # We don't want to parse strings character by character
        try: return tuple(iterable)
        except TypeError: return (iterable,)

    def __len__(self) -> int:
        return np.prod(self.n_per_group)

    @property
    def info(self):
        """ Some information about a specific sweep, e.g. what the input/output protocol is etc.
            This should simply be set using self.info = ...
        """
        return self._info if hasattr(self, '_info') else "No specific information provided for this sweep."
    
    @info.setter
    def info(self, value):
        self._info = dedent(value).strip()

    @property
    def sweeper(self):
        for index, _ in np.ndenumerate(np.zeros(self.n_per_group)): # Return an iterable of sorts for sweeping the hypercube of variables
            yield {key: self.variables[key][index[i]] for i, group in enumerate(self.groups) for key in group}

    def __iter__(self):
        """ A generator to conveniently iterate over self.sweeper. Yields a tuple containing
            the variables with their values in this iteration, as well as the experiment object.
        """
        for vars in self.sweeper:
            params = vars | self.constants
            experiment = self.create_experiment(params)
            yield (vars, experiment)

    def get_iteration(self, i: int):
        """ Basically returns one iteration of the self.__iter__() generator, the <i>th that would normally be generated. """
        index = np.unravel_index(i, self.n_per_group)
        vars = {key: self.variables[key][index[i]] for i, group in enumerate(self.groups) for key in group} # TODO: double code, how to unify?
        params = vars | self.constants
        experiment = self.create_experiment(params)
        return (vars, experiment)

    @abstractmethod
    def create_experiment(self, params: dict) -> Experiment:
        """ Subclasses should create an Experiment here according to <params> and return it. """
        pass

    def as_metadata_dict(self):
        return {
            'type': full_obj_name(self),
            'info': self.info,
            'variables': self.variables,
            'parameters': self.parameters,
            'groups': self.groups
        }

    def load_results(self, dir: str, save=True, verbose=True, return_savepath=False):
        """ Loads the collection of JSON files corresponding to a parameter sweep in directory <dir>,
            calculates the relevant results with Experiment().calculate_all() and saves these all to a single file.
            @param dir [str]: the path to the directory where all the sweep data was stored.
            @param sweep [Sweep]: the sweep that generated all the data in <dir>.
            @param save [bool|str] (True): if truthy, the results are saved to a file.
                If specified as a string, the base name of this saved file is this string.
        """
        dir = os.path.normpath(dir)
        savedir = os.path.dirname(dir)
        savename = os.path.basename(dir)

        data = Data.load_collection(dir) # Load all the iterations' data into one large object

        df = pd.DataFrame()
        vars: dict
        experiment: Experiment
        for i, (vars, experiment) in enumerate(self):
            if verbose and is_significant(i, len(self)): print(f"Calculating results of experiment {i+1} of {len(self)}...")
            # 1) Select the part with <vars> in <data>
            df_i = data.df.copy()
            for varname, value in vars.items():
                df_i = df_i.loc[np.isclose(df_i[varname], value)]
            # 2) Re-initialize <experiment> with this data
            experiment.load_dataframe(df_i)
            # 3) Calculate relevant metrics
            experiment.calculate_all(ignore_errors=False)
            # 4) Save these <experiment.results> and <vars> to a dataframe that we are calculating on the fly
            all_info = vars | experiment.results
            df = pd.concat([df, pd.DataFrame(data=all_info, index=[0])], ignore_index=True)

        data = Data(df, constants=data.constants, metadata=data.metadata)
        if save: save = data.save(dir=savedir, name=savename, timestamp=False)
        return save if (return_savepath and save) else data
    
    def plot(self, summary_files, param_x=None, param_y=None, unit_x=None, unit_y=None, transform_x=None, transform_y=None, name_x=None, name_y=None, title=None, save=True, plot=True, metrics_dict=None):
        """ Generic function to plot the results of a sweep in a general way.
            @param summary_file [list(str)]: list of path()s to file(s) as generated by Sweep.load_results().
                If more than one such file is passed on, multiple sweeps' metrics are averaged. This is for example
                useful when there is a random energy barrier that needs to be sampled many times for a correct distribution.
            @param param_x, param_y [str] (None): the column names of the sweeped parameters
                to be put on the X- and Y-axis, respectively.
            # TODO: for ND (N>2) sweeps, we need a way of specifying the values of the N-2 or N-1 unplottable parameters (e.g. by adding <other_params> dict)
            # TODO: Check if this function works for 1D sweeps, or higher than 3D
            # TODO: do something about transform_x and transform_y, they feel out-of-place right now
            # TODO: revisit save=True, plot=True, title=None...
            # TODO: is there a way to somehow detect the units of the axes without explicitly passing them to name_x and name_y?
        """
        if isinstance(summary_files, str):
            if os.path.isfile(summary_files):
                summary_files = [summary_files] # We use an iterable of strings, just in case they need to be averaged.
            else: # is dir
                summary_files = [file for file in os.listdir(summary_files) if (os.path.isfile(file) and file.endswith('.json'))]

        ## Default arguments
        colnames = [group[0] for group in self.groups] # For each group, just take the first parameter to show values/name
        experiment: type[Experiment] = type(self.get_iteration(0)[1]) # Gets the class type of the generated experiments
        if metrics_dict is None: # metrics_dict is a dict of name:function(df) pairs that return (combinations of) df columns
            metrics_dict = experiment.get_plot_metrics()
        n = len(metrics_dict) # Number of metrics

        if param_x is None:
            param_x = colnames[0] # There will be at least one colname, otherwise this is not a sweep
        if param_y is None:
            for colname in colnames:
                if (colname not in metrics_dict.keys()) and (colname != param_x):
                    param_y = colname
                    break
        is_2D = param_y is not None

        ## Load data and extract metrics
        all_metrics = [0 for _ in range(n)]
        for summary_file in summary_files:
            data = Data.load(summary_file)
            if is_2D:
                df = data.df.sort_values([param_x, param_y], ascending=[True, True])
                y_vals = df[param_y].unique()
            else:
                df = data.df.sort_values([param_x], ascending=[True])
            x_vals = df[param_x].unique()
            metrics = [[] for _ in range(n)]
            for val_x, dfi in data.df.groupby(param_x):
                metrics_i = [[] for _ in range(n)]
                if is_2D:
                    for val_y, dfj in dfi.groupby(param_y):
                        data_j = data.mimic(dfj)
                        for i, metric_func in enumerate(metrics_dict.values()):
                            metrics_i[i].append(metric_func(data_j).iloc[0])
                else:
                    data_i = data.mimic(dfi)
                    for i, metric_func in enumerate(metrics_dict.values()):
                        metrics_i[i] = metric_func(data_i).iloc[0]
                for i, _ in enumerate(metrics):
                    metrics[i].append(metrics_i[i])
            for i, metric in enumerate(metrics):
                all_metrics[i] += np.asarray(metric)
        all_metrics = [metric/len(summary_files) for metric in all_metrics]

        ## PLOTTING
        cmap = cm.get_cmap('viridis').copy()
        # cmap.set_under(color='black')
        init_fonts()
        n_plots = len(metrics)
        fig = plt.figure(figsize=(3.3*n, 3.5))

        if transform_x is not None: x_vals = transform_x(x_vals)
        if name_x is None: name_x = param_x
        x_lims = [(3*x_vals[0] - x_vals[1])/2] + [(x_vals[i+1] + x_vals[i])/2 for i in range(len(x_vals)-1)] + [3*x_vals[-1]/2 - x_vals[-2]/2]

        if is_2D:
            if transform_y is not None: y_vals = transform_y(y_vals)
            if name_y is None: name_y = param_y
            y_lims = [(3*y_vals[0] - y_vals[1])/2] + [(y_vals[i+1] + y_vals[i])/2 for i in range(len(y_vals)-1)] + [3*y_vals[-1]/2 - y_vals[-2]/2]
            X, Y = np.meshgrid(x_vals, y_vals)

        # Plot the metrics
        axes = []
        label_x = name_x if unit_x is None else f"{name_x} [{unit_x}]"
        if is_2D:
            label_y = name_y if unit_y is None else f"{name_y} [{unit_y}]"
            for i, name in enumerate(metrics_dict.keys()):
                ax = fig.add_subplot(1, n, i+1)
                axes.append(ax)
                ax.set_xlabel(label_x)
                ax.set_ylabel(label_y)
                ax.set_title(name)
                im = ax.pcolormesh(X, Y, np.transpose(all_metrics[i]), cmap=cmap) # OPT: can use vmin and vmax, but not without a Metric() class, which I think would lead us a bit too far once again
                c = plt.colorbar(im) # OPT: can use extend='min' for nice triangle at the bottom if range is known
                # c.ax.yaxis.get_major_locator().set_params(integer=True) # only integer colorbar labels
        else:
            ax = fig.add_subplot(1, 1, 1)
            for i, name in enumerate(metrics_dict.keys()):
                ax.set_xlabel(label_x)
                ax.set_ylabel("Reservoir metric")
                ax.plot(x_vals, all_metrics[i], label=name)

        if title is not None: plt.suptitle(title)
        multi = widgets.MultiCursor(fig.canvas, axes, color='black', lw=1, linestyle='dotted', horizOn=True, vertOn=True) # Assign to variable to prevent garbage collection
        plt.gcf().tight_layout()

        if save:
            save_path = os.path.splitext(summary_file)[0]
            save_plot(save_path, ext='.pdf')
        if plot:
            plt.show()


######## Below are subclasses of the superclasses above


class KernelQualityExperiment(Experiment):
    def __init__(self, inputter, outputreader, mm):
        """ Follows the paper
                Johannes H. Jensen and Gunnar Tufte. Reservoir Computing in Artificial Spin Ice. ALIFE 2020.
            to implement a similar simulation to determine the kernel-quality K and generalization-capability G.
        """
        super().__init__(inputter, outputreader, mm)
        if not self.inputter.datastream.is_binary: # This is necessary for the (admittedly very bad) way that the inputs are recorded into a dataframe now
            raise AttributeError("KernelQualityExperiment should be performed with a binary datastream only.")
        self.n_out = self.outputreader.n
        self.results = {'K': None, 'G': None, 'k': None, 'g': None} # Kernel-quality and Generalization-capability, and their normalized counterparts

    def run(self, input_length: int = 100, constant_fraction: float = 0.6, pattern=None, verbose=False):
        """ @param input_length [int]: the number of times inputter.input_single() is called,
                before every recording of the output state.
        """
        if verbose: log("Calculating kernel-quality K.")
        self.run_K(input_length=input_length, verbose=verbose, pattern=pattern)
        if verbose: log("Calculating generalization-capability G.")
        self.run_G(input_length=input_length, constant_fraction=constant_fraction, verbose=verbose, pattern=pattern)
        # Still need to call self.calculate_all() manually after this method.

    def run_K(self, input_length: int = 100, pattern=None, verbose=False):
        self.all_states_K = xp.zeros((self.n_out,)*2)
        self.all_inputs_K = ["" for _ in range(self.n_out)]

        for i in range(self.n_out): # To get a square matrix, record the state as many times as there are output values
            self.mm.initialize_m(self.mm._get_groundstate() if pattern is None else pattern, angle=0)
            for j in range(input_length):
                if verbose and is_significant(i*input_length + j, input_length*self.n_out, order=1):
                    log(f"Row {i+1}/{self.n_out}, value {j+1}/{input_length}...")
                val = self.inputter.input_single(self.mm)
                self.all_inputs_K[i] += str(int(val))

            state = self.outputreader.read_state()
            self.all_states_K[i,:] = state.reshape(-1)

    def run_G(self, input_length: int = 100, constant_fraction=0.6, pattern=None, verbose=False):
        """ @param constant_fraction [float] (0.6): the last <constant_fraction>*<input_length> bits will
                be equal for all <self.n_out> bit sequences.
        """
        self.all_states_G = xp.zeros((self.n_out,)*2)
        self.all_inputs_G = ["" for _ in range(self.n_out)]
        constant_length = int(input_length*constant_fraction)
        constant_sequence = self.inputter.datastream.get_next(n=constant_length)
        random_length = input_length - constant_length

        for i in range(self.n_out): # To get a square matrix, record the state as many times as there are output values
            self.mm.initialize_m(self.mm._get_groundstate() if pattern is None else pattern, angle=0)
            for j in range(random_length):
                if verbose and is_significant(i*input_length + j, input_length*self.n_out, order=1):
                    log(f"Row {i+1}/{self.n_out}, random value {j+1}/{random_length}...")
                val = self.inputter.input_single(self.mm)
                self.all_inputs_G[i] += str(int(val))
            for j, value in enumerate(constant_sequence):
                if verbose and is_significant(i*input_length + j + random_length, input_length*self.n_out, order=1):
                    log(f"Row {i+1}/{self.n_out}, constant value {j+1}/{constant_length}...")
                constant = self.inputter.input_single(self.mm, value=float(value))
                self.all_inputs_G[i] += str(int(constant))

            state = self.outputreader.read_state()
            self.all_states_G[i,:] = state.reshape(-1)

    def calculate_all(self, ignore_errors=True, **kwargs):
        """ Recalculates K, G, k and g if possible. """
        try:
            self.results['K'] = int(xp.linalg.matrix_rank(self.all_states_K))
            self.results['k'] = self.results['K']/self.n_out if self.n_out != 0 else 0
        except AttributeError:
            if not ignore_errors: raise

        try:
            self.results['G'] = int(xp.linalg.matrix_rank(self.all_states_G))
            self.results['g'] = self.results['G']/self.n_out if self.n_out != 0 else 0
        except AttributeError:
            if not ignore_errors: raise


    def to_dataframe(self):
        """ DF has columns 'metric' and 'y0', 'y1', ... 'y<self.n_out - 1>'.
            When 'metric' == "K", the row corresponds to a state from self.run_K(), and vice versa for 'G'.
        """
        u = self.all_inputs_K + self.all_inputs_G # Both are lists so we can concatenate them like this
        yK = asnumpy(self.all_states_K) # Need as NumPy array for pd
        yG = asnumpy(self.all_states_G) # Need as NumPy array for pd
        metric = np.array(["K"]*yK.shape[0] + ["G"]*yG.shape[0])
        if yK.shape[1] != yG.shape[1]: raise ValueError(f'K and G were not simulated with an equal amount of readout nodes.')

        df_front = pd.DataFrame({'metric': metric, 'inputs': u})
        df_yK = pd.DataFrame({f'y{i}': yK[:,i] for i in range(yK.shape[1])})
        df_yG = pd.DataFrame({f'y{i}': yG[:,i] for i in range(yG.shape[1])})
        df = pd.concat([df_yK, df_yG], axis=0, ignore_index=True) # Put K and G readouts below each other
        df = pd.concat([df_front, df], axis=1, ignore_index=False) # Add the 'metric' column in front
        return df

    def load_dataframe(self, df: pd.DataFrame):
        """ Loads the arrays <all_states_K> and <all_states_G> stored in the dataframe <df>
            into the <self.all_states_K> and <self.all_states_G> attributes, and returns both.
        """
        df_K = df[df['metric'] == "K"]
        df_G = df[df['metric'] == "G"]

        if df_K.empty or df_G.empty:
            raise ValueError("Dataframe is empty, so could not be loaded to KernelQualityExperiment.")

        # Select the y{i} columns
        pattern = re.compile(r'\Ay[0-9]+\Z') # Match 'y0', 'y1', ... (\A and \Z represent end and start of string, respectively)
        y_cols = [colname for colname in df if pattern.match(colname)]
        y_cols.sort(key=human_sort) # Usually of the format 'y0', 'y1', 'y2', ..., 'y10', where human_sort makes sure e.g. 10 comes after 2
        n_cols = len(y_cols)

        self.all_states_K = xp.asarray(df_K[y_cols]) if n_cols != 0 else xp.array([[1]]*len(df_K)) # n_cols can be 0 if all constant
        self.all_states_G = xp.asarray(df_G[y_cols]) if n_cols != 0 else xp.array([[1]]*len(df_G))
        self.all_inputs_K = list(df_K['inputs'])
        self.all_inputs_G = list(df_G['inputs'])

        return self.all_states_K, self.all_states_G
    
    @staticmethod
    def get_plot_metrics():
        return {
            'K': lambda data: data['K'],
            'G': lambda data: data['G'],
            'Q': lambda data: np.maximum(0, data['K'] - data['G'])
        }


class TaskAgnosticExperiment(Experiment):
    def __init__(self, inputter, outputreader, mm):
        """ Follows the paper
                J. Love, J. Mulkers, R. Msiska, G. Bourianoff, J. Leliaert, and K. Everschor-Sitte. 
                Spatial Analysis of Physical Reservoir Computers. arXiv preprint arXiv:2108.01512, 2022.
            to implement task-agnostic metrics for reservoir computing using a single random input signal.
        """
        super().__init__(inputter, outputreader, mm)
        if self.inputter.datastream.is_binary:
            warnings.warn("A TaskAgnosticExperiment might not work properly for a binary datastream.", stacklevel=2)
        self.results = {'NL': None, 'MC': None, 'S': None}
        self.initial_state = self.outputreader.read_state().reshape(-1)
        self.n_out = self.outputreader.n

    @classmethod
    def dummy(cls, mm: Magnets = None):
        """ Creates a minimalistic working TaskAgnosticExperiment instance.
            @param mm [hotspice.Magnets] (None): if specified, this is used as Magnets()
                object. Otherwise, a minimalistic hotspice.ASI.OOP_Square() instance is used.
        """
        if mm is None: mm = OOP_Square(1, 10, energies=(DipolarEnergy(), ZeemanEnergy()))
        datastream = RandomUniformDatastream(low=-1, high=1)
        inputter = FieldInputter(datastream)
        outputreader = RegionalOutputReader(2, 2, mm)
        return cls(inputter, outputreader, mm)

    def run(self, N=1000, add=False, pattern=None, verbose=False):
        """ @param N [int]: The total number of <self.inputter.input_single()> iterations performed.
            @param add [bool]: If True, the newly calculated iterations are appended to the
                current <self.u> and <self.y> arrays, and <self.final_state> is updated.
            @param verbose [int]: If 0, nothing is printed or plotted.
                If 1, significant iterations are printed to console.
                If >1, the magnetization profile is plotted after every input bit in addition to printing.
        """
        self.mm.initialize_m(self.mm._get_groundstate() if pattern is None else pattern, angle=0)
        if verbose:
            if verbose > 1:
                init_fonts()
                init_interactive()
                fig = None
            log(f"[0/{N}] Running TaskAgnosticExperiment: relaxing initial state...")

        if not add: 
            self.mm.relax()
            self.initial_state = self.outputreader.read_state().reshape(-1)
        # Run the simulation for <N> steps where each step consists of <inputter.n> full Monte Carlo steps.
        u = xp.zeros(N) # Inputs
        y = xp.zeros((N, self.n_out)) # Outputs
        for i in range(N):
            u[i] = self.inputter.input_single(self.mm)
            y[i,:] = self.outputreader.read_state().reshape(-1)
            if verbose:
                if verbose > 1: fig = show_m(self.mm, figure=fig)
                if is_significant(i, N):
                    log(f"[{i+1}/{N}] {self.mm.switches}/{self.mm.attempted_switches} switching attempts successful ({self.mm.MCsteps:.2f} MC steps).")
        self.mm.relax()
        self.final_state = self.outputreader.read_state().reshape(-1)

        self.u = xp.concatenate([self.u, u], axis=0) if add else u
        self.y = xp.concatenate([self.y, y], axis=0) if add else y

        if verbose > 1: close_interactive(fig) # Close the realtime figure as it is no longer needed
        # Still need to call self.calculate_all() manually after this method.

    def calculate_all(self, ignore_errors=False, **kwargs):
        """ Recalculates NL, MC, CP and S. Arguments passed to calculate_all()
            are directly passed through to the appropriate self.<metric> functions.
            @param ignore_errors [bool] (False): if True, exceptions raised by the
                self.<metric> functions are ignored (use with caution).
        """
        for metric, method in {'NL': self.NL, 'MC': self.MC, 'CP': self.CP, 'S': self.S}.items():
            try:
                self.results[metric] = method(**filter_kwargs(kwargs, method))
            except Exception:
                if not ignore_errors: raise

    def NL(self, k: int = 10, local: bool = False, test_fraction: float = 1/4, verbose: bool = False): # Non-linearity
        """ Returns a float (local=False) or an array (local=True) representing the nonlinearity,
            either globally or locally depending on <local>. For this, an estimator for the current readout state is
            trained which for any time instant has access to the <k> most recent inputs (including the present one).
        """
        if verbose: log(f"Calculating NL (local={local})...")
        
        train_test_cutoff = math.ceil(self.u.size*(1-test_fraction))
        if train_test_cutoff < k: raise ValueError(f"NL: k={k} must be <={train_test_cutoff} for the available data.")
        u_train_strided, u_test_strided = xp.split(strided(self.u, k), [train_test_cutoff])
        y_train, y_test = xp.split(self.y, [train_test_cutoff]) # Need to put train_test_cutoff in iterable for correct behavior
        
        # u_train_strided, y_train = u_train_strided[k-1:], y_train[k-1:] # First <k-1> rows don't have <k> previous entries yet to use as 'samples'
        Rsq = xp.empty(self.n_out) # R² correlation coefficient
        for j in range(self.n_out): # Train an estimator for the j-th output feature
            model = sm.OLS(asnumpy(y_train[:,j]), asnumpy(u_train_strided), missing='drop') # missing='drop' ignores samples with NaNs (i.e. the first k-1 samples)
            results = model.fit()
            y_hat_test = results.predict(asnumpy(u_test_strided))
            sigma_y_hat, sigma_y = np.var(y_hat_test), np.var(y_test[:,j])
            if sigma_y_hat == 0: # (highly unlikely) predicting constant value, so R² depends on whether this constant is the average or not
                Rsq[j] = float(xp.mean(y_hat_test) == xp.mean(y_test[:,j]))
            elif sigma_y == 0: # Constant readout, so NL should logically be 0, so Rsq = 1-NL = 1
                Rsq[j] = 1.
            else: # Rsq is 1 for very predictable 
                Rsq[j] = R_squared(y_hat_test, y_test[:,j]) # Same as np.corrcoef(y_hat_test, y_test[:,j])[0,1]**2

        # Handle <local> True or False and return appropriately
        if local:
            return 1 - Rsq
        else:
            return float(1 - xp.mean(Rsq))
    
    def MC(self, k=10, local=False, test_fraction=1/4, verbose=False): # Memory capacity
        """ Returns a float (local=False) or an array (local=True) representing the memory capacity,
            either globally or locally depending on <local>. For this, an estimator is trained which
            for any time instant attempts to predict the previous <k> inputs based on the current readout state.
        """
        if verbose: log(f"Calculating MC (local={local})...")

        train_test_cutoff = math.ceil(self.u.size*(1-test_fraction))
        if train_test_cutoff < k: raise ValueError(f"MC: k={k} must be <={train_test_cutoff} for the available data.")
        u_train, u_test = xp.split(self.u, [train_test_cutoff])
        y_train, y_test = xp.split(self.y, [train_test_cutoff]) # Need to put train_test_cutoff in iterable for correct behavior

        Rsq = xp.empty(k) # R² correlation coefficient
        if local: weights = xp.zeros((k, self.n_out))
        for j in range(1, k+1): # Train an estimator for the input j iterations ago
            # This one is a mess of indices, but at least we don't need striding like for non-linearity
            results = sm.OLS(asnumpy(u_train[k-j:-j]), asnumpy(y_train[k:,:]), missing='drop').fit() # missing='drop' ignores samples with NaNs (i.e. the first k-1 samples)
            if local: weights[j-1,:] = xp.asarray(results.params) # TODO: this gives an error I think
            u_hat_test = results.predict(asnumpy(y_test[j:,:])) # Start at y_test[j] to prevent leakage between train/test
            Rsq[j-1] = R_squared(u_hat_test, u_test[:-j])
        Rsq[xp.isnan(Rsq)] = 0 # If some std=0, then it is constant so R² actually gives 0

        # Handle <local> True or False and return appropriately
        if local:
            # TODO: I think some normalization is wrong here (result is all near 0.5, not full range between 0 and 1 as in paper)
            weights = weights - xp.min(weights, axis=1).reshape(-1, 1) # Minimum value becomes 0
            weights = weights / xp.max(weights, axis=1).reshape(-1, 1) # Maximum value becomes 1
            weights_avg = xp.mean(weights, axis=0) # Average across 'time'
            return weights_avg
        else:
            return float(xp.sum(Rsq))

    def S(self, relax=False): # Stability
        """ Calculates the stability: i.e. how similar the initial and final states are, after relaxation.
            NOTE: It can be advantageous to define a different metric for stability if the entire magnetization profile
                  is known, since this function only has access to the readout nodes, not the whole self.m array.
            @param relax [bool] (False): if True, the final state is determined by relaxing the current state of <self.mm>.
        """
        if relax:
            self.mm.relax()
            final_readout = self.outputreader.read_state().reshape(-1)
        elif not hasattr(self, 'final_state'):
            if self.y.size > 0:
                final_readout = self.y[-1,:]
            else:
                final_readout = self.outputreader.read_state().reshape(-1)

        initial_readout = self.initial_state # Assuming this was relaxed first
        final_readout = self.final_state # Assuming this was also relaxed first
        return 1 - distance.cosine(asnumpy(initial_readout), asnumpy(final_readout))/2 # distance.cosine is in range [0., 2.]
        # Another possibly interesting metric: Hamming distance 
        # (is proportion of disagreeing elements, but would require full access to state mm.m to work properly (-1 or 1 binary state))


    def to_dataframe(self, u: xp.ndarray = None, y: xp.ndarray = None):
        """ If <u> and <y> are not explicitly provided, the saved <self.u> and <self.y>
            arrays are used. When providing <u> and <y> directly,
                <u> should be a 1D array of length N, and
                <y> a <NxL> array, where L is the number of output nodes.
            The resulting dataframe has columns 'u', 'y0', 'y1', ... 'y<L-1>'.
        """
        if (u is None) != (y is None): raise ValueError("Either none or both of <u> and <y> arguments must be provided.")
        if u is None:
            u = xp.concatenate((xp.asarray([xp.nan]), self.u, xp.asarray([xp.nan]))) # self.u with NaN input as 0th index
        if y is None:
            y = xp.concatenate((self.initial_state.reshape(1, -1), self.y, self.final_state.reshape(1, -1)), axis=0) # self.y with pristine state as 0th index
        
        u = asnumpy(u) # Need as NumPy array for pd
        y = asnumpy(y) # Need as NumPy array for pd
        if u.shape[0] != y.shape[0]: raise ValueError(f"Length of <u> ({u.shape[0]}) and <y> ({y.shape[0]}) is incompatible.")

        df_u = pd.DataFrame({'u': u})
        df_y = pd.DataFrame({f'y{i}': y[:,i] for i in range(y.shape[1])})
        df = pd.concat([df_u, df_y], axis=1)
        return df

    def load_dataframe(self, df: pd.DataFrame, u: xp.ndarray = None, y: xp.ndarray = None):
        """ Loads the arrays <u> and <y> stored in the dataframe <df>
            into the <self.u> and <self.y> attributes, and returns both.
        """
        if u is None: u = xp.asarray(df['u'])
        if y is None:
            pattern = re.compile(r'\Ay[0-9]+\Z') # Match 'y0', 'y1', ... (\A and \Z represent end and start of string, respectively)
            y_cols = [colname for colname in df if pattern.match(colname)]
            y_cols.sort(key=human_sort)
            y = xp.asarray(df[y_cols])

        self.u = xp.asarray(u).reshape(-1)
        self.y = xp.asarray(y, dtype=float).reshape(self.u.shape[0], -1)
        self.n_out = y.shape[1]
        if math.isnan(self.u[0]):
            self.initial_state = self.y[0,:]
            self.u = self.u[1:]
            self.y = self.y[1:]
        if math.isnan(self.u[-1]):
            self.final_state = self.y[-1,:]
            self.u = self.u[:-1]
            self.y = self.y[:-1]
        return self.u, self.y

    @staticmethod
    def get_plot_metrics():
        return {
            'NL': lambda data: data['NL'],
            'MC': lambda data: data['MC'],
            'S':  lambda data: data['S']
        }