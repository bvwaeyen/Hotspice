import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import statsmodels.api as sm

from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.spatial import distance
from textwrap import dedent
from typing import Callable, Iterable, Literal

from .core import Magnets, DipolarEnergy, ZeemanEnergy
from .ASI import OOP_Square
from .io import Datastream, ScalarDatastream, IntegerDatastream, BinaryDatastream, Inputter, OutputReader, FieldInputter, RandomScalarDatastream, RegionalOutputReader
from .plottools import close_interactive, init_interactive, init_fonts, save_plot, show_m
from .utils import appropriate_SIprefix, asnumpy, Data, filter_kwargs, full_obj_name, human_sort, is_significant, log, R_squared, strided
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
    
    @abstractmethod
    def get_plot_metrics(self) -> dict[str, 'SweepMetricPlotparams']:
        """ Returns a dictionary with as many elements as there should be 2D or 1D plots
            in Sweep().plot(). Keys are names of these metrics, values are functions that
            take one argument (a hotspice.utils.Data object, which for this purpose is
            equivalent to a pandas.DataFrame), and returns a pandas.Series which is either
            a simple column from the Data, or a mathematical combination of several columns.
        """
        return {}


class Sweep(ABC): # TODO: add a method to finish an unfinished sweep, by specifying an output directory to be completed
    def __init__(self, groups: Iterable[tuple[str]] = None, names: dict[str, str] = None, units: dict[str, str] = None, **kwargs): # kwargs can be anything to specify the sweeping variables and their values.
        """ Sweep quacks like a generator yielding <Experiment> instances, or subclasses thereof.
            The purpose of a Sweep is to reliably generate a sequence of <Experiment>s in a consistent order.
            @param groups [iterable[tuple[str]]] (None): a tuple of tuples of strings. Each tuple represents a group:
                groups are sweeped together. The strings inside those tuples represent the names of the
                variables that are being swept together. By sweeping together, it is meant that those
                variables move through their sweeping values simultaneously, instead of forming a
                hypercube through their values in which each pair is visited. We only visit the diagonal.
                Example: groups=(('nx', 'ny'), ('res_x', 'res_y')) will cause the variables <nx> and <ny>
                    to sweep together, e.g. when <nx> is at its fourth value, <ny> will also be at its fourth value.
                    The same will go for <res_x> and <res_y> together, but they are independent of <nx> and <ny>.
            @param names [dict[str, str]] (None): a dictionary whose keys are the parameters (i.e. **kwargs names),
                and whose values are a readable name for said parameter. This is used in self.plot().
                By default, if a parameter is not present in <names>, the its name in **kwargs is used.
            @param units [dict[str, str]] (None): same as <names>, but now the values are the SI units of the parameter.
                By default, if a parameter is not present in <units>, its unit is None.
                Example: names={'a': "Lattice parameter", ...} and units={'a': "m", ...}

            Any additional keyword arguments (**kwargs) are interpreted as follows:
            If a kwarg is iterable then it is stored in self.variables as a sweeped variable,
                unless it has length 1, in which case its only element is stored in self.constants.
                WARN: watch out with multi-dimensional arrays, as their first axis can be interpreted as the 'sweep'!
            If an argument is not iterable, its value is stored in self.constants without converting to a length-1 tuple.
            Subclasses of <Sweep> determine which kwargs are accepted and how they are used to build <Experiment>s.
            @attr parameters [dict[str, tuple]]: stores all variables and constants with their values in a tuple,
                even if this tuple only has length 1.
            @attr variables [dict[str, tuple]]: stores the sweeped variables and their values in a tuple.
            @attr constants [dict[str, Any]]: stores the non-sweeped variables and their only value.
        """
        ## Create the main dictionaries (parameters, divided into variables and constants)
        self.parameters: dict[str, tuple] = {k: Sweep.variable(v) for k, v in kwargs.items()}
        self.variables = {k: v for k, v in self.parameters.items() if len(v) > 1}
        self.constants = {k: v[0] for k, v in self.parameters.items() if len(v) == 1}

        ## Create and verify groups of variables
        if groups is None: groups = []
        groups = [tuple([key for key in group if key in self.variables.keys()]) for group in groups] # Remove non-variable groups
        groups = [group for group in groups if len(group)] # Remove now-empty groups
        for k in self.variables.keys(): # Add ungrouped variables to their own lonely group
            if not any(k in group for group in groups): # Then we have a variable that is not in a group
                groups.append((k,))
        self.groups = tuple(groups) # Make immutable
        self._n_per_group = tuple([len(self.variables[group[0]]) for group in self.groups])
        self._n = np.prod(self._n_per_group)

        for group in self.groups: # Check that vars in a single group have the same sweeping length
            l = [len(self.variables[key]) for key in group]
            if not l.count(l[0]) == len(l):
                raise ValueError(f"Not all elements of {group} have same sweeping length: {l} respecitvely.")

        ## Utility attributes: names and units
        if names is None: names = {}
        if units is None: units = {}
        self.names = {paramname: names.get(paramname, paramname) for paramname in self.parameters.keys()}
        self.units = {paramname: units.get(paramname, None) for paramname in self.parameters.keys()}

        ## Attempt creating an experiment, to see if the sweep works correctly
        try:
            _, self._example_experiment = self.get_iteration(0)
        except Exception:
            raise RuntimeError("The sweep does not correctly generate Experiment instances.")

    @abstractmethod
    def create_experiment(self, params: dict) -> Experiment:
        """ Subclasses should create an Experiment here according to <params> and return it. """
        pass

    @staticmethod
    def variable(iterable):
        """ Converts <iterable> into a tuple, or a length-1 tuple if <iterable> is not iterable."""
        if isinstance(iterable, str): return (iterable,) # We don't want to parse strings character by character
        try: return tuple(iterable)
        except TypeError: return (iterable,)

    def __len__(self) -> int:
        return self._n

    @property
    def info(self):
        """ Some information about a specific sweep, e.g. what the input/output protocol is etc.
            This should simply be set using self.info = ...
        """
        if not hasattr(self, '_info'):
            self._info = self.__doc__
            if self._info is None: self._info = self.__init__.__doc__
            if self._info is None: self._info = "No specific information about this sweep was provided."
            self._info = dedent(self._info).strip()
        return self._info
    
    @info.setter
    def info(self, value: str):
        self._info = dedent(str(value)).strip()

    def __iter__(self):
        """ A generator to conveniently iterate as "for vars, exp in sweep:".
            Yields a tuple for this iteration: (variable:value dict, experiment object).
        """
        for i in range(len(self)):
            yield self.get_iteration(i)

    def get_iteration(self, i: int) -> tuple[dict, Experiment]:
        """ Returns one iteration of the self.__iter__() generator, the <i>th (zero-indexed) that would normally be generated. """
        vars = self.get_iteration_vars(i)
        params = vars | self.constants
        experiment = self.create_experiment(params)
        return (vars, experiment)
    
    def get_iteration_vars(self, i: int) -> dict:
        """ Returns a dictionary with the values of all variables in iteration <i> (zero-indexed). """
        index = np.unravel_index(i, self._n_per_group)
        return {key: self.variables[key][index[i]] for i, group in enumerate(self.groups) for key in group}

    def as_metadata_dict(self):
        return {
            'type': full_obj_name(self),
            'info': self.info,
            'system': {
                'ASI_type': full_obj_name(self._example_experiment.mm),
                'datastream': full_obj_name(self._example_experiment.inputter.datastream),
                'inputter': full_obj_name(self._example_experiment.inputter),
                'outputreader': full_obj_name(self._example_experiment.outputreader)
            },
            'paramnames': {paramname: (self.names[paramname], self.units[paramname]) for paramname in self.parameters.keys()},
            'parameters': self.parameters,
            'groups': self.groups
        }
    
    def process_single(self, iteration: int, run_kwargs=None, save_dir: Literal[False]|str = False):
        ## Run the experiment with the appropriate kwargs
        vars, experiment = self.get_iteration(iteration)
        if run_kwargs is None: run_kwargs = {}
        experiment.run(**run_kwargs)

        ## Collect and save the (meta)data of this iteration in the output directory
        df_i = experiment.to_dataframe()
        metadata = {
            'description': experiment.__doc__ or experiment.__init__.__doc__ or self.info,
            'sweep': self.as_metadata_dict()
        }
        constants = {
            'E_B': experiment.mm.E_B,
            '_experiment_run_kwargs': run_kwargs
            } | self.constants | vars # (<vars> are variable values in this iteration, so it is ok to put in 'constants')
        data_i = Data(df_i, metadata=metadata, constants=constants)
        if save_dir:
            num_digits = len(str(len(self)-1)) # It is important to include iteration number or other unique identifier in savename (timestamps can be same for different finishing GPU processes)
            savename = str(self.groups).replace('"', '').replace("'", "") + "_" + str(iteration).zfill(num_digits) # zfill pads with zeros to the left
            saved_path = data_i.save(dir=save_dir, name=savename, timestamp=True)
            log(f"Saved iteration #{iteration} to {saved_path}", style='success')

    def load_results(self, dir: str, save=True, verbose=True, return_savepath=False):
        """ Loads the collection of JSON files corresponding to a parameter sweep in directory <dir>,
            calculates the relevant results with Experiment().calculate_all() and saves these all to a single file.
            @param dir [str]: the path to the directory where all the sweep data was stored.
            @param sweep [Sweep]: the sweep that generated all the data in <dir>.
            @param save [bool|str] (True): if truthy, the results are saved to a file.
                If specified as a string, the base name of this saved file is this string.
        """
        if return_savepath and not save: raise ValueError("Can not have return_savepath=True if save=False.")
        dir = os.path.normpath(dir)
        savedir = os.path.dirname(dir)
        savename = os.path.basename(dir)

        df = pd.DataFrame() # Here we will put all the final results of all the iterations in
        varnames = [key for key in self.variables.keys()]
        vars_iterations = [self.get_iteration_vars(i) for i in range(len(self))]
        found_iterations = []
        datafiles = [os.path.join(dir, path) for path in os.listdir(dir) if path.endswith('.json')]
        if len(datafiles) == 0: raise FileNotFoundError(f"There are no sweep-JSON data files in the directory {dir}.")

        # Note to self: parallelizing this for loop gives no performance gain
        for filepath in datafiles: # We do not care about the order; self.plot() should take care of that for us.
            try:
                data_here = Data.load(filepath)
                vars_here = {varname: data_here.constants[varname] for varname in varnames}
            except Exception: continue # Then the file could not be loaded or does not contain the right variables
            ## 1) We need to find the index that this file belongs to, so we can generate the appropriate Experiment
            found = False
            for i, vars_compare in enumerate(vars_iterations):
                if all(np.isclose(vars_here[varname], vars_compare[varname], atol=0) for varname in varnames):
                    vars, experiment = self.get_iteration(i)
                    found_iterations.append(i)
                    found = True
                    break
            if not found: continue
            ## 2) Re-initialize <experiment> with this data and calculate metrics
            if verbose: print(f"Calculating results of iteration {i}/{len(datafiles)}... ({filepath})")
            experiment.load_dataframe(data_here.df)
            experiment.calculate_all(ignore_errors=False) # TODO: add a way to pass kwargs to this method without hardcoding ignore_errors
            ## 3) Put these <experiment.results> and <vars> into a nice df representation 
            all_info = vars | experiment.results
            for key, value in all_info.items():
                if isinstance(value, xp.ndarray): all_info[key] = [value] # Otherwise it is treated as an array
            df = pd.concat([df, pd.DataFrame(data=all_info, index=[i])], ignore_index=True)
        data = Data(df, constants=data_here.constants, metadata=data_here.metadata)

        ## Check if all iterations were unique and all iterations were found, otherwise warn a warning
        if not data.df.index.is_unique:#len(found_iterations) > len(set(found_iterations)):
            warnings.warn(f"Some iteration(s) was/were found multiple times in {dir}!", stacklevel=2)
        if len(found_iterations) != len(self):
            warnings.warn(f"Some iteration(s) was/were not found in {dir}!", stacklevel=2)

        ## Save or return the results
        if save: save = data.save(dir=savedir, name=savename, timestamp=False)
        return save if return_savepath else data
    
    # TODO: a video-making function of a single sweep iteration (no saved data required) to show behavior for those parameters. Perhaps make an option to see each switch individually, but that will be hard to extract from here in the code
    # TODO: this function is way too complicated (as usual)
    def plot(self, summary_files, param_x=None, param_y=None, unit_x=None, unit_y=None, transform_x=None, transform_y=None, name_x=None, name_y=None, title=None, save=True, plot=True, metrics: list[str]=None):
        """ Generic function to plot the results of a sweep in a general way.
            @param summary_files [list(str)]: list of path()s to file(s) as generated by Sweep.load_results().
                If more than one such file is passed on, multiple sweeps' metrics are averaged. This is for example
                useful when there is a random energy barrier that needs to be sampled many times for a correct distribution.
            @param param_x, param_y [str] (None): the column names of the sweeped parameters
                to be put on the X- and Y-axis, respectively.
            @param metrics [list(str)] (None): if specified, only the metrics in this list are plotted (i.e., the
                keys in experiment.get_plot_metrics()). If not specified, all possible metrics are plotted.
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
        metrics_dict = experiment.get_plot_metrics()
        if metrics is not None: # If it is None, we just ignore this and plot them all
            metrics_dict = {k: v for k, v in metrics_dict.items() if k in metrics}
        n = len(metrics_dict) # Number of metrics

        if param_x is None:
            param_x = colnames[0] # There will be at least one colname, otherwise this is not a sweep
        if param_y is None:
            for colname in colnames:
                if (colname not in metrics_dict.keys()) and (colname != param_x):
                    param_y = colname
                    break
        is_2D = param_y is not None

        ## Units and names
        if unit_x is None: unit_x = self.units[param_x]
        if unit_y is None: unit_y = self.units[param_y]
        if name_x is None: name_x = self.names[param_x]
        if name_y is None: name_y = self.names[param_y]

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
                        for i, metric_params in enumerate(metrics_dict.values()):
                            metrics_i[i].append(metric_params.data_extractor(data_j).iloc[0])
                else:
                    data_i = data.mimic(dfi)
                    for i, metric_params in enumerate(metrics_dict.values()):
                        metrics_i[i] = metric_params.data_extractor(data_i).iloc[0]
                for i, _ in enumerate(metrics):
                    metrics[i].append(metrics_i[i])
            for i, metric in enumerate(metrics):
                all_metrics[i] += np.asarray(metric)
        all_metrics = [metric/len(summary_files) for metric in all_metrics]

        ## PLOTTING
        # TODO: determine beforehand if the x or y axes would be better suited to put on a logarithmic scale
        cmap = cm.get_cmap('viridis').copy()
        # cmap.set_under(color='black')
        init_fonts()
        n_plots = len(metrics)
        fig = plt.figure(figsize=(3.3*n, 3.5))

        if transform_x is not None: x_vals = transform_x(x_vals)
        x_lims = [(3*x_vals[0] - x_vals[1])/2] + [(x_vals[i+1] + x_vals[i])/2 for i in range(len(x_vals)-1)] + [3*x_vals[-1]/2 - x_vals[-2]/2]

        if is_2D:
            if transform_y is not None: y_vals = transform_y(y_vals)
            y_lims = [(3*y_vals[0] - y_vals[1])/2] + [(y_vals[i+1] + y_vals[i])/2 for i in range(len(y_vals)-1)] + [3*y_vals[-1]/2 - y_vals[-2]/2]
            X, Y = np.meshgrid(x_vals, y_vals)

        # Plot the metrics
        axes = []
        label_x = name_x if unit_x is None else f"{name_x} [{unit_x}]"
        if is_2D:
            label_y = name_y if unit_y is None else f"{name_y} [{unit_y}]"
            for i, params in enumerate(metrics_dict.values()):
                try:
                    ax = fig.add_subplot(1, n, i+1, sharex=ax, sharey=ax)
                except NameError:
                    ax = fig.add_subplot(1, n, i+1)
                axes.append(ax)
                ax.set_xlabel(label_x)
                ax.set_ylabel(label_y)
                ax.set_title(params.full_name)
                im = ax.pcolormesh(X, Y, np.transpose(all_metrics[i]), cmap=cmap, vmin=params.min_value, vmax=params.max_value) # OPT: can use vmin and vmax, but not without a Metric() class, which I think would lead us a bit too far once again
                c = plt.colorbar(im) # OPT: can use extend='min' for nice triangle at the bottom if range is known
                # c.ax.yaxis.get_major_locator().set_params(integer=True) # only integer colorbar labels
        else:
            ax = fig.add_subplot(1, 1, 1)
            for i, params in enumerate(metrics_dict.values()):
                ax.set_xlabel(label_x)
                ax.set_ylabel("Reservoir metric")
                ax.plot(x_vals, all_metrics[i], label=params.full_name)

        if title is not None: plt.suptitle(title)
        multi = widgets.MultiCursor(fig.canvas, axes, color='black', lw=1, linestyle='dotted', horizOn=True, vertOn=True) # Assign to variable to prevent garbage collection
        plt.gcf().tight_layout()

        if save:
            save_path = os.path.splitext(summary_file)[0]
            save_plot(save_path, ext='.pdf')
        if plot:
            plt.show()

@dataclass
class SweepMetricPlotparams:
    ''' Stores some parameters to correctly plot the metrics belonging to a certain experiment.
        @param full_name [str]: A human-readable name for the metric.
        @param data_extractor [function]: A function that takes one argument, namely a Data object.
            It returns a DataFrame column with the metric. Usually, this will just be a column from Data().df.
            The simplest example of this would be something like "lambda data: data[<column_of_metric>]".
            Combinations of columns are also possible, e.g. "lambda data: data[<some_metric>] - data[<other_metric>]".
        @param min_value [float] (None): The minimum value that the metric can be.
        @param max_value [float] (None): The maximum value that the metric can be.
        If <min_value> or <max_value> are None, the plot is simply scaled to the min/max values throughout the sweep.
    '''
    full_name: str # e.g.: 'Nonlinearity'
    data_extractor: Callable # e.g.: lambda data: data['NL']
    min_value: float = None # e.g.: 0
    max_value: float = None # e.g.: 1


######## Below are subclasses of the superclasses above


class KernelQualityExperiment(Experiment):
    def __init__(self, inputter, outputreader, mm):
        """ Follows the paper
                Johannes H. Jensen and Gunnar Tufte. Reservoir Computing in Artificial Spin Ice. ALIFE 2020.
            to implement a similar simulation to determine the kernel-quality K and generalization-capability G.
        """
        super().__init__(inputter, outputreader, mm) # TODO: allow float datastreams as well
        if not isinstance(self.inputter.datastream, BinaryDatastream): # This is necessary for the (admittedly very bad) way that the inputs are recorded into a dataframe now
            raise AttributeError("KernelQualityExperiment should be performed with a binary datastream only.")
        self.n_out = self.outputreader.n
        self.results = {'K': None, 'G': None, 'k': None, 'g': None} # Kernel-quality and Generalization-capability, and their normalized counterparts

    def run(self, input_length: int = 100, constant_fraction: float = 0.6, pattern=None, verbose=False):
        """ @param input_length [int]: the number of inputter.input() calls before each recording of the output state. """
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
                val = self.inputter.input(self.mm)
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
                val = self.inputter.input(self.mm)
                self.all_inputs_G[i] += str(int(val))
            for j, value in enumerate(constant_sequence):
                if verbose and is_significant(i*input_length + j + random_length, input_length*self.n_out, order=1):
                    log(f"Row {i+1}/{self.n_out}, constant value {j+1}/{constant_length}...")
                constant = self.inputter.input(self.mm, float(value))
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
        """ DF has columns 'metric' and 'y'.
            When 'metric' == "K", the row corresponds to a state from self.run_K(), and vice versa for 'G'.
        """
        u = self.all_inputs_K + self.all_inputs_G # Both are lists so we can concatenate them like this
        yK = asnumpy(self.all_states_K) # Need as NumPy array for pd
        yG = asnumpy(self.all_states_G) # Need as NumPy array for pd
        metric = np.array(["K"]*yK.shape[0] + ["G"]*yG.shape[0])
        if yK.shape[1] != yG.shape[1]: raise ValueError(f'K and G were not simulated with an equal amount of readout nodes.')

        return pd.DataFrame({'metric': metric, 'inputs': u, 'y': list(yK) + list(yG)})

    def load_dataframe(self, df: pd.DataFrame):
        """ Loads the arrays <all_states_K> and <all_states_G> stored in the dataframe <df>
            into the <self.all_states_K> and <self.all_states_G> attributes, and returns both.
        """
        df_K = df[df['metric'] == "K"]
        df_G = df[df['metric'] == "G"]

        if df_K.empty or df_G.empty:
            raise ValueError("Dataframe is empty, so could not be loaded to KernelQualityExperiment.")
        if 'y' not in df_K or 'y' not in df_G: # Then 'y' is not in the dataframe because it is constant (this should not happen anymore, but let's keep this to be sure)
            self.all_states_K = xp.array([[1]]*len(df_K))
            self.all_states_G = xp.array([[1]]*len(df_G))
        else:
            self.all_states_K = xp.asarray([xp.asarray(readout) for readout in df_K['y']])
            self.all_states_G = xp.asarray([xp.asarray(readout) for readout in df_G['y']])
        self.all_inputs_K = list(df_K['inputs'])
        self.all_inputs_G = list(df_G['inputs'])

        return self.all_states_K, self.all_states_G
    
    def get_plot_metrics() -> dict[SweepMetricPlotparams]:
        return {
            'K': SweepMetricPlotparams("Kernel-quality", lambda data: data['K'], min_value=0), # TODO: can only add max_value=self.n_out if this is no longer a staticmethod
            'G': SweepMetricPlotparams("Generalization-capability", lambda data: data['G'], min_value=0),
            'Q': SweepMetricPlotparams("Compute quality", lambda data: np.maximum(0, data['K'] - data['G']), min_value=0)
        }


class TaskAgnosticExperiment(Experiment): # TODO: add a plot method to this class that plots the spatial metrics, and leave the total (non-local) metrics exposed for the Sweep.plot().
    def __init__(self, inputter, outputreader, mm):
        """ Follows the paper
                J. Love, J. Mulkers, R. Msiska, G. Bourianoff, J. Leliaert, and K. Everschor-Sitte. 
                Spatial Analysis of Physical Reservoir Computers. arXiv preprint arXiv:2108.01512, 2022.
            to implement task-agnostic metrics for reservoir computing using a single random input signal.
        """
        super().__init__(inputter, outputreader, mm)
        self.initial_state = self.outputreader.read_state().reshape(-1)
        self.n_out = self.outputreader.n
        self.results = {'NL': None, 'NL_local': None, 'MC': None, 'MC_local': None, 'S': None, 'S_local': None}

    @classmethod
    def dummy(cls, mm: Magnets = None):
        """ Creates a minimalistic working TaskAgnosticExperiment instance.
            @param mm [hotspice.Magnets] (None): if specified, this is used as Magnets()
                object. Otherwise, a minimalistic hotspice.ASI.OOP_Square() instance is used.
        """
        if mm is None: mm = OOP_Square(1, 10, energies=(DipolarEnergy(), ZeemanEnergy()))
        datastream = RandomScalarDatastream(low=-1, high=1)
        inputter = FieldInputter(datastream)
        outputreader = RegionalOutputReader(2, 2, mm)
        return cls(inputter, outputreader, mm)

    def run(self, N=1000, pattern=None, verbose=False):
        """ @param N [int]: The total number of <self.inputter.input()> iterations performed.
            @param pattern [str] (None): The state that <self.mm> is initialized in. If not specified,
                the ground state of the spin ice geometry is used.
            @param verbose [int] (False): If 0, nothing is printed or plotted.
                If 1, significant iterations are printed to console.
                If >1, the magnetization profile is plotted after every input bit in addition to printing.
        """
        # Ground state initialization, relax and store the initial state
        self.mm.initialize_m(self.mm._get_groundstate() if pattern is None else pattern, angle=0)
        if verbose:
            if verbose > 1:
                init_fonts()
                init_interactive()
                fig = None
            log(f"[0/{N}] Running TaskAgnosticExperiment: relaxing initial state...")
        self.mm.relax()
        self.initial_state = self.outputreader.read_state().copy() # TODO: is copying the right way to solve our problems?
        # Run the simulation for <N> steps where each step consists of <inputter.n> full Monte Carlo steps.
        self.u = xp.zeros(N) # Inputs
        self.y = xp.zeros((N, self.n_out)) # Outputs
        for i in range(N):
            self.u[i] = self.inputter.input(self.mm)
            self.y[i,:] = self.outputreader.read_state().copy()
            if verbose:
                if verbose > 1: fig = show_m(self.mm, figure=fig)
                if is_significant(i, N):
                    log(f"[{i+1}/{N}] {self.mm.switches}/{self.mm.attempted_switches} switching attempts successful ({self.mm.MCsteps:.2f} MC steps).")
        if verbose > 1: close_interactive(fig) # Close the realtime figure as it is no longer needed

        # Relax and store the final state
        self.inputter.remove_stimulus(self.mm)
        self.mm.relax()
        self.final_state = self.outputreader.read_state().copy()
        # Still need to call self.calculate_all() manually after this method.

    def calculate_all(self, ignore_errors=False, **kwargs):
        """ Recalculates NL, MC, CP and S. Arguments passed to calculate_all()
            are directly passed through to the appropriate self.<metric> functions.
            @param ignore_errors [bool] (False): if True, exceptions raised by the
                self.<metric> functions are ignored (use with caution).
        """
        kwargs.setdefault('use_stored', True)
        for metric, method in {'NL_local': self.NL_local, 'NL': self.NL, 'MC_local': self.MC_local, 'MC': self.MC, 'S_local': self.S_local, 'S': self.S}.items():
            try:
                self.results[metric] = method(**kwargs)
            except Exception:
                if not ignore_errors: raise

    def NL_local(self, k: int = 10, test_fraction: float = 1/4, verbose: bool = False, **kwargs) -> xp.ndarray:
        """ Returns an array representing the nonlinearity of each output node. For this, a linear
            estimator is trained which attempts to predict the current output (self.y) based on the
            <k> most recent inputs (self.u) (including the current one).
            @param k [int] (10): the number of previous inputs visible to the linear estimator to
                estimate the output (this should be longer than the relaxation time of the reservoir).
            @param test_fraction [float] (.25): this is the fraction of data points used in the test set
                to evaluate the metrics. The remaining 1-<test_fraction> are used to train the estimators.
        """
        if verbose: log(f"Calculating NL_local...")
        
        train_test_cutoff = math.ceil(self.u.size*(1 - test_fraction))
        if train_test_cutoff < k: raise ValueError(f"Nonlinearity: size of training set ({train_test_cutoff}) must be >= k ({k}).")
        u_train_strided, u_test_strided = xp.split(strided(self.u, k), [train_test_cutoff]) # train_test_cutoff in iterable for correct behavior
        y_train, y_test = xp.split(self.y, [train_test_cutoff])

        Rsq = xp.empty(self.n_out) # R² correlation coefficient for each output node
        for j in range(self.n_out): # Train an estimator for the j-th output node
            Y = asnumpy(y_train[:,j])
            X = sm.add_constant(asnumpy(u_train_strided), has_constant='add') # Also add constant 'c' from formula 3 of paper
            model = sm.OLS(Y, X, missing='drop') # missing='drop' ignores samples with NaN (here: the first k-1 samples)
            results = model.fit()
            y_hat_test = results.predict(sm.add_constant(asnumpy(u_test_strided), has_constant='add'))
            if np.var(y_hat_test) == 0: # Predicting constant value, so R² depends on whether this constant is the average or not
                Rsq[j] = float(xp.mean(y_hat_test) == xp.mean(y_test[:,j]))
            elif np.var(y_test[:,j]) == 0: # Constant readout, so NL should be 0, so Rsq = 1-NL = 1
                Rsq[j] = 1.
            else:
                Rsq[j] = R_squared(y_hat_test, y_test[:,j])
        return 1 - Rsq

    def NL(self, use_stored=False, **kwargs) -> float:
        """ Returns the average nonlinearity throughout the system. """
        NL = self.results["NL_local"] if use_stored else self.NL(**kwargs)
        return float(xp.mean(NL))

    def MC_local(self, k: int = 10, test_fraction: float = 1/4, threshold_dist: float = None, verbose: bool = False, **kwargs) -> xp.ndarray:
        """ Returns an array representing the memory capacity of each output node. For this, a
            linear estimator is trained to recall the history of inputs (self.u) based on the
            output values (self.y) of the output nodes within a threshold distance of each node.
            @param k [int] (10): the furthest amount of steps back in time that an estimator is
                trained to recall (this should be longer than the relaxation time of the reservoir).
            @param test_fraction [float] (.25): this is the fraction of data points used in the test set
                to evaluate the metrics. The remaining 1-<test_fraction> are used to train the estimators.
            @param threshold_dist [float] (2*NN_dist): to determine local MC, a neighborhood (all output
                nodes at a distance of at most <threshold_dist>) around the output node is used to provide
                data for the estimator.
        """
        if verbose: log(f"Calculating MC_local...")
        train_test_cutoff = math.ceil(self.u.size*(1-test_fraction))
        u_train, u_test = xp.split(self.u, [train_test_cutoff]) # train_test_cutoff in iterable for correct behavior
        if u_train.size <= k: raise ValueError(f"Memory capacity: size of training set ({u_train.size}) must be > k ({k}).")
        if u_test.size <= k: raise ValueError(f"Memory capacity: size of test set ({u_test.size}) must be > k ({k}).")

        dist_matrix = xp.asarray(distance.cdist(asnumpy(self.outputreader.node_coords), asnumpy(self.outputreader.node_coords)))
        if threshold_dist is None: # By default we take twice the minimum NN distance to get some averaging nearby
            mx = ma.masked_array(dist_matrix, mask=(dist_matrix==0))
            min_dist_for_each_cell = mx.min(axis=1).data
            threshold_dist = 2*xp.max(min_dist_for_each_cell)

        N = self.outputreader.n if threshold_dist != xp.inf else 1 # If distance is xp.inf, all nodes will give the same, so set N=1 for efficiency
        MC_of_each_node = xp.zeros(self.outputreader.n)
        for outputnode in range(N):
            neighbors = xp.where(dist_matrix[outputnode,:] < threshold_dist)[0] # \gamma_n in paper
            y_train, y_test = np.split(sm.add_constant(asnumpy(self.y[:,neighbors]), has_constant='add'), [train_test_cutoff])
            Rsq = xp.empty(k) # R² correlation coefficient
            for tau in range(1, k+1): # Train an estimator for the input <tau> iterations ago
                # This one is a mess of indices, but at least we don't need striding like for NL
                results = sm.OLS(asnumpy(u_train[k-tau:-tau]), y_train[k:,:], missing='drop').fit() # missing='drop' ignores samples with NaNs (i.e. the first k-1 samples)
                u_hat_test = results.predict(y_test[tau:,:]) # Start at y_test[j] to prevent leakage between train/test
                Rsq[tau-1] = R_squared(u_hat_test, u_test[:-tau])
            MC_of_each_node[outputnode] = float(xp.sum(Rsq))
        if N == 1: MC_of_each_node[:] = MC_of_each_node[0]
        return MC_of_each_node

    def MC(self, **kwargs) -> float:
        """ Returns the memory capacity based on an estimator with access to all the output nodes. """
        kwargs["threshold_dist"] = xp.inf # To remove locality
        return float(xp.mean(self.MC_local(**kwargs))) # xp.mean to get float instead of oddly-sized array

    def S_local(self, **kwargs):
        """ Returns an array representing the stability of each output node, which is a boolean
            value that is True if the initial and final states are the same, otherwise False.
        """
        return (self.initial_state == self.final_state).astype(float) # (ideally these should be full mm.m states, not the output readout)
        
    def S(self, **kwargs):
        """ Returns the stability of the complete system using a cosine distance metric. """
        return 1 - distance.cosine(asnumpy(self.initial_state), asnumpy(self.final_state))/2 # distance.cosine is in range [0., 2.]
        # Another possibly interesting metric: Hamming distance (would require access to full mm.m state to work properly)

    # TODO: can add parity check metric from "Numerical simulation of artificial spin ice for reservoir computing", which has a very similar calculation procedure

    def to_dataframe(self, u: xp.ndarray = None, y: xp.ndarray = None):
        """ If <u> and <y> are not explicitly provided, the saved <self.u> and <self.y>
            arrays are used. When providing <u> and <y> directly,
                <u> should be a 1D array of length N, and
                <y> a <NxL> array, where L is the number of output nodes.
            The resulting dataframe has columns 'u' and 'y'.
        """
        if (u is None) != (y is None): raise ValueError("Either none or both of <u> and <y> arguments must be provided.")
        if u is None:
            u = xp.concatenate((xp.asarray([xp.nan]), self.u, xp.asarray([xp.nan]))) # self.u with NaN input as 0th index
        if y is None:
            y = xp.concatenate((self.initial_state.reshape(1, -1), self.y, self.final_state.reshape(1, -1)), axis=0) # self.y with pristine state as 0th index
        
        u = asnumpy(u) # Need as NumPy array for pd
        y = asnumpy(y) # Need as NumPy array for pd
        if u.shape[0] != y.shape[0]: raise ValueError(f"Length of <u> ({u.shape[0]}) and <y> ({y.shape[0]}) is incompatible.")

        return pd.DataFrame({'u': u, 'y': list(y)})

    def load_dataframe(self, df: pd.DataFrame, u: xp.ndarray = None, y: xp.ndarray = None):
        """ Loads the arrays <u> and <y> stored in the dataframe <df>
            into the <self.u> and <self.y> attributes, and returns both.
        """
        if u is None: u = xp.asarray(df['u'])
        if y is None: y = xp.asarray(df['y'])

        self.u = xp.asarray(u).reshape(-1)
        self.y = xp.asarray([arr for arr in y], dtype=float).reshape(self.u.shape[0], -1)
        self.n_out = self.y.shape[1]
        if math.isnan(self.u[0]):
            self.initial_state = self.y[0,:]
            self.u = self.u[1:]
            self.y = self.y[1:]
        if math.isnan(self.u[-1]):
            self.final_state = self.y[-1,:]
            self.u = self.u[:-1]
            self.y = self.y[:-1]
        return self.u, self.y

    # TODO: make a self.plot() function for NL_local, MC_local and S_local, but this will need voronoi in the most general case which is not nice
    def get_plot_metrics() -> dict[str:SweepMetricPlotparams]: # TODO: Better name to avoid confusion with self.plot(), add information about min and max values, name...
        return {
            'NL': SweepMetricPlotparams('Nonlinearity', lambda data: data['NL'], min_value=0, max_value=1),
            'MC': SweepMetricPlotparams('Memory Capacity', lambda data: data['MC'], min_value=0),
            'S':  SweepMetricPlotparams('Stability', lambda data: data['S'], min_value=0, max_value=1)
        }


class IODistanceExperiment(Experiment):
    # TODO: try some distance metric that weighs more recent bits more? Or some memory-like distance like that
    def __init__(self, inputter: Inputter, outputreader: OutputReader, mm: Magnets):
        if not isinstance(inputter.datastream, BinaryDatastream): raise ValueError("IODistanceExperiment must use an inputter with a binary datastream.")
        super().__init__(inputter, outputreader, mm)
    
    def run(self, N=10, input_length=100, pattern=None, verbose=False):
        """ Performs some input sequences and records the output after each sequence.
            @param N [int] (10): The number of distinct input sequences whose distances will be compared.
            @param input_length [int] (100): The number of bits in each input sequence.
        """
        self.input_sequences = xp.zeros((N, input_length), dtype=self.inputter.datastream.dtype) # dtype provided for faster pdist if possible
        self.output_sequences = xp.zeros((N, self.outputreader.n))
        for i in range(N):
            self.mm.initialize_m(self.mm._get_groundstate() if pattern is None else pattern, angle=0)
            for j in range(input_length):
                if verbose and is_significant((iter := i*input_length + j), N*input_length): log(f"Inputting bit ({i}, {j}) of ({N}, {input_length})...")
                value = self.inputter.input(self.mm)
                self.input_sequences[i, j] = value # Works if value is scalar or size-1 array, which it should be
            self.output_sequences[i,:] = self.outputreader.read_state().reshape(-1)
        # Still need to call self.calculate_all() manually after this method.
    
    def calculate_all(self, input_metric='hamming', output_metric='euclidean', input_metric_kwargs=None, output_metric_kwargs=None):
        """ (Re)calculates all the metrics in the self.results dict.
            @param input_metric [str] ('hamming'): the distance metric to use between two input sequences.
                The default metric 'hamming' represents the fraction of disagreeing elements between two input sequences.
            @param output_metric [str] ('euclidean'): the distance metric to use between two output sequences.
            @param input_metric_kwargs, output_metric_kwargs [dict] (None): optional kwargs for the SciPy metrics.
        """
        # This calculation should be relatively fast, hence why we save the input_sequences instead of input_distances etc.
        self.input_metric, self.output_metric = input_metric, output_metric
        if input_metric_kwargs is None or not isinstance(input_metric_kwargs, dict): input_metric_kwargs = {}
        if output_metric_kwargs is None or not isinstance(output_metric_kwargs, dict): output_metric_kwargs = {}
        self.input_distances = distance.pdist(asnumpy(self.input_sequences), input_metric).reshape(-1)
        self.output_distances = distance.pdist(asnumpy(self.output_sequences), output_metric).reshape(-1)
    
    def to_dataframe(self):
        df = pd.DataFrame({'input_sequence': self.input_sequences, 'output_sequence': self.output_sequences})
        return df
    
    def load_dataframe(self, df: pd.DataFrame):
        """ Loads data generated by self.to_dataframe() to the current object. """
        self.input_sequences = xp.asarray(df['input_sequence'])
        self.output_sequences = xp.asarray(df['output_sequence'])
        self.calculate_all()
        return self.input_sequences, self.output_sequences
    
    def plot(self, input_metric: str = None, output_metric: str = None):
        plt.scatter(asnumpy(self.input_distances), asnumpy(self.output_distances))
        plt.xlabel(f"Input distance ({self.input_metric if input_metric is None else input_metric})")
        plt.ylabel(f"Output distance ({self.output_metric if output_metric is None else output_metric})")
        plt.show()
    
    @staticmethod
    def get_plot_metrics(self):
        """ Returns a dictionary with as many elements as there should be 2D or 1D plots
            in Sweep().plot(). Keys are names of these metrics, values are functions that
            take one argument which is a pd.DataFrame, and returns either a simple column
            or a mathematical combination of several columns.
        """
        return {}
