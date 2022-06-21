import math
import re

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from context import hotspin
from hotspin.experiments import TaskAgnosticExperiment
from hotspin.utils import Data, filter_kwargs


def create_TaskAgnosticExperiment(mm: hotspin.Magnets, ext_magnitude=0.04, ext_angle=0, MCsteps=2, sine=False, resolution=None, res_x=5, res_y=5, **kwargs):
    if resolution is not None: res_x = res_y = resolution

    datastream = hotspin.io.RandomUniformDatastream(low=-1, high=1)
    inputter = hotspin.io.FieldInputter(datastream, magnitude=ext_magnitude, angle=ext_angle, n=MCsteps, sine=sine)
    outputreader = hotspin.io.RegionalOutputReader(res_x, res_y, mm)
    return TaskAgnosticExperiment(inputter, outputreader, mm)

def single_taskagnostic(experiment: TaskAgnosticExperiment, iterations=1000, dirname=None, verbose=False, save=True, plot=False, **kwargs):
    experiment.run(N=iterations, verbose=verbose)

    df = experiment.to_dataframe()
    metadata = {"description": r"Contains the input values <u> and state vectors <y> used for calculating task agnostic metrics of the system as proposed in `Task Agnostic Metrics for Reservoir Computing` by Love et al."}
    constants = {"ext_magnitude": experiment.inputter.magnitude, "ext_angle": experiment.inputter.angle, "out_nx": experiment.outputreader.nx, "out_ny": experiment.outputreader.ny,
                 "inputter": hotspin.utils.full_obj_name(experiment.inputter), "outputreader": hotspin.utils.full_obj_name(experiment.outputreader), "datastream": hotspin.utils.full_obj_name(experiment.inputter.datastream),
                 "moment": experiment.mm.moment_avg, "T": experiment.mm.T_avg, "E_B": experiment.mm.E_B_avg, "nx": experiment.mm.nx, "ny": experiment.mm.ny, "dx": experiment.mm.dx, "dy": experiment.mm.dy, "ASI_type": hotspin.utils.full_obj_name(experiment.mm)}
    data = Data(df, metadata=metadata, constants=constants)
    if save:
        dir = "results/TaskAgnosticExperiment"
        if dirname is not None: dir += f"/{dirname}"
        save = data.save(dir=dir, name=f"TaskAgnostic_{type(experiment.mm).__name__}_{experiment.mm.nx}x{experiment.mm.ny}")
    if plot: single_taskagnostic_plot(experiment, experiment.outputreader, save=save)
    return data

def single_taskagnostic_plot(experiment: TaskAgnosticExperiment, save=False):
    # TODO: make this a plotting function that only uses the Data instance returned by single_taskagnostic()
    print(experiment.results)
    plt.ioff()
    hotspin.utils.shell()
    NL = experiment.NL(local=True)
    MC = experiment.MC(local=True)
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(experiment.outputreader.inflate_flat_array(NL)[:,:,0].get())
    ax1.set_title("NL x")
    plt.colorbar(im1)
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(experiment.outputreader.inflate_flat_array(NL)[:,:,1].get())
    ax2.set_title("NL y")
    plt.colorbar(im2)
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(experiment.outputreader.inflate_flat_array(MC)[:,:,0].get())
    ax3.set_title("MC x")
    plt.colorbar(im3)
    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(experiment.outputreader.inflate_flat_array(MC)[:,:,1].get())
    ax4.set_title("MC y")
    plt.colorbar(im4)
    # print('Nonlinearity nonlocal:', experiment.NL(local=False))
    # print('Memory capacity nonlocal:', experiment.MC(local=False))
    # print('Complexity averaged:', experiment.CP(transposed=False))
    # print('Complexity transposed:', experiment.CP(transposed=True))
    # print('Stability:', experiment.S())
    print(experiment.results)
    if save: hotspin.plottools.save_plot("results/TaskAgnosticExperiment/NL_and_MC_local/test.pdf")
    plt.show()


def sweep_taskagnostic(ASI_type: type[hotspin.Magnets], variables: dict = None, save=True, **constants):
    ''' The dictionary <variables> contains keys representing the parameters to be
        swept, whose value is an array containing their values. A full sweep through
        the hypercube formed by these arrays is performed. The keys should be valid
        arguments to the <ASI_type> class (nx/ny, dx/dy, T, E_B, moment...) or the
        TaskAgnosticExperiment class (ext_field, ext_angle...),
        and should only be scalar values.
            Example: {"T": [250, 300, 350], "V": [2e-22, 4e-22]}
        All additional kwargs are passed directly to the ASI constructor without sweep.
        NOTE: this function requires ASI_type.__init__ to have no purely positional arguments!
    '''
    # Set some defaults in the dictionaries for the required arguments of ASI and experiment
    if "n" not in variables: constants.setdefault("n", 300)
    if "a" not in variables: constants.setdefault("a", 420e-9*math.sqrt(2))
    # Separate constants for ASI and experiment
    constants_mm = filter_kwargs(constants, hotspin.Magnets.__init__) | filter_kwargs(constants, ASI_type.__init__)
    constants_exp = filter_kwargs(constants, create_TaskAgnosticExperiment)
    kwargs = filter_kwargs(constants, single_taskagnostic)

    # Iterate over all varying variables
    all_data = []
    variables_l = [(key, cp.asarray(value).get().reshape(-1)) for key, value in variables.items()] # Make dict ordering consistent, by turning it into a list
    n_per_var = [value.size for _, value in variables_l]
    for index, _ in np.ndenumerate(np.zeros(n_per_var)): # Sweep across the hypercube of variables
        kwargs_i = {} # Now iteratively construct the appropriate dict for this specific index
        for i, (key, fullrange) in enumerate(variables_l):
            kwargs_i[key] = fullrange[index[i]]
        # Separate variables for ASI and experiment
        variables_mm = filter_kwargs(kwargs_i, hotspin.Magnets.__init__) | filter_kwargs(kwargs_i, ASI_type.__init__)
        variables_exp = filter_kwargs(kwargs_i, create_TaskAgnosticExperiment)

        # Create ASI and perform the task agnostic experiment on it
        mm = ASI_type(**constants_mm, **variables_mm, energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()))
        experiment = create_TaskAgnosticExperiment(mm, **constants_exp, **variables_exp)
        data_i = Data.load(single_taskagnostic(experiment, **kwargs))
        df_i = data_i.df
        for i, (varname, value) in enumerate(kwargs_i.items()): df_i.insert(loc=i, column=varname, value=value) # Set variable columns to the value they had here
        all_data.append(data_i)

    data = Data.load_collection(all_data)
    if save: data.save(dir="results/TaskAgnosticExperiment/Sweep", name=save if isinstance(save, str) else f'{",".join(variables.keys())}')


def load_sweep_to_experiments(data: Data):
    ''' Given a Data instance <data> containing the combined data of a parameter sweep of
        experiments, as e.g. generated by `hotspin.utils.load_collection()`, this function loads
        all separate individual `TaskAgnosticExperiment`s which were used to generate and store
        these data, based on the "constants" and variable columns in the full dataframe.
        Returns two things: a dict containing the variable names and their values, and a
        list of lists of lists of ... (nested as much as there are variables) containing
        `TaskAgnosticExperiment` objects fully initialized to represent the data and other
        parameters (inputter, outputter...) as accurately as can possibly be determined from <data>.
    '''
    # First determine what the variables are
    pattern = re.compile(r"\A(u|y[0-9]+)\Z") # Matches "u", "y0", "y1", ...
    variables = {colname: data.df[colname].unique() for colname in data.df.columns if not pattern.match(colname)}
    # Then create a multidimensional array and fill it with experiments
    all_experiments = np.empty([len(vals) for vals in variables.values()], dtype=object) # axis=i represents variables[i]
    dfs_iterable = [dfi for vars, dfi in data.df.groupby(list(variables.keys()))]
    for index, _ in np.ndenumerate(all_experiments):
        i = np.ravel_multi_index(index, all_experiments.shape)
        variables_i = {key: fullrange[index[i]] for i, (key, fullrange) in enumerate(variables.items())} # Variable values in this iteration
        kwargs_i = data.constants | variables_i # Variables overwrite constants if both present
        dfi = dfs_iterable[i]
        all_experiments[index] = load_single_experiment(Data(dfi, constants=kwargs_i), calculate_metrics=False)

    calc_all_metrics(all_experiments)
    return variables, all_experiments

def load_single_experiment(data: Data, *, calculate_metrics=True):
    try:
        ASI_type = eval(data.constants["ASI_type"])
    except:
        try:
            ASI_type = eval(f'hotspin.ASI.{data.constants["ASI_type"]}')
        except:
            ASI_type = hotspin.ASI.IsingASI

    kwargs_mm = {'a': 1, 'n': 10} | filter_kwargs(data.constants, hotspin.Magnets.__init__) | filter_kwargs(data.constants, ASI_type.__init__)
    kwargs_exp = filter_kwargs(data.constants, create_TaskAgnosticExperiment)
    mm = ASI_type(**kwargs_mm)
    experiment = create_TaskAgnosticExperiment(mm, **kwargs_exp)
    experiment.load_dataframe(data.df)
    if calculate_metrics: experiment.calculate_all()
    return experiment

def calc_all_metrics(all_experiments):
    ''' For calculating all metrics of a collection of TaskAgnosticExperiments all at once. '''
    def NL(x: TaskAgnosticExperiment): return x.NL(local=False, k=10)
    def MC(x: TaskAgnosticExperiment): return x.MC(local=False, k=10)
    def CP(x: TaskAgnosticExperiment): return x.CP(transposed=False)
    def S(x: TaskAgnosticExperiment): return x.S(relax=False)
    all_NL = np.vectorize(NL)(all_experiments)
    all_MC = np.vectorize(MC)(all_experiments)
    all_CP = np.vectorize(CP)(all_experiments)
    all_S = np.vectorize(S)(all_experiments)
    return all_NL, all_MC, all_CP, all_S


if __name__ == "__main__":
    # sweep_taskagnostic(hotspin.ASI.PinwheelASI, variables={"n": [25, 30], "T": [400]}, 
    #                    E_B=hotspin.utils.eV_to_J(71), V=470e-9*170e-9*10e-9, PBC=False, pattern='vortex',
    #                    iterations=20, ext_angle=math.pi/180*7, verbose=True)

    # load_sweep_to_experiments(Data.load("results/TaskAgnosticExperiment/Sweep/n,T_20220517162856.json"))
    # print(load_single_experiment(Data.load("results/TaskAgnosticExperiment/Sweep/_20220617153123.json")).results)


    # A RANDOM TEST I TRIED WHICH I THOUGHT HAD REASONABLE VALUES:
    # sweep_taskagnostic(hotspin.ASI.PinwheelASI, variables={}, 
    #                    E_B=hotspin.utils.eV_to_J(0.250), T=300, V=100e-9*100e-9*10e-9, n=100, PBC=True, pattern='vortex',
    #                    iterations=1000, ext_angle=math.pi/180*7, ext_magnitude=0.001, resolution=10, verbose=True
    # )

    # THE SITUATION THEY HAVE IN 'Computation in artificial spin ice' BY TUFTE ET AL.:
    simparams = hotspin.SimParams(UPDATE_SCHEME='Néel')
    sweep_taskagnostic(hotspin.ASI.SquareASI, variables={}, params=simparams,
                       E_B=hotspin.utils.eV_to_J(5), T=300, V=220e-9*80e-9*25e-9, Msat=860e3, a=320e-9, n=9, PBC=False, pattern='random',
                       iterations=1000, ext_angle=math.pi/4, ext_magnitude=0.017, sine=200e6, verbose=True
    ) # TODO: continue developing this Tufte et al. situation, with the frequency correctly being taken into account in the Néel scheme.
