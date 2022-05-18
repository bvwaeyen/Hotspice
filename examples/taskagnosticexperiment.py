import math
import re

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import examplefunctions as ef
from context import hotspin
from hotspin.experiments import TaskAgnosticExperiment


def create_TaskAgnosticExperiment(mm: hotspin.Magnets, ext_magnitude=0.04, ext_angle=0, MCsteps=2, resolution=None, res_x=5, res_y=5, **kwargs):
    if resolution is not None:
        res_x = resolution
        res_y = resolution

    datastream = hotspin.io.RandomUniformDatastream(low=-1, high=1)
    inputter = hotspin.io.FieldInputter(datastream, magnitude=ext_magnitude, angle=ext_angle, n=MCsteps)
    outputreader = hotspin.io.RegionalOutputReader(res_x, res_y, mm)
    return TaskAgnosticExperiment(inputter, outputreader, mm)

def single_taskagnostic(experiment: TaskAgnosticExperiment, iterations=1000, dirname=None, verbose=False, save=True, plot=False, **kwargs):
    experiment.run(N=iterations, verbose=verbose, calc=False)

    df = experiment.to_dataframe()
    metadata = {"description": r"Contains the input values <u> and state vectors <y> used for calculating task agnostic metrics of the system as proposed in `Task Agnostic Metrics for Reservoir Computing` by Love et al."}
    constants = {"ext_magnitude": experiment.inputter.magnitude, "ext_angle": experiment.inputter.angle, "out_nx": experiment.outputreader.nx, "out_ny": experiment.outputreader.ny,
                 "V": experiment.mm.V, "T": experiment.mm.T_avg, "E_b": experiment.mm.E_b_avg, "nx": experiment.mm.nx, "ny": experiment.mm.ny, "dx": experiment.mm.dx, "dy": experiment.mm.dy, "ASI_type": hotspin.utils.full_obj_name(experiment.mm)}
    full_json = hotspin.utils.combine_json(df, metadata=metadata, constants=constants)
    if save:
        path = "results/TaskAgnosticExperiment"
        if dirname is not None: path += f"/{dirname}"
        save = hotspin.utils.save_json(full_json, path=path, name=f"TaskAgnostic_{type(experiment.mm).__name__}_{experiment.mm.nx}x{experiment.mm.ny}")
    if plot: single_taskagnostic_plot(experiment, experiment.outputreader, save=save)
    return full_json

def single_taskagnostic_plot(experiment: TaskAgnosticExperiment, save=False):
    # TODO: make this a plotting function that only uses the JSON data generated in single_taskagnostic()
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
        arguments to the <ASI_type> class (nx/ny, dx/dy, T, E_b, Msat...) or the
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
    constants_mm = hotspin.utils.filter_kwargs(constants, hotspin.Magnets.__init__) | hotspin.utils.filter_kwargs(constants, ASI_type.__init__)
    constants_exp = hotspin.utils.filter_kwargs(constants, create_TaskAgnosticExperiment)
    kwargs = hotspin.utils.filter_kwargs(constants, single_taskagnostic)

    # Iterate over all varying variables
    all_data = []
    variables_l = [(key, cp.asarray(value).get().reshape(-1)) for key, value in variables.items()] # Make dict ordering consistent, by turning it into a list
    n_per_var = [value.size for _, value in variables_l]
    for index, _ in np.ndenumerate(np.zeros(n_per_var)): # Sweep across the hypercube of variables
        kwargs_i = {} # Now iteratively construct the appropriate dict for this specific index
        for i, (key, fullrange) in enumerate(variables_l):
            kwargs_i[key] = fullrange[index[i]]
        # Separate variables for ASI and experiment
        variables_mm = hotspin.utils.filter_kwargs(kwargs_i, hotspin.Magnets.__init__) | hotspin.utils.filter_kwargs(kwargs_i, ASI_type.__init__)
        variables_exp = hotspin.utils.filter_kwargs(kwargs_i, create_TaskAgnosticExperiment)

        # Create ASI and perform the task agnostic experiment on it
        mm = ASI_type(**constants_mm, **variables_mm, energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()), params=hotspin.SimParams(UPDATE_SCHEME='Glauber'))
        experiment = create_TaskAgnosticExperiment(mm, **constants_exp, **variables_exp)
        json_i = hotspin.utils.read_json(single_taskagnostic(experiment, **kwargs))
        df_i = json_i["data"]
        for i, (varname, value) in enumerate(kwargs_i.items()): df_i.insert(loc=i, column=varname, value=value) # Set variable columns to the value they had here
        all_data.append(json_i)

    full_json = hotspin.utils.combine_all(all_data)
    if save: hotspin.utils.save_json(full_json, "results/TaskAgnosticExperiment/Sweep", name=save if isinstance(save, str) else f'{",".join(variables.keys())}')


if __name__ == "__main__":
    sweep_taskagnostic(hotspin.ASI.PinwheelASI, variables={"n": [25, 30], "T": [400]}, 
                       E_b=hotspin.Energy.eV_to_J(71), V=470e-9*170e-9*10e-9, PBC=False, pattern='vortex',
                       iterations=20, ext_angle=math.pi/180*7, verbose=True)
