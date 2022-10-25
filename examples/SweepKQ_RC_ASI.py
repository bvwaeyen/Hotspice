import argparse
import math

import numpy as np

try: from context import hotspin
except ModuleNotFoundError: import hotspin


## Define, parse and clean command-line arguments
# Usage: python <this_file.py> [-h] [-o [OUTDIR]] [N]
if __name__ == "__main__": # Need this because, when imported as a module, the args are not meant for this script.
    parser = argparse.ArgumentParser(description='Process an iteration of the sweep defined in this file. If not specified, the data in --outdir are summarized into a single data file using calculate_all().')
    parser.add_argument('iteration', metavar='N', type=int, nargs='?', default=None,
                        help='the index of the sweep-iteration to be performed.')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, nargs='?', default=None)
    args = parser.parse_args()
    outdir = args.outdir if args.outdir is not None else 'results/Sweep'


## Create custom Sweep class to generate the relevant type of experiments
class SweepKQ_RC_ASI(hotspin.experiments.Sweep):
    def __init__(self, groups=None, **kwargs): # If unsure, provide a kwarg as a tuple directly
        ''' Performs a sweep as in `Reservoir Computing in Artificial Spin Ice` by J. H. Jensen and G. Tufte.
            i.e. with binary input and RegionalOutputReader of variable resolution, with PerpFieldInputter.
        '''
        kwargs = { # TODO: issue with ASI_type is that user can't define ASI types themselves for this like this
            "ASI_type": "IP_Pinwheel", "a": 300e-9, "nx": 21, "ny": 21, "T": 300, "E_B": hotspin.utils.eV_to_J(90),
            "PBC": False, "moment": 860e3*220e-9*80e-9*20e-9,
            "ext_field": 0.07, "ext_angle": 7*math.pi/180, "res_x": 5, "res_y": 5
            } | kwargs # Dict with all parameter values as tuples of length 1 or greater
        super().__init__(groups=groups, **kwargs)

    def create_experiment(self, params: dict) -> hotspin.experiments.KernelQualityExperiment:
        mm = getattr(hotspin.ASI, params["ASI_type"])(params["a"], params["nx"], ny=params["ny"], T=params["T"], E_B=params["E_B"], moment=params["moment"], PBC=params["PBC"],
            pattern='uniform', energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()),
            params=hotspin.SimParams(UPDATE_SCHEME='Néel', SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF=0)) # Need Néel for inputter
        datastream = hotspin.io.RandomBinaryDatastream()
        inputter = hotspin.io.PerpFieldInputter(datastream, magnitude=params["ext_field"], angle=params["ext_angle"], n=2, sine=True, frequency=1, half_period=False) # Frequency does not matter as long as it is nonzero and reasonable
        outputreader = hotspin.io.RegionalOutputReader(params["res_x"], params["res_y"], mm) # TODO: update this!!1!1!!I!
        experiment = hotspin.experiments.KernelQualityExperiment(inputter, outputreader, mm)
        return experiment


## From 'Reservoir Computing in Artificial Spin Ice' by J. H. Jensen and G. Tufte:
#! Do not put this in an 'if __name__ == "__main__"' block! <sweep> variable must be importable!
# res_range = tuple([i+1 for i in range(11)])
res_range = 5
dist_range = tuple(np.cbrt(3e-23/(alpha := np.linspace(3e-5, 3e-3, 16))))
# dist_range = tuple(np.cbrt(3e-23/(alpha := np.linspace(1.2e-3, 3e-3, 16))))
field_range = tuple(np.linspace(66e-3, 81e-3, 16))
# field_range = tuple(np.linspace(75e-3, 81e-3, 16))
nx = ny = 21
sweep = SweepKQ_RC_ASI(groups=[("res_x", "res_y")],
    nx=nx, ny=ny,
    res_x=res_range, res_y=res_range,
    ext_field=field_range,
    a=dist_range,
    moment=3e-16, # As derived from flatspin alpha parameter
    E_B=(hotspin.utils.eV_to_J(110)*np.ones((nx, ny))*np.random.normal(1, 0.05, size=(nx, ny)),) # Random 'coercivity'
)


def process_single(vars: dict, experiment: hotspin.experiments.KernelQualityExperiment, save=True, **kwargs):
    #! This is a pretty generic function that won't change much even for different Sweep or Experiment subclasses
    #! except for some arguments to the experiment.run() call and details in the (meta)data.
    experiment_kwargs = {k: kwargs[k] for k in ("input_length", "verbose") if k in kwargs}
    experiment.run(**experiment_kwargs)

    ## Collect and save the (meta)data of this iteration in the output directory
    df_i = experiment.to_dataframe()
    for i, (varname, value) in enumerate(vars.items()): # Set variable columns to the value they had in this iteration
        df_i.insert(loc=i, column=varname, value=(value,)*len(df_i))
    metadata = {
        "description": r"Contains the states (and input bit sequences) used to calculate kernel-quality and generalization-capability as in `Reservoir Computing in Artificial Spin Ice` by J. H. Jensen and G. Tufte.",
        "sweep": sweep.as_metadata_dict()
    }
    constants = {"inputter": hotspin.utils.full_obj_name(experiment.inputter), "outputreader": hotspin.utils.full_obj_name(experiment.outputreader), "datastream": hotspin.utils.full_obj_name(experiment.inputter.datastream),
        "ASI_type": hotspin.utils.full_obj_name(experiment.mm)} | sweep.constants
    data_i = hotspin.utils.Data(df_i, metadata=metadata, constants=constants)
    if save:
        saved_path = data_i.save(dir=outdir, name=savename, timestamp=True)
        hotspin.utils.log(f"Saved iteration #{args.iteration} to {saved_path}", style='success')


if __name__ == "__main__":
    if args.iteration is not None: # Then one specific iteration of sweep() should be calculated on the first available GPU.
        save = True
        options = {
            "input_length": 10,
            "verbose": True
        }

        if save:
            if isinstance(save, str):
                savename = save
            else: # It is important to include iteration number or other unique identifier in savename (timestamps can be same for different finishing GPU processes)
                num_digits = len(str(len(sweep)-1))
                savename = str(sweep.groups).replace('"', '').replace("'", "") + "_" + str(args.iteration).zfill(num_digits) # zfill pads with zeros to the left
        else: savename = ''
        
        process_single(*sweep.get_iteration(args.iteration), save=save, **options)


    if args.iteration is None: # This gets run if the file is called without command line arguments
        sumdir = "results/Sweeps/Sweep_RC_ASI_sweep256_out50_in10_minimize" if args.outdir is None else args.outdir
        sweep.load_results(sumdir, save=True, verbose=True)
