import argparse
import math

try: from context import hotspin
except: import hotspin
from hotspin.experiments import TaskAgnosticExperiment
from hotspin.utils import Data


## Define, parse and clean command-line arguments
# Usage: <this_file.py> [-h] [-o [OUTDIR]] [N]
parser = argparse.ArgumentParser(description='Process an iteration of the sweep defined in this file.')
parser.add_argument('iteration', metavar='N', type=int, nargs='?', default=None,
                    help='the index of the sweep-iteration to be performed')
parser.add_argument('-o', '--outdir', dest='outdir', type=str, nargs='?', default='results/Sweep')
args = parser.parse_args()


## Create custom Sweep class to generate the relevant type of experiments
class SweepTA_RC_ASI(hotspin.experiments.Sweep):
    def __init__(self, groups=None, **kwargs): # If unsure, provide a kwarg as a tuple directly
        ''' Performs a sweep as in `Reservoir Computing in Artificial Spin Ice` by J. H. Jensen and G. Tufte.
            i.e. with binary input and RegionalOutputReader of variable resolution, with PerpFieldInputter.
        '''
        kwargs = { # TODO: issue with ASI_type is that user can't define ASI types themselves for this like this
            "ASI_type": "IP_Pinwheel", "a": 600e-9, "nx": 21, "ny": 21, "T": 300, "E_B": hotspin.utils.eV_to_J(71),
            "PBC": False, "moment": 860e3*470e-9*170e-9*10e-9,
            "ext_field": 0.07, "ext_angle": 7*math.pi/180, "res_x": 5, "res_y": 5, "sine": False
            } | kwargs # Dict with all parameter values as tuples of length 1 or greater
        super().__init__(groups=groups, **kwargs)

    def create_experiment(self, params: dict) -> TaskAgnosticExperiment:
        mm = getattr(hotspin.ASI, params["ASI_type"])(params["nx"], params["a"], ny=params["ny"], T=params["T"], E_B=params["E_B"], moment=params["moment"], PBC=params["PBC"],
            pattern='random', energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()), params=hotspin.SimParams(UPDATE_SCHEME='Néel'))
        datastream = hotspin.io.RandomBinaryDatastream()
        inputter = hotspin.io.PerpFieldInputter(datastream, magnitude=params["ext_field"], angle=params["ext_angle"], n=2, sine=params["sine"])
        outputreader = hotspin.io.RegionalOutputReader(params["res_x"], params["res_y"], mm)
        experiment = TaskAgnosticExperiment(inputter, outputreader, mm)
        return experiment


## From 'Reservoir Computing in Artificial Spin Ice' by J. H. Jensen and G. Tufte:
# res_range = tuple([i+1 for i in range(11)])
res_range = 5
dist_range = tuple([500e-9+i*50e-9 for i in range(10)])
# dist_range = 600e-9
field_range = tuple([0.06+i*0.002 for i in range(10)])
# field_range = 0.07
sweep = SweepTA_RC_ASI(groups=[("res_x", "res_y")],
    res_x=res_range, res_y=res_range,
    ext_field=field_range,
    a=dist_range
)


def process_single(vars: dict, experiment: TaskAgnosticExperiment):
    #! This is a pretty generic function that won't change much even for different Sweep or Experiment subclasses
    #! except for some arguments to the experiment.run() call and details in the (meta)data.
    experiment.run(N=iterations, verbose=verbose)

    ## Collect and save the (meta)data of this iteration in the output directory
    df_i = experiment.to_dataframe()
    for i, (varname, value) in enumerate(vars.items()): # Set variable columns to the value they had in this iteration
        df_i.insert(loc=i, column=varname, value=(value,)*len(df_i))
    metadata = {
        "description": r"Contains the input values <u> and state vectors <y> as can be used for calculating task agnostic metrics of the system as proposed in `Task Agnostic Metrics for Reservoir Computing` by Love et al.",
        "sweep": {
            "type": hotspin.utils.full_obj_name(sweep),
            "variables": sweep.variables,
            "parameters": sweep.parameters,
            "groups": sweep.groups
        }
    }
    constants = {"inputter": hotspin.utils.full_obj_name(experiment.inputter), "outputreader": hotspin.utils.full_obj_name(experiment.outputreader), "datastream": hotspin.utils.full_obj_name(experiment.inputter.datastream),
        "ASI_type": hotspin.utils.full_obj_name(experiment.mm)} | sweep.constants
    data_i = Data(df_i, metadata=metadata, constants=constants)
    if save:
        saved_path = data_i.save(dir=args.outdir, name=savename, timestamp=True)
        hotspin.utils.log(f"Saved iteration #{args.iteration} to {saved_path}")


if __name__ == "__main__":
    iterations = 20
    verbose = True
    save = True
    
    if save:
        if isinstance(save, str):
            savename = save
        else: #! important to include iteration number or other unique identifier, since timestamps can be the same for different finishing GPU processes!
            num_digits = len(str(len(sweep)-1))
            savename = str(sweep.groups).replace('"', '').replace("'", "") + "_" + str(args.iteration).zfill(num_digits) # zfill pads with zeros to the left
    else: savename = ''
    if args.iteration is not None:
        process_single(*sweep.get_iteration(args.iteration))
    else:
        hotspin.utils.log("No specific iteration was provided as command-line argument, so the entire sweep will be run.")
        for vars, experiment in sweep:
            process_single(vars, experiment)
