import math
import os

import numpy as np

from typing import Literal

import hotspice


## Create custom Sweep class to generate the relevant type of experiments
class Sweep_OOPSquare_AFM2_KQ_byteInput(hotspice.experiments.Sweep):
    def __init__(self, groups=None, **kwargs):
        self.info = """
            The inputter is a hotspice.io.OOPSquareChessStepsInputter, and the outputreader
            is a hotspice.io.OOPSquareChessOutputReader. The inputter is designed to move domain
            walls without achieving saturation after each input bit, the outputter is designed to
            separately recognize the two degenerate ground states of OOP_Square ASI.
        """
        # WARN: E_B_std will cause random variation throughout the sweep, one can keep the randomness without this in-sweep variation by using E_B_std=0 and defining E_B directly as randomized once when initializing the sweep
        kwargs = {
            'a': 190e-9, 'nx': 20, 'ny': 20, 'PBC': False,
            'T': 300, 'E_B_mean': hotspice.utils.eV_to_J(22.5), 'E_B_std': 0, 'moment': 1063242*1.5e-9*(math.pi*85e-9**2)*7,
            'ext_field': 0.015, 'res_x': 10, 'res_y': 10
        } | kwargs # Dict with all parameter values as tuples of length 1 or greater
        names = {
            'a': "Lattice spacing", 'nx': "$n_x$", 'ny': "$n_y", 'PBC': "Periodic Boundary Conditions", 
            'T': "Temperature", 'E_B_mean': "Mean energy barrier", 'E_B_std': "Energy barrier standard deviation", 'E_B': "Energy barrier",
            'moment': "Magnetic moment", 'ext_field': "External stimulus magnitude", 'res_x': "# readout nodes along x-axis", 'res_y': '# readout nodes along y-axis'
        }
        units = {'a': 'm', 'T': 'K', 'E_B_mean': 'J', 'E_B_std': '%', 'E_B': 'J', 'moment': 'Am²', 'ext_field': 'T'}
        super().__init__(groups=groups, names=names, units=units, **kwargs)

    def create_experiment(self, params: dict) -> hotspice.experiments.KernelQualityExperiment:
        # If E_B in params, it will override E_B_mean and E_B_std (if E_B is absent, mean and std create random E_B)
        E_B = params.get('E_B', params['E_B_mean']*np.random.normal(1, params['E_B_std'], size=(nx, ny)))
        mm = hotspice.ASI.OOP_Square(params['a'], params['nx'], ny=params['ny'], T=params['T'], E_B=E_B, moment=params['moment'], PBC=params['PBC'],
            energies=(hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()),
            params=hotspice.SimParams(UPDATE_SCHEME=hotspice.Scheme.NEEL, SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF=0)) # Need Néel for inputter
        datastream = hotspice.io.RandomBinaryDatastream()
        inputter = hotspice.io.OOPSquareChessStepsInputter(datastream, magnitude=params['ext_field'], n=2, frequency=1) # Frequency does not matter as long as it is nonzero and reasonable
        outputreader = hotspice.io.OOPSquareChessOutputReader(params['res_x'], params['res_y'], mm)
        experiment = hotspice.experiments.KernelQualityExperiment(inputter, outputreader, mm)
        return experiment


## From Alex' Mathematica document:
#! Do not put this in an 'if __name__ == "__main__"' block! <sweep> variable must be importable!
dist_min, dist_max = 220e-9, 260e-9
# dist_range = 230e-9 # Single decent value
# dist_range = tuple(np.linspace(dist_min, dist_max, 16)) # For linear sweep of 'lattice parameter'
dist_range = tuple(np.linspace(dist_min**-3, dist_max**-3, 16)**(-1/3)) # For linear sweep of 'interaction strength'

field_min, field_max = 66e-3, 75e-3
field_range = tuple(np.linspace(field_min, field_max, 16))

# E_B = (hotspice.utils.eV_to_J(110)*np.random.normal(1, 0.05, size=(nx, ny)),)
E_B = hotspice.utils.eV_to_J(110)
# E_B_std = 0
E_B_std = 0.05
# E_B_std = tuple(np.linspace(0, 0.05, 16))

nx = ny = 20
pixel_size = 2


sweep = Sweep_OOPSquare_AFM2_KQ_byteInput(groups=[('res_x', 'res_y')],
    nx=nx, ny=ny,
    res_x=math.ceil(nx/pixel_size), res_y=math.ceil(ny/pixel_size),
    ext_field=field_range,
    a=dist_range,
    E_B_mean=E_B, # Randomness in E_B, put in a tuple since E_B can be a Field
    E_B_std=E_B_std
)


def process_single(sweep: hotspice.experiments.Sweep, iteration: int, run_kwargs=None, save_dir: Literal[False]|str = False):
    # TODO: at this point in time, I don't think this function has any reason to be in this file
    #       But where to put it? Perhaps as a classmethod of Sweep()? Putting it in ParallelJobs seems very hard but more logical...
    vars, experiment = sweep.get_iteration(iteration)
    if run_kwargs is None: run_kwargs = {}
    experiment.run(**run_kwargs)

    ## Collect and save the (meta)data of this iteration in the output directory
    description = experiment.__doc__
    if description is None: description = experiment.__init__.__doc__
    if description is None: description = sweep.info
    metadata = {
        'description': description,
        'sweep': sweep.as_metadata_dict()
    }
    constants = {
        'E_B': experiment.mm.E_B,
        '_experiment_run_kwargs': run_kwargs
        } | sweep.constants | vars # (<vars> are variable values in this iteration, so it is ok to put in 'constants')
    df_i = experiment.to_dataframe()
    data_i = hotspice.utils.Data(df_i, metadata=metadata, constants=constants)
    if save_dir:
        num_digits = len(str(len(sweep)-1)) # It is important to include iteration number or other unique identifier in savename (timestamps can be same for different finishing GPU processes)
        savename = str(sweep.groups).replace('"', '').replace("'", "") + "_" + str(iteration).zfill(num_digits) # zfill pads with zeros to the left
        saved_path = data_i.save(dir=save_dir, name=savename, timestamp=True)
        hotspice.utils.log(f"Saved iteration #{iteration} to {saved_path}", style='success')


if __name__ == "__main__":
    ## Define, parse and clean command-line arguments
    # Usage: python <this_file.py> [-h] [-o [OUTDIR]] [N]
    import argparse # Only parse args if __name__ == "__main__", otherwise args are not meant for this script
    parser = argparse.ArgumentParser(description="Process an iteration of the sweep defined in this file. If not specified, the data in --outdir are summarized into a single data file using calculate_all().")
    parser.add_argument('iteration', metavar='N', type=int, nargs='?', default=None,
                        help="the index of the sweep-iteration to be performed.")
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, nargs='?', default=None)
    args = parser.parse_args()
    outdir = args.outdir if args.outdir is not None else "results/Sweeps"

    if args.iteration is not None: # Then one specific iteration of sweep() should be calculated on the first available GPU/CPU.
        save = True
        experiment_run_kwargs = {
            'input_length': 100,
            'verbose': True,
            'pattern': 'afm'
        }
        process_single(sweep, args.iteration, run_kwargs=experiment_run_kwargs, save_dir=outdir if save else False)


    if args.iteration is None: # This gets run if the file is called without command line arguments
        if args.outdir is None:
            sweepsdir = "../../results/Sweeps/"
            args.outdir = sweepsdir + next(os.walk(sweepsdir))[1][-1] # Newest directory in sweepsdir
        summary_savepath = sweep.load_results(args.outdir, save=True, verbose=True, return_savepath=True)
        sweep.plot(summary_savepath)
