import math
import os

import hotspice


class SweepTAExperiment(hotspice.experiments.Sweep): # Example of a fully implemented Sweep subclass
    def __init__(self, groups=None, **kwargs): # If unsure, provide a kwarg as a tuple directly
        kwargs = {
            'ASI_type': hotspice.ASI.IP_Square, 'a': 320e-9, 'nx': 9, 'ny': 9, 'T': 300, 'E_B': hotspice.utils.eV_to_J(5),
            'PBC': False, 'moment': 220e-9*80e-9*25e-9*860e3,
            'H_min': 0.015, 'H_max': 0.017, 'H_angle': math.pi/4, 'res_x': 5, 'res_y': 5, 'sine': 100e6
            } | kwargs # Dict with all parameter values as tuples of length 1 or greater
        super().__init__(groups=groups, **kwargs)

    def create_experiment(self, params: dict) -> hotspice.experiments.TaskAgnosticExperiment:
        mm = params['ASI_type'](params['a'], params['nx'], ny=params['ny'], T=params['T'], E_B=params['E_B'], moment=params['moment'], PBC=params['PBC'],
            pattern='random', energies=(hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()), params=hotspice.SimParams(UPDATE_SCHEME=hotspice.Scheme.NEEL))
        datastream = hotspice.io.RandomBinaryDatastream()
        inputter = hotspice.io.FieldInputterBinary(datastream, magnitudes=(params['H_min'], params['H_max']), angle=params['H_angle'], n=2, sine=params['sine'])
        outputreader = hotspice.io.RegionalOutputReader(params['res_x'], params['res_y'], mm)
        experiment = hotspice.experiments.TaskAgnosticExperiment(inputter, outputreader, mm)
        return experiment


class SweepTA_RC_ASI(hotspice.experiments.Sweep):
    def __init__(self, groups=None, **kwargs): # If unsure, provide a kwarg as a tuple directly
        """ Performs a sweep as in `Reservoir Computing in Artificial Spin Ice` by J. H. Jensen and G. Tufte.
            i.e. with binary input and RegionalOutputReader of variable resolution, with PerpFieldInputter.
        """
        kwargs = { # TODO: issue with ASI_type is that user can't define ASI types themselves for this like this
            'ASI_type': 'IP_Pinwheel', 'a': 600e-9, 'nx': 21, 'ny': 21, 'T': 300, 'E_B': hotspice.utils.eV_to_J(71),
            'PBC': False, 'moment': 860e3*470e-9*170e-9*10e-9,
            'ext_field': 0.07, 'ext_angle': 7*math.pi/180, 'res_x': 5, 'res_y': 5, 'sine': False
            } | kwargs # Dict with all parameter values as tuples of length 1 or greater
        super().__init__(groups=groups, **kwargs)

    def create_experiment(self, params: dict) -> hotspice.experiments.TaskAgnosticExperiment:
        mm = getattr(hotspice.ASI, params['ASI_type'])(params['nx'], params['a'], ny=params['ny'], T=params['T'], E_B=params['E_B'], moment=params['moment'], PBC=params['PBC'],
            pattern='random', energies=(hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()), params=hotspice.SimParams(UPDATE_SCHEME=hotspice.Scheme.NEEL))
        datastream = hotspice.io.RandomBinaryDatastream()
        inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=params['ext_field'], angle=params['ext_angle'], n=2, sine=params['sine'])
        outputreader = hotspice.io.RegionalOutputReader(params['res_x'], params['res_y'], mm)
        experiment = hotspice.experiments.TaskAgnosticExperiment(inputter, outputreader, mm)
        return experiment


def TAsweep(sweep: hotspice.experiments.Sweep, iterations=1000, verbose=False, save=True):
    """ Performs a parameter sweep of a `TaskAgnosticExperiment`.
        (!ON A SINGLE GPU! To run on multiple GPU, use hotspice/scripts/ParallelJobs.py and an approprate secondary script like examples/SweepKQ_RC_ASI.py)
        @param sweep [hotspice.experiments.Sweep]: a `Sweep` instance that generates `TaskAgnosticExperiment` instances.
        @param iterations [int] (1000): the number of input bits applied for each value of parameters.
        @param save [bool|str] (True): if truthy, the sweep data is saved in "results/TaskAgnosticExperiment/Sweep.temp<timestamp>/".
            If specified as a string, the base name of all saved files in this directory is this string.
        @return [str]: absolute path to the directory where all iterations of this sweep are stored.
    """ # TODO: this function seems to have become obsolete since the creation of ParallelJobs.py
    if save: savename = save if isinstance(save, str) else str(sweep.groups).replace('"', '').replace("'", "")
    else: savename = ''
    temp_time = hotspice.utils.timestamp()
    temp_dir = f"results/TaskAgnosticExperiment/Sweep.temp{temp_time}"
    vars: dict
    experiment: hotspice.experiments.TaskAgnosticExperiment
    for i, (vars, experiment) in enumerate(sweep):
        if verbose: hotspice.utils.log(f"Variables in this iteration: {vars}\nConstants: {sweep.constants}")
        experiment.run(N=iterations, verbose=verbose)

        # Preliminary save of the data in this iteration, just in case an error would occur later on 
        df_i = experiment.to_dataframe()
        for i, (varname, value) in enumerate(vars.items()): # Set variable columns to the value they had in this iteration
            df_i.insert(loc=i, column=varname, value=(value,)*len(df_i))
        metadata = {
            'description': r"Contains the input values <u> and state vectors <y> as can be used for calculating task agnostic metrics of the system as proposed in `Task Agnostic Metrics for Reservoir Computing` by Love et al.",
            'sweep': sweep.as_metadata_dict()
        }
        constants = {'inputter': hotspice.utils.full_obj_name(experiment.inputter), 'outputreader': hotspice.utils.full_obj_name(experiment.outputreader), 'datastream': hotspice.utils.full_obj_name(experiment.inputter.datastream),
            'ASI_type': hotspice.utils.full_obj_name(experiment.mm)} | sweep.constants
        data_i = hotspice.utils.Data(df_i, metadata=metadata, constants=constants)
        if save: save = data_i.save(dir=temp_dir, name=savename, timestamp=True)
    
    return os.path.abspath(temp_dir)


if __name__ == "__main__":
    ## From 'Reservoir Computing in Artificial Spin Ice' by J. H. Jensen and G. Tufte:
    # res_range = tuple([i+1 for i in range(11)])
    res_range = 5
    dist_range = tuple([500e-9+i*50e-9 for i in range(10)])
    # dist_range = 600e-9
    field_range = tuple([0.06+i*0.002 for i in range(10)])
    # field_range = 0.07
    sweep = SweepTA_RC_ASI(groups=[('res_x', 'res_y')],
        res_x=res_range, res_y=res_range,
        ext_field=field_range,
        a=dist_range
    )
    # temp_dir = TAsweep(sweep, iterations=20, verbose=True, save=True)
    # TAsweep_load(temp_dir, save=True)

    ## THE SITUATION IN 'Computation in artificial spin ice' BY JENSEN ET AL.:
    # simparams = hotspice.SimParams(UPDATE_SCHEME=hotspice.Scheme.NEEL) # Needed because we are working with frequency
    # sweep_taskagnostic(hotspice.ASI.IP_Square, variables={}, params=simparams,
    #                    E_B=hotspice.utils.eV_to_J(5), T=300, V=220e-9*80e-9*25e-9, Msat=860e3, a=320e-9, n=9, PBC=False, pattern='random',
    #                    iterations=1000, ext_angle=math.pi/4, ext_magnitude=(0.015, 0.017), sine=100e6, verbose=1
    # ) # TODO: continue developing this Jensen et al. situation, with the frequency correctly being taken into account in the Néel scheme.
    #pass # TODO: sweep ext_magnitude, increase MCsteps_max or however this is called
