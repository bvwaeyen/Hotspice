# A program to determine the kernel quality for various temperatures.
# Created 26/10/2022
import pandas as pd

print("Importing...")
import math
import numpy as np
import time
import matplotlib.pyplot as plt

import hotspice
print("Everything imported")


def sweep_HTQ(experiment: hotspice.experiments.KernelQualityExperiment, H_array, T_array, verbose=False, experiment_dir: str = None):
    """
    @param experiment: KernelQualityExperiment to run for different H and T
    @param H_array: list of magnetic fields to sweep
    @param T_array: list of temperatures to sweep
    @param verbose: verbosity of experiment
    @param experiment_dir: If not None, saves the individual experiments in this directory.
    @return: Data from full sweep
    """

    mm = experiment.mm

    H_len, T_len = len(H_array), len(T_array)
    l = H_len*T_len  # data_dict length
    data_dict = {"H": [0]*l, "T": [0]*l, "K": [0]*l, "G": [0]*l, "Q": [0]*l, "k": [0]*l, "g": [0]*l, "q": [0]*l, "ComputationTime[s]": [0]*l}
    constants = {"Grid": f"{mm.nx}x{mm.ny}", "Geometry": type(mm).__name__, "a": mm.a, "PBC": mm.PBC, "angle": f"{experiment.inputter.angle/math.pi*180:.0f}°",
                 "MCS_per_input": experiment.inputter.n, "Perceptron": f"{experiment.outputreader.nx}x{experiment.outputreader.ny}"}

    for h, H in enumerate(H_array):
        experiment.inputter.magnitude = H                   # Set magnetic field H

        for t, T in enumerate(T_array):
            experiment.mm.T = T                             # Set temperature T

            print(f"Running H = {H}, T = {T}")
            start_time = time.time()
            experiment.run(pattern="uniform", verbose=verbose)  # Run experiment
            experiment.calculate_all()                      # Calculate results
            end_time = time.time()

            if experiment_dir is not None:
                exp_df = experiment.to_dataframe()
                exp_data = hotspice.utils.Data(exp_df)
                exp_data.save(dir=experiment_dir, name=f"KernelQualityExperiment H={H} T={T} ")

            results = experiment.results
            K, G, Q, k, g, q = results["K"], results["G"], results["K"] - results["G"], results["k"], results["g"], results["k"] - results["g"]
            ComputationTime = int(end_time - start_time)
            print(f"Done. This took {ComputationTime} seconds.\n" +
                  f"k = {k}, g = {g}, q = {q}{'    HOORAY!!!' if q > 0 else ''}\n")

            index = h*T_len + t                             # Log results
            data_dict["H"][index], data_dict["T"][index] = H, T
            data_dict["K"][index], data_dict["G"][index], data_dict["Q"][index] = K, G, Q
            data_dict["k"][index], data_dict["g"][index], data_dict["q"][index] = k, g, q
            data_dict["ComputationTime[s]"][index] = ComputationTime

    df = pd.DataFrame(data=data_dict)
    data = hotspice.utils.Data(df, constants)

    return data


#-----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    a = 300e-09
    n = 10
    PBC=False
    angle = (45+7)*math.pi/180  # angle of magnetic field, 45° for Diamond geometry

    n_output = 5

    MCS = 4  # seems to (barely) saturate around here from HTm plots
    frequency = 0  # Hz
    H_array = np.arange(5, 8.1, 1) * 1e-4  # seemed to be semi interesting from Néel experiments
    T_array = [1, 3000, 5000]  # cold, start of phase transition, at phase transition?
    filename = "Glauber KQ 300 nm"
    experiment_dir = filename
    verbose = True
    relax = False

    mm = hotspice.ASI.IP_PinwheelDiamond(a, n, PBC=PBC)
    mm.add_energy(hotspice.ZeemanEnergy())
    mm.params.UPDATE_SCHEME = "Glauber"

    datastream = hotspice.io.RandomBinaryDatastream()
    inputter = hotspice.io.PerpFieldInputter(datastream, angle=angle, n=MCS, frequency=frequency, relax=relax)
    outputreader = hotspice.io.RegionalOutputReader(n_output, n_output, mm)
    experiment = hotspice.experiments.KernelQualityExperiment(inputter, outputreader, mm)

    data = sweep_HTQ(experiment, H_array, T_array, verbose=verbose, experiment_dir=experiment_dir)
    data.save(name=filename, dir=experiment_dir)
