# A program to determine the kernel quality for various temperatures.
# Created 26/10/2022

if __name__ == "__main__": print("Importing...")
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

import hotspice
if __name__ == "__main__": print("Everything imported")


def sample_HTQ(experiment: hotspice.experiments.KernelQualityExperiment, H, T, N,verbose=False, experiment_dir: str = None):
    """
    Sample N times at magnetic field strength H and temperature T.
    @param experiment: KernelQualityExperiment to run for different H and T.
    @param H: Magnetic field strength of inputter of experiment.
    @param T: Temperature of mm of experiment.
    @param verbose: verbosity of experiment
    @param experiment_dir: If not None, saves the individual experiments in this directory.
    @return: list of results, including ComputationTime[s]
    """
    experiment.inputter.magnitude = H
    experiment.mm.T = T

    result_list = [None]*N
    for sample in range(N):
        start_time = time.time()
        experiment.run(pattern="uniform", verbose=verbose)  # Run experiment
        experiment.calculate_all()  # Calculate results
        end_time = time.time()
        ComputationTime = int(end_time - start_time)

        results = experiment.results
        results["ComputationTime[s]"] = ComputationTime
        result_list[sample] = results

        if experiment_dir is not None:
            exp_df = experiment.to_dataframe()
            exp_data = hotspice.utils.Data(exp_df)
            exp_data.save(dir=experiment_dir, name=f"KernelQualityExperiment H={H} T={T} sample {sample} ")

    return result_list


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
            print(f"Done. This took {ComputationTime} seconds.\n" + f"k = {k}, g = {g}, q = {q}\n")

            index = h*T_len + t                             # Log results
            data_dict["H"][index], data_dict["T"][index] = H, T
            data_dict["K"][index], data_dict["G"][index], data_dict["Q"][index] = K, G, Q
            data_dict["k"][index], data_dict["g"][index], data_dict["q"][index] = k, g, q
            data_dict["ComputationTime[s]"][index] = ComputationTime

    df = pd.DataFrame(data=data_dict)
    data = hotspice.utils.Data(df, constants)

    return data


def plot_HTQ(data: hotspice.utils.Data):
    """ Plots kernel qualities of a set of experiments. Uses plot() if either T or H is constant, otherwise uses imshow().
    @param data: data of multiple experiments.
    @return: fig, axs
    """

    qualities = "kgq"
    H, T = data.get("H"), data.get("T")

    qualities_lists = []
    for quality in qualities:
        qualities_lists.append(data.get(quality))

    H_unique, T_unique = np.unique(H), np.unique(T)
    nH, nT = H_unique.size, T_unique.size

    if nH > 1 and nT > 1:  # 2D plots using imshow
        fig, axs = plt.subplots(ncols=3)

        # make extent of imshow, so that xtics can be correctly centered
        left = H_unique.min() - 0.5*(H_unique[1] - H_unique[0])
        right = H_unique.max() + 0.5*(H_unique[-1] - H_unique[-2])
        down = T_unique.min() - 0.5*(T_unique[1] - T_unique[0])
        up = T_unique.max() + 0.5*(T_unique[-1] - T_unique[-2])
        extent = [left, right, down, up]

        for ax, quality_list, quality in zip(axs.flatten(), qualities_lists, qualities):
            quality_mesh = np.reshape(quality_list, (nH, nT)).T  # make mesh as if made by meshgrid
            ax.imshow(quality_mesh, extent=extent)  # actual imshow
            # make ticks centered
            ax.set_xticks(H_unique)
            ax.set_yticks(T_unique)
            ax.set_title(quality)

    else:  # relevant 1D plots using plot()
        fig, ax = plt.subplots()
        axs = [ax]
        ax.set_ylim(0, 1)

        if nH == 1:
            x = T_unique
            ax.set_xlabel("T [K]")
        elif nT == 1:
            x = H_unique
            ax.set_xlabel("H [T]")
        else:
            raise Exception("Nothing to plot. Data is a single point.")

        for quality_list, quality, marker in zip(qualities_lists, qualities, ["s", "D", "o"]):
            ax.plot(x, quality_list, label=quality, marker=marker)  # Actual plot
            ax.legend()

    return fig, axs


#-----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    a = 300e-09
    n = 10
    PBC=False
    angle = (45+7)*math.pi/180  # angle of magnetic field, 45° for Diamond geometry
    n_output = 5

    MCS = 1  # 4 seemed to be too long. Maybe 1 is okay? Or should it be less?
    frequency = 0  # Hz
    # H_array = np.arange(7, 11.1, 0.2) * 1e-4  # Around 0.8 mT because of previous test
    # T_array = [5000]  # Most interesting temperature from previous test
    # filename = "KQ 5000K around 0p8mT"
    experiment_dir = "Samples of 1 MCS at 0p8mT 5000K"
    verbose = True
    relax = False

    mm = hotspice.ASI.IP_PinwheelDiamond(a, n, PBC=PBC)
    mm.add_energy(hotspice.ZeemanEnergy())
    mm.params.UPDATE_SCHEME = "Glauber"

    datastream = hotspice.io.RandomBinaryDatastream()
    inputter = hotspice.io.PerpFieldInputter(datastream, angle=angle, n=MCS, frequency=frequency, relax=relax)
    outputreader = hotspice.io.RegionalOutputReader(n_output, n_output, mm)
    experiment = hotspice.experiments.KernelQualityExperiment(inputter, outputreader, mm)

    result_list = sample_HTQ(experiment, 0.0008, 5000, 10, verbose=verbose, experiment_dir=experiment_dir)

    q_list = []
    for result in result_list:
        q = result["k"] - result["g"]
        q_list.append(q)
        print(f"q: {q}")

    print("-----")
    q_min, q_max = min(q_list), max(q_list)
    q_avg, q_std = np.mean(q_list), np.std(q_list)
    print(f"Minimum = {q_min}\nMaximum = {q_max}\nAverage = {q_avg}\nStandard Deviation = {q_std}")
