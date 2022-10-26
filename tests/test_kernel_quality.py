# A program to determine the kernel quality for various temperatures.
# Created 26/10/2022
import pandas as pd

print("Importing...")
import math
import numpy as np
import time
import matplotlib.pyplot as plt

import hotspin
print("Everything imported")


def sweep_HTQ(experiment: hotspin.experiments.KernelQualityExperiment, H_array, T_array):

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
            experiment.run(pattern="uniform")               # Run experiment
            experiment.calculate_all()                      # Calculate results
            end_time = time.time()

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
    data = hotspin.utils.Data(df, constants)

    return data


#-----------------------------------------------------------------------------------------------------------------------

a = 1e-06
n = 25
PBC=False
angle = (180)*math.pi/180  # angle of magnetic field, 45° for Diamond geometry

n_perceptron = 5
# H_array = np.arange(2, 12, 2) * 1e-6
H_array = [1e-6, 1e-5, 1e-4]
T_array = np.arange(200, 750, 50)

T = 200
H = 4e-6
MCS = 2  # Monte Carlo Steps per input single

filename = "Kernel Quality Rough Test "

mm = hotspin.ASI.IP_PinwheelDiamond(a, n, PBC=PBC, energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()))
datastream = hotspin.io.RandomBinaryDatastream()
inputter = hotspin.io.PerpFieldInputter(datastream, angle=angle, n=MCS, frequency=0)
outputreader = hotspin.io.RegionalOutputReader(n_perceptron, n_perceptron, mm)
experiment = hotspin.experiments.KernelQualityExperiment(inputter, outputreader, mm)

#data = sweep_HTQ(experiment, H_array, T_array)
#data.save(name=filename)


# debug code
experiment.inputter.magnitude = 2e-6
experiment.run()
experiment.calculate_all()

df = experiment.to_dataframe()
print(experiment.results)