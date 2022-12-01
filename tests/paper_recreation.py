# Created 29/11 by Ian Lateur

import numpy as np
import math
import time
import os
import hotspice

# PARAMETERS
n = 21  # 220 actual spins
T = 0  # deterministic
# relax = True  # deterministic and very fast, although actually not deterministic because partly random
relax = False  # minimize does not work
frequency = 0
MCS = 2
input_length = 20  # amount of bits of one signal

# default values from Magnets
Msat = 800e3  # magnetisation saturation, default value of Magnets
V = 2e-22  # nucleation volume, seems large?
H_coercive = 55e-3  # Coercive field. In paper this is 200mT, but they use Stoner-Wohlfarth model to update
E_B = 0.5 * H_coercive * Msat * V  # needs an extra factor 1/2 according to Jonathan Maes

angle = 7*math.pi/180  # used to be almost symmetric
n_output = 5  # ~~full~~ best resolution
verbose = True

# a_array = np.arange(20, 1020, 20) * 1e-9  # 20nm to 1000nm, paper has ~215nm to 1000nm
H_array = np.arange(46, 55, 1) * 1e-3  # 50mT to 100mT
# nsamples = 5  # paper has 30 samples, is a bit much maybe, so 5?
# H_array = np.arange(75, 80, 2) * 1e-3
a_array = [300e-9]
# H_array = np.arange(38, 52, 2) * 1e-3  # seemed interesting from magnetization effect test

nsamples = 3

# files
directory = "paper/"
if not os.path.exists(directory):
    os.makedirs(directory)
file = open(directory + "long_summary.txt", "w")
print("# a H sample K G Q k g q ComputationTime[s]", file=file)
file.close()
file = open(directory + "average_summary.txt", "w")
print("# a H K G Q k g q", file=file)
file.close()

# IMPLEMENTATION OF EXPERIMENT
for a in a_array:  # lattice parameters
    for H in H_array:  # magnetic field magnitude of input

        average_results = {"K": 0, "G": 0, "k": 0, "g": 0}

        # Preparing experiment
        mm = hotspice.ASI.IP_Pinwheel(a, n, T=T, E_B=E_B, Msat=Msat, V=V)  # recreate mm to change a
        mm.add_energy(hotspice.ZeemanEnergy())

        datastream = hotspice.io.RandomBinaryDatastream()
        inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=H, angle=angle, relax=relax, frequency=frequency, n=MCS)
        outputreader = hotspice.io.RegionalOutputReader(n_output, n_output, mm)  # Uses mm, so can't be reused
        experiment = hotspice.experiments.KernelQualityExperiment(inputter, outputreader, mm)  # uses mm, can't be reused

        for sample in range(nsamples):
            print(f"Calculating a = {a*1e9:.0f}nm H = {H*1000:.1f}mT sample {sample+1}/{nsamples}...")

            # Running Experiment
            start_time = time.time()
            experiment.run(pattern="uniform", verbose=verbose, input_length=input_length)  # Run experiment
            end_time = time.time()
            ComputationTime = int(end_time - start_time)

            # Log results
            experiment.calculate_all()  # Calculate results
            results = experiment.results
            K, G, Q, k, g, q = results["K"], results["G"], results["K"] - results["G"], results["k"], results["g"], results["k"] - results["g"]
            for key, value in results.items():
                average_results[key] += value/nsamples
            # save experiment
            exp_df = experiment.to_dataframe()
            exp_data = hotspice.utils.Data(exp_df)
            exp_data.save(dir=directory, name=f"KernelQualityExperiment a={a} H={H} sample {sample} ")
            # save summary
            file = open(directory+"long_summary.txt", "a")
            print(a, H, sample, K, G, Q, k, g, q, ComputationTime, file=file)
            file.close()

            print(f"Done, K={K} G={G} Q={Q}, q = {q}, this took {ComputationTime} seconds.\n")


        # save average summary
        file = open(directory+"average_summary.txt", "a")
        K, G, Q = average_results["K"], average_results["G"], average_results["K"] - average_results["G"]
        k, g, q = average_results["k"], average_results["g"], average_results["k"] - average_results["g"]
        print(a, H, K, G, Q, k, g, q, file=file)
        file.close()
