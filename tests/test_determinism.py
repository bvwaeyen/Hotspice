# Created 01/12

import numpy as np
import math
import time
import os
import hotspice

# PARAMETERS
n = 21  # 220 actual spins
relax = False  # minimize does not work
frequency = 0
MCS = 10
input_length = 20  # amount of bits of one signal

# default values from Magnets
Msat = 800e3  # magnetisation saturation, default value of Magnets
V = 2e-22  # nucleation volume, seems large?
H_coercive = 55e-3  # Coercive field. In paper this is 200mT, but they use Stoner-Wohlfarth model to update
E_B = 0.5 * H_coercive * Msat * V  # needs an extra factor 1/2 according to Jonathan Maes
print(E_B)

angle = 7*math.pi/180  # used to be almost symmetric
n_output = 5  # ~~full~~ best resolution
verbose = True

a = 300e-9
H = 46e-3

nsamples = 3
input_number = 1
input_length = 20

# Preparing experiment
mm = hotspice.ASI.IP_Pinwheel(a, n, E_B=E_B, Msat=Msat, V=V, pattern="uniform")
mm.add_energy(hotspice.ZeemanEnergy())
mm.params.UPDATE_SCHEME = "Glauber"

datastream = hotspice.io.RandomBinaryDatastream()
inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=H, angle=angle, relax=relax, frequency=frequency, n=MCS)
outputreader = hotspice.io.RegionalOutputReader(n_output, n_output, mm)  # Uses mm, so can't be reused
experiment = hotspice.experiments.DeterminismExperiment(inputter, outputreader, mm)  # uses mm, can't be reused

mm.T = 1e-5
score_list = []


for sample in range(nsamples):
    # Running experiment
    experiment.run(input_number=input_number, input_length=20, pattern="uniform", verbose=verbose)

    # Tally results
    experiment.calculate_all()
    results = experiment.results

    hotspice.plottools.show_m(mm)

    score_list.append(results['D'])

print(f"On average {np.average(score_list):.2f} unique results of {input_number} identical signals.")

