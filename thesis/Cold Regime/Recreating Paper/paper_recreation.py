# Created 29/11 by Ian Lateur

import numpy as np
import pandas as pd
import math
import os
import hotspice

# PARAMETERS
n = 21  # 220 actual spins
T = 1  # almost deterministic
relax = False  # minimize does not work
frequency = 1  # 1 Hz
input_length = 30  # amount of bits of one signal  FIXME: used to be 100
update_scheme = "Néel"  # better for low temperatures

moment = 3e-16  # from Jonathan Maes, comes from ratio of flatspin alpha and hotspice a
E_B = hotspice.utils.eV_to_J(110)  # from Jonathan Maes

angle = 7*math.pi/180  # used to be almost symmetric

# B_min, B_max, B_step = 70, 110, 2  # mT
# B_array = np.arange(B_min, B_max + B_step, B_step) * 1e-3  # at 90mT no effect, at 100mT last bit completely saturates everything
B_array = [76e-3, 86e-3, 96e-3]  # two interesting transitions? and a position in between

samples = 1  # paper has 30 samples, is a bit much maybe, so 5?
a_array = [500e-9]  # pretty interesting region on avalanche figure

verbose = True

# files
directory = "redoing_paper_KQ/"
if not os.path.exists(directory):
    os.makedirs(directory)

def filename(a, B, sample):
    return f"KQ a {a*1e9:.0f} nm, B {B*1e3:.1f} mT, sample {sample}.json"

# IMPLEMENTATION OF EXPERIMENT

for sample in range(samples):
    np.random.seed(sample)  # sample number is seed
    E_B_distr = E_B *np.random.normal(1, 0.05, size=(n, n))  # for randomness

    for a in a_array:  # lattice parameters
        for B in B_array:  # magnetic field magnitude of input
            print(f"Calculating a = {a*1e9:.0f}nm B = {B*1e3:.1f}mT sample {sample}/{samples}...")

            if os.path.exists(directory + filename(a, B, sample)):
                print(f"File already exists! Skipping this one!")
                continue

            # Preparing experiment
            mm = hotspice.ASI.IP_Pinwheel(a, n, T=T, E_B=E_B_distr, moment=moment)  # recreate mm to change a
            mm.params.UPDATE_SCHEME = update_scheme
            mm.add_energy(hotspice.ZeemanEnergy())

            datastream = hotspice.io.RandomBinaryDatastream()
            inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=B, angle=angle, relax=relax, frequency=frequency, n=np.inf, half_period=False)  # NO HALF PERIOD
            fulloutputreader = hotspice.io.FullOutputReader(mm)
            # outputreader = hotspice.io.RegionalOutputReader(n_output, n_output, mm)  # Uses mm, so can't be reused
            experiment = hotspice.experiments.KernelQualityExperiment(inputter, fulloutputreader, mm)  # uses mm, can't be reused

            experiment.run(pattern="uniform", input_length=input_length, verbose=verbose)  # Run experiment
            experiment.to_dataframe().to_json(directory + filename(a, B, sample))  # save

            # show results
            experiment.calculate_all()  # Calculate results
            results = experiment.results
            K, G, Q, k, g, q = results["K"], results["G"], results["K"] - results["G"], results["k"], results["g"], results["k"] - results["g"]
            print(f"Done, K={K} G={G} Q={Q}, q = {q}")

