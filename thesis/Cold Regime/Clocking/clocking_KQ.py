# Created 14/07 to run KQ experiment with clocking

import hotspice
import numpy as np
import os

# ASI
a = 248e-9  # 248 nm, fixed
n = 21  # 220 spins
moment = 3e-16  # from Jonathan Maes
E_B = hotspice.utils.eV_to_J(110)  # add randomness per sample
randomness = 0.05
T = 1  # cold
update_scheme = "Néel"
lattice_angle = 0. * np.pi/180.  # TODO: manually vary in {0°, 4°, 8°, 12°}

mm = hotspice.ASI.IP_Pinwheel(a, n, moment=moment, E_B=E_B, T=T, angle=lattice_angle)
mm.params.UPDATE_SCHEME = update_scheme
mm.add_energy(hotspice.ZeemanEnergy())

# input
dt = 1.  # 1 second
spread = np.pi/4.  # 45°, NOT 22.5°
B_min, B_max, dB = 55, 63, 0.1  # in mT
B_array = np.arange(B_min, B_max + dB, dB) * 1e-3  # in T

# experiment
samples = 20  # also determines seed of E_B (and Néel and the KQ Experiments I suppose)
m = 220  # is total number of spins
constant_fraction = 0.6  # last 60% of GC is constant
input_length = 30  # is this enough? too big? too late now
pattern = "uniform"

# experimental setup
datastream = hotspice.io.RandomBinaryDatastream()  # input will always be random
inputter = hotspice.io.ClockingFieldInputter(datastream=datastream, magnitude=B_min, angle=lattice_angle, spread=spread,
                                             frequency=1./dt, n=np.inf, relax=False)  # magnitude changed later
fulloutputreader = hotspice.io.FullOutputReader(mm) # Full resolution
experiment = hotspice.experiments.KernelQualityExperiment(inputter, fulloutputreader, mm)

# directory stuff
directory = f"KQ/angle {int(lattice_angle * 180./np.pi)}/"
os.makedirs(directory, exist_ok=True)
def filename(B, sample):
    return f"KQ B {B*1e3:.1f} mT, sample {sample}.json"


# Make new data
make_new_data = False
if make_new_data:
    print("Creating new data")

    for sample in range(0, samples):
        np.random.seed(sample)  # sample number is seed
        E_B_distr = E_B * np.ones((n, n)) * np.random.normal(1, randomness, size=(n, n))
        mm.E_B = E_B_distr

        for B in B_array:
            inputter.magnitude = B
            print(f"Calculating B = {B * 1e3 :.1f} mT, sample {sample}/{samples}")

            if os.path.exists(directory + filename(B, sample)):
                print(f"File already exists! Skipping this one!")
                continue

            experiment.run(input_length=input_length, constant_fraction=constant_fraction, pattern=pattern)  # run
            experiment.to_dataframe().to_json(directory + filename(B, sample))  # save

            # Calculate and show results
            experiment.calculate_all()
            results = experiment.results
            K, G, Q, k, g, q = results["K"], results["G"], results["K"] - results["G"], results["k"], results["g"], results["k"] - results["g"]
            print(f"Done, K={K} G={G} Q={Q}, q = {q}")

    print("Done creating data!")
