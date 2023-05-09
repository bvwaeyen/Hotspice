# Created 27/02/2023 by Ian Lateur
# A simple IODistanceExperiment test
# parameters *inspired* by paper recreation (with creative liberty)
# CHECK EVERYTHING AGAIN IF YOU WANT TO OFFICIALLY "RECREATE" THE PAPER

import numpy as np
import math
import hotspice
import matplotlib.pyplot as plt


def exponential_hamming_metric(u, v, a=1.3):
    hamming = np.invert(u == v)
    N = u.size
    norm = 1/( (1-a**N)/(1-a) - N )
    return norm * np.sum(a**hamming - 1)

def last_bit_metric(u, v):
    return u[-1] != v[-1]

# PARAMETERS
number_of_points = 6
input_length = 6  # one byte

# n = 21  # 220 actual spins
n = 11  # smaller than paper for make it 4 times faster
T = 5000  # maybe too high/low
relax = False  # minimize does not work
frequency = np.float64(0)  # Hz, only important for Néel
MCS = 5  # only important for Glauber. Could be way too short. May be infinity compared to IRL
update_scheme = "Glauber"  # Better for higher temperatures, idk

# default values from Magnets
moment = 3e-16  # from Jonathan Maes, comes from ratio of flatspin alpha and hotspice a
E_B = hotspice.utils.eV_to_J(110)  # from Jonathan Maes
# E_B = E_B * np.ones((n, n))*np.random.normal(1, 0.05, size=(n, n))  # for randomness

angle = 7*math.pi/180  # used to be almost symmetric
verbose = True  # for debugging

a = 300e-9  # 300 nm
H = 100e-3  # idk

# Preparing experiment
mm = hotspice.ASI.IP_Pinwheel(a, n, T=T, E_B=E_B, moment=moment, pattern="uniform")
dummy_mm = hotspice.ASI.IP_Pinwheel(a, n, T=T, E_B=E_B, moment=moment)
mm.params.UPDATE_SCHEME = update_scheme
mm.add_energy(hotspice.ZeemanEnergy())

datastream = hotspice.io.RandomBinaryDatastream()
inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=H, angle=angle, relax=relax, frequency=frequency, n=MCS)
outputreader = hotspice.io.FullOutputReader(mm)  # Full output instead of regional
experiment = hotspice.experiments.IODistanceExperiment(inputter, outputreader, mm)

experiment.run(number_of_points, input_length, verbose=verbose, pattern="uniform")

for input, output in zip(experiment.input_sequences, experiment.output_sequences):
    print("\ninput:")
    print(input)
    print("output:")
    print(output)
    dummy_mm.m[np.nonzero(dummy_mm.m)] = output
    hotspice.plottools.show_m(dummy_mm)

# for last bit metric
print("First showing last bit metric")
experiment.calculate_all(input_metric=last_bit_metric, output_metric="hamming")
experiment.plot_corankning_matrix()
experiment.plot_shepard_diagram(input_metric="Last Bit", xlim=(-0.05,1.05), ylim=(-0.05,1.05))

# for hamming metric
print("Second showing normal hamming metric")
experiment.calculate_all(input_metric="hamming", output_metric="hamming")
experiment.plot_corankning_matrix()
experiment.plot_shepard_diagram(xlim=(-0.05,1.05), ylim=(-0.05,1.05))

# plotting trustworthyness and continuity
Q = experiment.coranking_matrix
T = hotspice.coranking.trustworthiness_range(Q)
C = hotspice.coranking.continuity_range(Q)
plt.plot(range(1, T.size+1), T, label="T")
plt.plot(range(1, T.size+1), C, label="C")
plt.ylim(0, 1)
plt.legend()
plt.show()