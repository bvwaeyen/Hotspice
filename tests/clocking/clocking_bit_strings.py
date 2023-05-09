# Created 27/03/2023 by Ian Lateur

import hotspice
import numpy as np
import itertools
import matplotlib.pyplot as plt

# PARAMETERS
a = 124e-9  # 124 nm  seems very close together
a *= 2.0  # otherwise too bound  # Let's try the original again
n = 101  # 5100 actual spins; too large for real time
V = 4.4e-22  # 220nm length; 80nm width; 25nm thickness;  Gets some randomness later on
T = 1  # deterministic
E_B = hotspice.utils.eV_to_J(110)  # from Jonathan Maes from other paper
# PBC = True  # I was unable to get clocking results with a boundary
PBC = False  # It would be nice if this worked
pattern = "vortex"  # more fair for 0 and 1? Will it still move or will it be stuck?

# randomness
randomness = 0.05  # 5% seems to be around the sweet spot. Higher is too many random small domains, lower is too homogenous
random_distr = np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))
random_distr[random_distr < 0.0] = 1.0  # do not have negative E_B
E_B_distr = E_B * random_distr

lattice_angle = 12. * np.pi/180.  # the magic sauce? TODO Should be between 0 and 12°

mm = hotspice.ASI.IP_Pinwheel(a, n, V=V, E_B=E_B_distr, T=T, pattern=pattern, PBC=PBC, angle=lattice_angle + np.pi/2.0)  # FIXME gave it an extra 90° to make it symmetrical
mm.params.UPDATE_SCHEME = "Néel"  # better at low temperatures and high E_B; should maybe even use relax() ?
zeeman = hotspice.ZeemanEnergy()
mm.add_energy(zeeman)

# ==================================================
# it turns out to be fully deterministic (at least to the eye)  # TODO: check with numbers?


number_of_bits = 5  # number of bits per 'point'
repeat = 1  # number of times the same point needs to be repeatedly tested
unique_points_list = list(itertools.product([0, 1], repeat=number_of_bits))  # all binary combinations
number_of_unique_points = len(unique_points_list)

data_list = []
for point in unique_points_list:
    for _ in range(repeat):
        data_list += list(point)

datastream = hotspice.io.BinaryListDatastream(data_list)
H = 56.9e-3  # TODO: sweep H?
inputter = hotspice.io.ClockingFieldInputter(datastream, magnitude=H, angle=lattice_angle)
outputreader = hotspice.io.FullOutputReader(mm)  # Full output instead of regional
# outputreader = hotspice.io.RegionalOutputReader(5, 5, mm)  # averaging to reduce spatial noise

experiment = hotspice.experiments.IODistanceExperiment(inputter, outputreader, mm)

experiment.run(repeat * number_of_unique_points, number_of_bits, verbose=True, pattern=pattern)
experiment.calculate_all(input_metric="hamming", output_metric="hamming")  # Need euclidian if using RegionalOutputReader!
experiment.plot_corankning_matrix()
experiment.plot_shepard_diagram()

