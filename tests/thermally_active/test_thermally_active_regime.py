# Created 27/03/2023 by Ian Lateur

import hotspice
import numpy as np
import itertools
import matplotlib.pyplot as plt

# PARAMETERS
T = 300  # about room temperature
E_B = hotspice.utils.eV_to_J(25e-3)  # Room temp energy barier about 23meV
moment = hotspice.utils.eV_to_J(25)  # 40 times smaller than usual ; to get zeeman energy of order 25meV when H = 1 mT
n = 50  # probably need large systems for thermally active regime?
update_scheme = "Néel"  # Need to do stuff with time scales
# PBC = True  # use small systems to simulate large systems ; and vortex is not ground state anymore ; but is less physical
PBC = False  # works and is more physical
pattern = "uniform"
# dipolar energy scales with 1e-7 * moment²/a³ (with extra factor for summation over all spins)
# choice of a depends on n, PBC and pattern
a = 100e-9  # get ~25eV dipolar energy at thermally relaxed configuration  TODO: sweep a

# randomness in E_B still recommended
randomness = 0.05
random_distr = np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))
E_B = E_B * random_distr  # in E_B

mm = hotspice.ASI.IP_Pinwheel(a, n, E_B=E_B, T=T, moment=moment, pattern=pattern, PBC=PBC)
mm.params.UPDATE_SCHEME = update_scheme

mm.add_energy(hotspice.ZeemanEnergy(0, 0))

# realtime = hotspice.plottools.RealTime(mm, E_B_range=None, T_range=None, moment_range=None, H_range=(0, 10e-3))
# realtime.run()


def exponential_hamming_metric(u, v, a=1.3):
    hamming = np.invert(u == v)
    N = u.size
    norm = 1/( (1-a**N)/(1-a) - N )
    return norm * np.sum(a**hamming - 1)

def last_bit_metric(u, v):
    return u[-1] != v[-1]

number_of_bits = 3  # number of bits per 'point'
repeat = 3  # number of times the same point needs to be repeatedly tested
unique_points_list = list(itertools.product([0, 1], repeat=number_of_bits))  # all binary combinations
number_of_unique_points = len(unique_points_list)

data_list = []
for point in unique_points_list:
    for _ in range(repeat):
        data_list += list(point)

datastream = hotspice.io.BinaryListDatastream(data_list)
H = 0.1e-3  # 1 mT  TODO: figure out how low it should be, 1 mT is too much
# TODO: If H is smaller anyway, maybe Previous2DFieldInputter is not needed, as flipping happens via intermediate path
# FIXME: previous_value is not gooddesign anyway and I do not know how to properly fix it.
frequency = 1.0 / (1e-9)  # used to be 0.1 ns  TODO: re-figure-out time
max_MCsteps = 10  # plenty of time
angle = 0  # TODO maybe should be np.pi/8?
inputter = hotspice.io.FieldInputterBinary(datastream, magnitudes=(-H, H), angle=angle, n=max_MCsteps, frequency=frequency, half_period=True)
# Using new Previous2DFieldInputter  FIXME: previous_value is not good design
# inputter = hotspice.io.Previous2DFieldInputter(datastream, magnitude=H, n=max_MCsteps, frequency=frequency)
# outputreader = hotspice.io.FullOutputReader(mm)  # Full output instead of regional
# outputreader = hotspice.io.RegionalOutputReader(5, 5, mm)  # averaging to reduce spatial noise
outputreader = hotspice.io.CorrelationLengthOutputReader(mm)  # only correlation length

"""
experiment = hotspice.experiments.IODistanceExperiment(inputter, outputreader, mm)

experiment.run(repeat * number_of_unique_points, number_of_bits, verbose=True, pattern=pattern)
experiment.calculate_all(input_metric="hamming", output_metric="euclidean")  # Need euclidian if using RegionalOutputReader!
experiment.plot_corankning_matrix()
experiment.plot_shepard_diagram()
"""


# Repeating bit_string to see what it gives each time

avg = hotspice.plottools.Average.resolve(True, mm)  # average for rgb

total_fig, total_axs = plt.subplots(repeat, number_of_unique_points)  # shows only last stage of bit input
total_fig.suptitle(f"Total result of {number_of_bits} bit inputs with {repeat} repeats")

for point_nr, point in enumerate(unique_points_list):
    print(f"Running {point_nr+1}/{number_of_unique_points}: {point}")

    fig, axs = plt.subplots(repeat, number_of_bits+1)  # axs shows every repeat and their progress
    fig.suptitle(f"Input progress of {''.join([str(b) for b in point])} repeated {repeat} times")

    for r in range(repeat):
        mm.initialize_m("uniform")
        try:
            inputter.previous_value = 0  # FIXME: this should not be needed :(
        except:
            pass
        rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
        axs[r, 0].imshow(rgb)  # show initial
        axs[r, 0].set_title("Initial")
        axs[r, 0].axis("off")

        for bit in range(number_of_bits):
            inputter.input(mm)  # input next bit
            rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
            axs[r, 1+bit].imshow(rgb)  # show change
            axs[r, 1+bit].set_title("".join([str(b) for b in point[:bit+1]]))  # let know what you did
            axs[r, 1+bit].axis("off")

        # show total input
        rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
        total_axs[r, point_nr].imshow(rgb)
        total_axs[r, point_nr].set_title("".join([str(b) for b in point]))
        total_axs[r, point_nr].axis("off")

    #fig.show()

total_fig.show()

plt.show()  # does not show anything, but needs to pause, otherwise the program ends



"""
For a cool shepard diagram and coranking matrix, use itertools.product([0, 1], repeat=dimension)
for every binary vector in a space of dimension dimension.
use BinaryListDatastream, FieldInputterBinary and FullOutputReader
use the "hamming" metric for both spaces
"""

"""
A problem:
You should absolutely repeat same points multiple times, because the ASI mapping is RANDOM! (yes also for Néel)
BUT coranking matrix is not designed for non-unique points! meaning point p wil rank as 1 2 3 4 for itself if repeated 5 times,
but going to the mapping, it might go to 4 2 3 1 which is arbitrary and TOTALLY OKAY, but will hurt performance metrics!

Possible solution? Only look at K-ary neighbourhoods with K > #repeats ???
"""