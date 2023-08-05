# Created 04/08/2023 to visually inspect the thermally active behaviour
# but heavily based on test_thermally_active_regime.py of 07/03/2023

import hotspice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools


# PARAMETERS
n = 21
E_B = hotspice.utils.eV_to_J(25e-3)
np.random.seed(0)
E_B_distr = E_B * np.random.normal(1, 0.05, (n, n))
pattern = "vortex"
mm = hotspice.ASI.IP_Pinwheel(a=21e-9, n=n, T=300, E_B=E_B_distr, moment=hotspice.utils.eV_to_J(2.5), pattern=pattern)
mm.params.UPDATE_SCHEME = "Néel"

# input
dt = 1.35e-9  # 1.35 ns
B = 3.0e-3  # 3 mT
angle = 7. * np.pi/180.  # 7 degrees to break symmetry

zeeman = hotspice.ZeemanEnergy(magnitude=B, angle=angle)
mm.add_energy(zeeman)



number_of_bits = 3  # number of bits per 'point'
repeats = 3  # number of times the same point needs to be repeatedly tested
unique_points_list = list(itertools.product([0, 1], repeat=number_of_bits))  # all binary combinations
unique_points_list = [point[::-1] for point in unique_points_list]  # reverse elements, so similar inputs are plotted closer together
number_of_unique_points = len(unique_points_list)

datastream = hotspice.io.BinaryListDatastream(unique_points_list, periodic=True)
inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=B, angle=angle, n=np.inf, relax=False, frequency=1./dt, half_period=True)


# Repeating bit_string to see what it gives each time

avg = hotspice.plottools.Average.resolve(True, mm)  # average for rgb

total_fig, total_axs = plt.subplots(repeats, number_of_unique_points)  # shows only last stage of bit input
for point_nr, point in enumerate(unique_points_list): total_axs[0, point_nr].set_title("".join([str(b) for b in point]))
for repeat in range(repeats):
    [[xmin, ymin], [xmax, ymax]] = total_axs[repeat, 0].get_position()._points
    total_fig.text(0.8*xmin, 0.5 * (ymin + ymax), f"Repeat {repeat+1}", va="center", rotation="vertical")

for point_nr, point in enumerate(unique_points_list):
    print(f"Running {point_nr+1}/{number_of_unique_points}: {point}")

    fig, axs = plt.subplots(repeats, number_of_bits+1)  # axs shows every repeat and their progress
    fig.suptitle(f"Input progress of {''.join([str(b) for b in point])} repeated {repeats} times")

    for repeat in range(repeats):
        mm.initialize_m(pattern)
        rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
        axs[repeat, 0].imshow(rgb)  # show initial
        axs[repeat, 0].set_title("Initial")
        axs[repeat, 0].axis("off")

        for bit_nr, bit in enumerate(point):
            inputter.input(mm, bit)  # input next bit
            rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
            axs[repeat, 1+bit_nr].imshow(rgb)  # show change
            axs[repeat, 1+bit_nr].set_title("".join([str(b) for b in point[:bit_nr+1]]))  # let know what you did
            axs[repeat, 1+bit_nr].axis("off")

        # show total input
        rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
        total_axs[repeat, point_nr].imshow(rgb)
        total_axs[repeat, point_nr].axis("off")

    #fig.show()

total_fig.show()

plt.show()  # does not show anything, but needs to pause, otherwise the program ends

