# Created 06/05/2023 to test the clocking mechanism with the addition of angles

import hotspice
import numpy as np
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
pattern = "uniform"

# randomness
randomness = 0.05  # 5% seems to be around the sweet spot. Higher is too many random small domains, lower is too homogenous
random_distr = np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))
random_distr[random_distr < 0.0] = 1.0  # do not have negative E_B
E_B_distr = E_B * random_distr

lattice_angle = 12. * np.pi/180.  # the magic sauce? TODO Should be between 0 and 12°
# TODO 0 definitely does not work. Consistently get avalanches. 12° has a tendency to "tie knots into itself"?

extra_angle = lattice_angle  # include lattice angle; should it be 0, lattice_angle, lattice_angle/2? Only lattice_angle works well

mm = hotspice.ASI.IP_Pinwheel(a, n, V=V, E_B=E_B_distr, T=T, pattern=pattern, PBC=PBC, angle=lattice_angle)
mm.params.UPDATE_SCHEME = "Néel"  # better at low temperatures and high E_B; should maybe even use relax() ?
zeeman = hotspice.ZeemanEnergy()
mm.add_energy(zeeman)


# realtime = hotspice.plottools.RealTime(mm, E_B_range=None, T_range=None, moment_range=None, H_range=(50e-3, 60e-3))
# realtime.run()

# Bits input tester
bits = 8 * [1] + 8 * [0]

datastream = hotspice.io.BinaryListDatastream(bits, periodic=True)
inputter = hotspice.io.ClockingFieldInputter(datastream, angle=extra_angle)


for H in np.arange(56.5, 57.2, 0.2) * 1e-3:  # TODO: sweep H?
    print(f"Testing H = {H*1000} mT")

    mm.initialize_m(pattern=pattern)
    inputter.magnitude = H

    fig, axs = plt.subplots(2, len(bits)//2)
    flat_axs = axs.flatten()
    avg = hotspice.plottools.Average.resolve(True, mm)  # average for rgb

    for i in range(len(bits)):
        inputter.input(mm)

        print(f"Inserted bit {bits[i]}")
        rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
        flat_axs[i].imshow(rgb)
        flat_axs[i].set_title("".join([str(bit) for bit in bits[:i+1]]))
        flat_axs[i].axis("off")

    plt.show()