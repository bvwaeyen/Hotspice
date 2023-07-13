import hotspice
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
a = 248e-9  # 248 nm
n = 101  # 5100 actual spins
moment = 3e-16  # from Jonathan Maes, comes from ratio of flatspin alpha and hotspice aV
T = 1  # "deterministic"
E_B = hotspice.utils.eV_to_J(110)  # from Jonathan Maes from other paper

# randomness
randomness = 0.05  # seems to need more than 5% randomness
random_distr = np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))
E_B = E_B * random_distr  # in E_B

mm = hotspice.ASI.IP_Pinwheel(a, n, moment=moment, E_B=E_B, T=T, pattern="uniform")
mm.params.UPDATE_SCHEME = "Néel"  # better at low temperatures and high E_B; should maybe even use relax() ?
mm.add_energy(hotspice.ZeemanEnergy())

B = 60.5e-3
spread = np.pi/4.  # 45°, NOT 22.5° LIKE IN CLOCKING "PAPER"
bits = 6 * [1] + 6 * [0]

datastream = hotspice.io.BinaryListDatastream(bits, periodic=True)
inputter = hotspice.io.ClockingFieldInputter(datastream, magnitude=B, spread=spread)

fig, axs = plt.subplots(2, 1 + len(bits)//2)
flat_axs = axs.flatten()
avg = hotspice.plottools.Average.resolve(True, mm)  # average for rgb

rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
axs[0][0].imshow(rgb)
axs[0][0].set_title("initial")
axs[0][0].axis("off")

axs[1][0].imshow(np.ones_like(rgb))
axs[1][0].axis("off")

for i, bit in enumerate(bits):
    inputter.input(mm)

    rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
    axs_index = 2 - bit + i
    flat_axs[axs_index].imshow(rgb)
    flat_axs[axs_index].set_title("".join([str(bit) for bit in bits[:i+1]]))
    flat_axs[axs_index].axis("off")

plt.show()
