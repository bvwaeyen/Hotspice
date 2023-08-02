# Created 15/05/2023 to show difference in behaviour for different lattice spacings a
import hotspice
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
n = 21
update_scheme = "Néel"
T = 300  # about room temperature;  this yields ~25.85 eV
E_B = hotspice.utils.eV_to_J(25e-3)  # same as E_T
moment = hotspice.utils.eV_to_J(2.5)  # 400 times smaller than usual ; to get zeeman energy of order 2.5meV when H = 1 mT
randomness = 0.05
E_B_distr = E_B * np.random.normal(1, randomness, size=(n, n))

# for now
relax_time = 50e-9  # 50 ns, is enough if smallest a becomes vortex
a_array = np.arange(17, 25)  * 1e-9

fig, axs = plt.subplots(2, 4, figsize=(10., 4.8))

for i, a in enumerate(a_array):  # always same E_B distribution
    print(f"Calculating a = {int(np.round(a * 1e9))} nm")
    ax = axs.flatten()[i]

    mm = hotspice.ASI.IP_Pinwheel(a, n, E_B=E_B_distr, T=T, moment=moment, pattern="uniform", PBC=False)
    mm.params.UPDATE_SCHEME = update_scheme  # NEVER FORGET

    mm.progress(t_max=relax_time, MCsteps_max=np.inf)  # 50ns is enough for the smallest a

    avg = hotspice.plottools.Average.resolve(True, mm)
    rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)

    E_d = mm.get_energy("dipolar").E
    E_d_avg = hotspice.utils.J_to_eV(np.average(E_d[np.nonzero(E_d)])) * 1000  # meV

    im = ax.imshow(rgb, origin="lower", cmap="hsv", vmin=0., vmax=2.*np.pi)
    ax.axis("off")
    ax.set_title(f"a = {int(np.round(a * 1e9))} nm\n$E_d$ = {E_d_avg :.2f} meV")

# manually add colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.70])
cbar = fig.colorbar(im, cax=cbar_ax, ticks=[i * np.pi/2. for i in range(0, 5)])
cbar.ax.set_yticklabels([f"{i * 90}$^\circ$" for i in range(0, 5)])

plt.show()