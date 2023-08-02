# Created 02/08/2023 to show difference in behaviour for different lattice spacings a, but now with statistics
import hotspice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# PARAMETERS
n = 21
update_scheme = "Néel"
T = 300  # about room temperature;  this yields ~25.85 eV
E_B = hotspice.utils.eV_to_J(25e-3)  # same as E_T
moment = hotspice.utils.eV_to_J(2.5)  # 400 times smaller than usual ; to get zeeman energy of order 2.5meV when H = 1 mT
randomness = 0.05

# for now
relax_time = 50e-9  # 50 ns, is enough if smallest a becomes vortex
a_array = np.arange(17, 25)  * 1e-9
samples = 20

dir = "tuned_E_d/"
os.makedirs(dir, exist_ok=True)
def filename(a, sample):
    return f"a {a*1e9:.0f} nm, sample {sample} after {relax_time*1e9:.0f} ns.npy"

calculate = False
if calculate:
    for sample in range(samples):
        np.random.seed(sample)
        E_B_distr = E_B * np.random.normal(1, randomness, size=(n, n))
        for a in a_array:  # always same E_B distribution
            print(f"Calculating a = {int(np.round(a * 1e9))} nm, sample {sample}")
            if os.path.exists(dir + filename(a, sample)):
                print(f"File already exists! Skipping this one!")
                continue

            mm = hotspice.ASI.IP_Pinwheel(a, n, E_B=E_B_distr, T=T, moment=moment, pattern="uniform", PBC=False)
            mm.params.UPDATE_SCHEME = update_scheme  # NEVER FORGET

            mm.progress(t_max=relax_time, MCsteps_max=np.inf)
            np.save(dir+filename(a, sample), mm.m)



# --------------------------------------------------
# Analysation

matplotlib.rcParams.update({'font.size': 12})  # larger font

Ed_data = np.zeros(shape=(a_array.size, samples))
for a_i, a in enumerate(a_array):
    mm_dummy = hotspice.ASI.IP_Pinwheel(a, n, E_B=E_B, T=T, moment=moment, pattern="uniform", PBC=False)
    for sample in range(samples):
        mm_dummy.m = np.load(dir + filename(a, sample), "r")
        E_d = mm_dummy.get_energy("dipolar")
        E_d.update()  # need to recalculate after manual reassignment of mm.m
        Ed_data[a_i, sample] = -hotspice.utils.J_to_eV(np.average(E_d.E[np.nonzero(E_d.E)])) * 1000  # meV

fig, ax = plt.subplots()
ax.boxplot([Ed_data[a_i, :] for a_i in range(a_array.size)], positions=[int(a) for a in a_array*1e9], flierprops={"marker":(4, 2, 45), "alpha":0.5})
ax.plot([16.5, 24.5], [25, 25], ls="--", color="r", label="target")  # show preferred line at 25meV
ax.set_xlabel("a (nm)")
ax.set_ylabel(r"$|E_d|$ (meV)")
ax.legend()

plt.show()
