"""
Created 26/04/2023, edited 08/05/2023
Figured out a lot of parameters, but dt needs to be determined more precisely. The idea is to do both thermal decay and
magnetization and fit an exponential function to magnetization plots. This needs to be done on average to eliminate the
thermal noise.
I have opted to use progress() instead of update().
This file contains code to do figure out magnetization dt.
Edit: cut off initial time and use Langevin function instead of tanh.
"""

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from thermally_active_system import *


zeeman = hotspice.ZeemanEnergy(magnitude=0, angle=0)
mm.add_energy(zeeman)

t_total = 15e-9
samples = 5  # FIXME

# Using progress()
t_progress = 0.1e-9  # 0.05 ns  Should not go too low
MCsteps_max = 10
steps = int(t_total / t_progress)

create_new_data = False
directory = "Magnetization/"


if create_new_data:
    os.makedirs(directory, exist_ok=True)

    for sample in range(0, samples):
        print(f"Calculating sample {sample + 1}/{samples}")

        file = open(directory + f"sample {sample}.txt", "w")  # open file

        # Initialize new sample
        mm.initialize_m(pattern="random")  # closest to thermally relaxed?
        mm.E_B = E_B * np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))  # new E_B
        zeeman.set_field(0, 0)  # remove magnetic field
        mm.progress(t_max=np.inf, MCsteps_max=MCsteps_max)  # thermal relaxation? TODO does this work?
        zeeman.set_field(H, 0)  # set magnetic field

        print(0.0, mm.m_avg_x, file=file)  # initial magnetization

        for step in range(steps):
            dt, dMCsteps = mm.progress(t_max=t_progress, MCsteps_max=MCsteps_max)  # progress a bit
            print((1+step) * t_progress, mm.m_avg_x, file=file)  # write down measurement

            if dMCsteps == MCsteps_max:
                print(f"WARNING! Only progressed {dt} s instead of {t_progress} s")

        file.close()

    print("Done creating data!")

# ==================================================
# Done creating new data, time to do calculations

t_skip = 0.  # amount of ns to skip  # TODO I don't think this is needed now?
skip = int(t_skip / (t_progress * 1e9))  # steps to skip in the fit


# calculate average
ms = np.zeros(steps+1)
for sample in range(samples):
    times, m_avg_x = np.loadtxt(directory + f"sample {sample}.txt", unpack=True)
    ms += m_avg_x
ms /= samples

# times = 1e9 * np.arange(0, t_total+t_progress, t_progress)  # work in nanosecond scale
times *= 1e9  # work in nanosecond scale

# fitting function
def magn(t, tau, M):
    return M * (1 - np.power(2, -t/tau))

fit_params = curve_fit(magn, times[skip:], ms[skip:])
tau, M = fit_params[0]  # in nanoseconds
print(f"Found half life tau = {tau} ns with maximal magnetization {M} or {100 * np.sqrt(2) * M}%")

# plot
plt.plot(times, ms, label="Average measurement")
plt.plot(times[skip:], magn(times[skip:], tau, M), label="Fit")
plt.plot(times, np.zeros_like(times), color="grey", zorder=0)
plt.title(f"Magnetization of thermally relaxed state with {H*1000} mT field with {samples} samples")
plt.ylabel("Average horizontal magnetization"); plt.xlabel("Time (ns)")
plt.legend()

plt.show()