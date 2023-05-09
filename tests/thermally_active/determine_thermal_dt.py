"""
Created 25/04/2023, edited 07/05/2023
Figured out a lot of parameters, but dt needs to be determined more precisely. The idea is to do both thermal decay and
magnetization and fit an exponential function to magnetization plots. This needs to be done on average to eliminate the
thermal noise.
But how to perform an average when update() steps are not a set length? Use progress() with set t_max resolution? This
can eliminate longer updates from happening, especially if t_max is chosen too small. But otherwise the only possibility
is to do a very noisy fit...
I will simply try to do both and see if there is a significant difference.
There is...
I have opted to use progress() instead of update()
"""

import hotspice
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from thermally_active_system import *  # system

# THERMAL RELAXATION
t_total = 15e-9  # was 50e-9
samples = 100

# Using progress()
t_progress = 0.1e-9  # was 0.1 ns
steps = int(t_total / t_progress)
MCsteps_max = 10

create_new_data = True
directory = "Thermal Relaxation/"


if create_new_data:
    os.makedirs(directory, exist_ok=True)

    for sample in range(0, samples):
        print(f"Calculating sample {sample + 1}/{samples}")

        file = open(directory + f"sample {sample}.txt", "w")  # open file

        # Initialize new sample
        mm.initialize_m(pattern=pattern)
        mm.E_B = E_B * np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))

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

t0 = 1.  # amount of ns to skip, can be quite low, has a small impact

def decay(t, tau, m0):
    return m0 * np.power(2, -(t - t0)/tau)

def decay_remaining(t, tau, m0, m_inf):  # TODO not needed, remove me
    return decay(t, tau, m0) + m_inf

# Is not super necessary, but does seem to fit a bit better
def total_half_life(tau, m0, m_inf):
    """Half life of total magnetization. <tau> is the half life of the additional magnetisation (so minus m_inf)"""
    return tau * np.log2(2 / (1 - m_inf/m0))

# calculate average
ms = np.zeros(steps+1)
for sample in range(samples):
    times, m_avg_x = np.loadtxt(directory + f"sample {sample}.txt", unpack=True)
    ms += m_avg_x
ms /= samples

# times = 1e9 * np.arange(0, t_total+t_progress, t_progress)  # work in nanosecond scale
times *= 1e9  # work in nanosecond scale


skip = int(t0 / (t_progress * 1e9))  # steps to skip in the fit

# calculate fit without m_inf
fit_params = curve_fit(decay, times[skip:], ms[skip:])
tau, m0 = fit_params[0]  # in nanoseconds
print(f"Found half life tau = {tau} ns without remaining m_inf")

# calculate fit with remaining m_inf
fit_params = curve_fit(decay_remaining, times[skip:], ms[skip:])
tau2, m02, m_inf = fit_params[0]  # in nanoseconds
print(f"Found additional half life = {tau2} ns or total half life {total_half_life(tau2, m02, m_inf)} ns and remaining m_inf = {m_inf}")

# plot
plt.plot(times, ms, label="Average measurement")
plt.plot(times[skip:], decay(times[skip:], tau, m0), label="Fit decay")
plt.plot(times[skip:], decay_remaining(times[skip:], tau2, m02, m_inf), label="Fit decay+m")
plt.plot(times, np.zeros_like(times), color="grey", zorder=0)
plt.title(f"Thermal decay of horizontal magnetization with {samples} samples")
plt.legend()
plt.ylabel("Average horizontal magnetization"); plt.xlabel("Time (ns)")

plt.show()
