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
---
For error estimation, I try to split into bags, fit each bag, then use spread as error?
"""

import hotspice
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import os

# PARAMETERS
n = 21  # 220 spins
update_scheme = "Néel"
E_B = hotspice.utils.eV_to_J(25e-3)  # same as E_T
randomness = 0.05
pattern = "uniform"

mm = hotspice.ASI.IP_Pinwheel(21e-9, n, E_B=E_B, T=300, moment=hotspice.utils.eV_to_J(2.5), pattern=pattern, PBC=False)
mm.params.UPDATE_SCHEME = "Néel"

# THERMAL RELAXATION
pattern = "uniform"
t_total = 15e-9  # was 50e-9
samples = 200

# Using progress()
t_progress = 0.1e-9  # was 0.1 ns
steps = int(t_total / t_progress)

create_new_data = False
directory = "Thermal Relaxation/"

if create_new_data:
    os.makedirs(directory, exist_ok=True)

    for sample in range(0, samples):
        print(f"Calculating sample {sample + 1}/{samples}")

        file = open(directory + f"sample {sample}.txt", "w")  # open file

        # Initialize new sample
        np.random.seed(sample)
        mm.initialize_m(pattern=pattern)
        mm.E_B = E_B * np.random.normal(1, randomness, size=(n, n))

        print(0.0, mm.m_avg_x, file=file)  # initial magnetization

        for step in range(steps):
            dt, dMCsteps = mm.progress(t_max=t_progress, MCsteps_max=np.inf)  # progress a bit
            print((1+step) * t_progress, mm.m_avg_x, file=file)  # write down measurement

        file.close()

    print("Done creating data!")

# ==================================================
# Done creating new data, time to do calculations

t0 = 0.5  # amount of ns to skip, can be quite low, has a small impact

def decay(t, tau, m0):
    return m0 * np.power(2, -t/tau)


# unpack data
ms = np.zeros((steps+1, samples))
for sample in range(samples):
    times, m_avg_x = np.loadtxt(directory + f"sample {sample}.txt", unpack=True)
    ms[:, sample] = m_avg_x

# -------------------------
# total average

mean = np.mean(ms, axis=-1)
min, max = np.min(ms, axis=-1), np.max(ms, axis=-1)
q1, q3 = np.quantile(ms, q=0.25, axis=-1), np.quantile(ms, q=0.75, axis=-1)

# times = 1e9 * np.arange(0, t_total+t_progress, t_progress)  # work in nanosecond scale
times *= 1e9  # work in nanosecond scale

skip = int(t0 / (t_progress * 1e9))  # steps to skip in the fit

# calculate fit
fit_params = curve_fit(decay, times[skip:], mean[skip:])
tau, m0 = fit_params[0]  # in nanoseconds
tau_err, m0_err = np.sqrt(np.diag(fit_params[1]))  # standard deviation is sqrt of variance, which is diag of covariance matrix
print(f"Found half life tau = {tau} ns +- {tau_err} ns without remaining m_inf")

# plot
matplotlib.rcParams.update({'font.size': 12})  # larger font
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_decay = False
if plot_decay:
    plt.plot(times, mean, label="Average", color="b")
    plt.fill_between(times, q1, q3, label="Quartiles", color="b", alpha=0.5)
    plt.fill_between(times, min, max, label="Extrema", color="b", alpha=0.2)
    plt.plot(times[skip:], decay(times[skip:], tau, m0), label="Fit", color="r")
    plt.plot([0, times[-1]], [0, 0], color="k", zorder=-1)

    plt.legend()
    plt.ylabel("Average horizontal magnetization"); plt.xlabel("Time (ns)")
    plt.xlim(0, times[-1]); plt.ylim(np.min(min), 1/np.sqrt(2))

    plt.show()

# -------------------------
# Bag error estimation

bags = 5
samples_per_bag = samples // bags

bag_taus, bag_ms = np.zeros(bags), np.zeros(bags)

for bag in range(bags):
    bag_mean = np.mean(ms[:, bag*samples_per_bag:(bag+1)*samples_per_bag], axis=-1)

    # calculate fit
    bag_fit_params = curve_fit(decay, times[skip:], bag_mean[skip:])
    bag_taus[bag], bag_ms[bag] = bag_fit_params[0]  # in nanoseconds

print(bag_taus)
print(f"With bags found average {np.mean(bag_taus)} ns +- {np.std(bag_taus)} ns")
