"""
Created 26/04/2023, edited 08/05/2023, expanded 09/05/2023
Figured out a lot of parameters, but dt needs to be determined more precisely. The idea is to do both thermal decay and
magnetization and fit an exponential function to magnetization plots. This needs to be done on average to eliminate the
thermal noise.
Magnetization half life depends on magnetic field strength H. It would be ideal to have this be around the same time as
thermal half life. Time to vary H, fit function, and plot tau(H).
"""

import hotspice
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams.update({'font.size': 12})  # larger font
import matplotlib.pyplot as plt
import os

# PARAMETERS
n = 21  # 220 spins
E_B = hotspice.utils.eV_to_J(25e-3)  # same as E_T
randomness = 0.05

mm = hotspice.ASI.IP_Pinwheel(21e-9, n, E_B=E_B, T=300, moment=hotspice.utils.eV_to_J(2.5), PBC=False)
mm.params.UPDATE_SCHEME = "Néel"

zeeman = hotspice.ZeemanEnergy(magnitude=0, angle=0)
mm.add_energy(zeeman)

# magnetization
t_total = 15e-9
samples = 200
B_min, B_max, dB = 0.5, 5.0, 0.5  # in mT
B_list = np.arange(B_min, B_max + 0.5*dB, dB) * 1e-3

# Using progress()
t_progress = 0.05e-9  # 0.05 ns  Should not go too low
MCsteps_max = np.inf
steps = int(t_total / t_progress)

directory = "Magnetization Vortex/"
create_new_data = False

if create_new_data:
    os.makedirs(directory, exist_ok=True)

    for sample in range(samples):
        # Initialize new sample
        print(f"Calculating sample {sample + 1}/{samples}")

        np.random.seed(sample)
        mm.E_B = E_B * np.random.normal(1, randomness, size=(n, n))  # new E_B

        for B in B_list:
            print(f"Calculating B = {B * 1000 :.2f} mT")
            if os.path.exists(directory + f"B {B*1e3:.2f} mT, sample {sample}.txt"):
                print(f"File already exists! Skipping this one! Delete if you want to recalculate")
                continue

            file = open(directory + f"B {B*1e3:.2f} mT, sample {sample}.txt", "w")  # open file

            mm.initialize_m(pattern="vortex")  # ground state
            zeeman.set_field(B, 0)  # set magnetic field

            print(0.0, mm.m_avg_x, file=file)  # initial magnetization

            for step in range(steps):
                dt, dMCsteps = mm.progress(t_max=t_progress, MCsteps_max=MCsteps_max)  # progress a bit
                print((1+step) * t_progress, mm.m_avg_x, file=file)  # write down measurement

            file.close()

    print("Done creating data!")

# ==================================================
# Done creating new data, time to do calculations

# fitting function
def magn(t, tau, M):
    return M * (1 - np.power(2, -t/tau))

# retrieve data
ms = np.zeros((steps+1, len(B_list), samples))
for B_i, B in enumerate(B_list):
    for sample in range(samples):
        times, m_avg_x = np.loadtxt(directory + f"B {B*1e3:.2f} mT, sample {sample}.txt", unpack=True)
        ms[:, B_i, sample] = m_avg_x

times *= 1e9  # work in nanosecond scale
mean = np.mean(ms, axis=-1)
min, max = np.min(ms, axis=-1), np.max(ms, axis=-1)
q1, q3 = np.quantile(ms, q=0.25, axis=-1), np.quantile(ms, q=0.75, axis=-1)

# --------------------------------------------------
# calculate taus and Ms
# should there be a skip here aswell? hmmnah?

taus, Ms = np.zeros(len(B_list)), np.zeros(len(B_list))
tau_errs, M_errs = np.zeros(len(B_list)), np.zeros(len(B_list))
for B_i, B in enumerate(B_list):
    fit_params = curve_fit(magn, times, mean[:, B_i])
    taus[B_i], Ms[B_i] = fit_params[0]  # in nanoseconds
    tau_errs[B_i], M_errs[B_i] = np.sqrt(np.diag(fit_params[1]))  # errors

    print(f"For {B * 1000 :.2f} mT found half life tau = {taus[B_i]:.5f} ns +- {tau_errs[B_i]:.5f} ns with maximal magnetization" +
          f" {Ms[B_i]:.4f} +- {M_errs[B_i]:.4f} or {100 * np.sqrt(2) * Ms[B_i] :.2f}% += {100 * np.sqrt(2) * M_errs[B_i] :.2f}%")

# --------------------------------------------------
# make individual plots to see what is going on

plot_evolution = True
if plot_evolution:
    for B_i, B in enumerate(B_list):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(times, mean[:, B_i], label="Average", color="b")
        ax.fill_between(times, q1[:, B_i], q3[:, B_i], label="Quartiles", color="b", alpha=0.5)
        ax.fill_between(times, min[:, B_i], max[:, B_i], label="Extrema", color="b", alpha=0.2)
        ax.plot(times, magn(times, taus[B_i], Ms[B_i]), label="Fit", color="r")

        ax.plot(times, np.zeros_like(times), color="k", zorder=0)  # 0 line
        ax.set_title(f"B = {B*1000:.2f} mT, {samples} samples")
        ax.set_ylabel("Average horizontal magnetization"); ax.set_xlabel("Time (ns)")
        ax.set_xlim(0, times[-1]); ax.set_ylim(0, 1/np.sqrt(2))
        ax.legend()

    plt.show()


# ==================================================
# Bag Summary
print("--------------------------------------------------\nBags")

bags = 5
samples_per_bag = samples // bags

bag_taus, bag_Ms = np.zeros((len(B_list), bags)), np.zeros((len(B_list), bags))

for B_i, B in enumerate(B_list):
    for bag in range(bags):
        bag_mean = np.mean(ms[:, B_i, bag*samples_per_bag:(bag+1)*samples_per_bag], axis=-1)

        # calculate fit
        bag_fit_params = curve_fit(magn, times, bag_mean)
        bag_taus[B_i, bag], bag_Ms[B_i, bag] = bag_fit_params[0]  # in nanoseconds

    print(f"For {B * 1000 :.2f} mT found half life tau = {np.mean(bag_taus[B_i]):.5f} ns +- {np.std(bag_taus[B_i]):.5f} ns with maximal magnetization" +
          f" {np.mean(bag_Ms[B_i]):.4f} +- {np.std(bag_Ms[B_i]):.4f} or {100 * np.sqrt(2) * np.mean(bag_Ms[B_i]) :.2f}% += {100 * np.sqrt(2) * np.std(bag_Ms[B_i]) :.2f}%")

# tau(B)
fig_bag_tau, ax_bag_tau = plt.subplots(figsize=(5, 4))
ax_bag_tau.errorbar(B_list*1e3, np.mean(bag_taus, axis=-1), yerr=np.std(bag_taus, axis=-1), marker="o", color="b",label=r"Bag fit $\overline{\tau}_B$", zorder=-1)
ax_bag_tau.scatter(B_list*1e3, taus, marker="x", color="g", label=r"Total fit $\tau_B$")
ax_bag_tau.plot([B_min - 0.5*dB, B_max + 0.5*dB], [1.35, 1.35], ls="--", color="r", label=r"$\overline{\tau}_T$")  # show preferred line
ax_bag_tau.fill_between([B_min - 0.5*dB, B_max + 0.5*dB], [1.35-0.07]*2, [1.35+0.07]*2, color="r", alpha=0.2, ls="--", label=r"$\sigma(\tau_T)$")
ax_bag_tau.set_xlabel("B (mT)")
ax_bag_tau.set_ylabel(r"$\tau_B$ (ns)")
ax_bag_tau.set_xlim([B_min - 0.5*dB, B_max + 0.5*dB])
handles, labels = ax_bag_tau.get_legend_handles_labels()
order = [0, 3, 1, 2]
ax_bag_tau.legend([handles[i] for i in order], [labels[i] for i in order])


# M(B)
fig_bag_M, ax_bag_M = plt.subplots(figsize=(5, 4))
ax_bag_M.errorbar(B_list*1e3, np.mean(bag_Ms, axis=-1), yerr=np.std(bag_Ms, axis=-1), marker="o", color="b", label=r"Bag fit", zorder=-1)
ax_bag_M.scatter(B_list*1e3, Ms, marker="x", color="g", label=r"Total fit")
ax_bag_M.set_xlabel("B (mT)")
ax_bag_M.set_ylabel("$m_\infty$")
ax_bag_M.set_ylim(0, 1/np.sqrt(2))
ax_bag_M.legend()


plt.show()
