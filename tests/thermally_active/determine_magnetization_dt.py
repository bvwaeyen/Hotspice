"""
Created 26/04/2023, edited 08/05/2023, expanded 09/05/2023
Figured out a lot of parameters, but dt needs to be determined more precisely. The idea is to do both thermal decay and
magnetization and fit an exponential function to magnetization plots. This needs to be done on average to eliminate the
thermal noise.
Magnetization half life depends on magnetic field strength H. It would be ideal to have this be around the same time as
thermal half life. Time to vary H, fit function, and plot tau(H).
"""

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from thermally_active_system import *


zeeman = hotspice.ZeemanEnergy(magnitude=0, angle=0)
mm.add_energy(zeeman)

t_total = 15e-9
samples = 100  # need around 100 or so
# H_list = np.arange(0.5, 2.01, 0.1) * 1e-3  # 0.5 mT to 2 mT  # TODO
H_list = [1e-3]  # only this in detail for now  TODO: run this

# Using progress()
t_progress = 0.1e-9  # 0.05 ns  Should not go too low
MCsteps_max = 10
steps = int(t_total / t_progress)

create_new_data = True
directory = "Magnetization/"


if create_new_data:
    os.makedirs(directory, exist_ok=True)

    for H in H_list:
        print(f"Calculating H = {H * 1000 :.2f} mT")

        for sample in range(0, samples):
            print(f"Calculating sample {sample + 1}/{samples}")

            file = open(directory + f"H {H} sample {sample}.txt", "w")  # open file

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

# fitting function
def magn(t, tau, M):
    return M * (1 - np.power(2, -t/tau))

# calculate averages
summary_file = open("Summary.txt", "w")

for H in H_list:
    ms = np.zeros(steps+1)
    for sample in range(samples):
        times, m_avg_x = np.loadtxt(directory + f"H {H} sample {sample}.txt", unpack=True)
        ms += m_avg_x
    ms /= samples

    # times = 1e9 * np.arange(0, t_total+t_progress, t_progress)  # work in nanosecond scale
    times *= 1e9  # work in nanosecond scale

    fit_params = curve_fit(magn, times, ms)
    tau, M = fit_params[0]  # in nanoseconds
    print(f"For {H * 1000 :.2f} mT found half life tau = {tau:.5f} ns with maximal magnetization {M:.4f} or {100 * np.sqrt(2) * M :.2f}%")
    print(f"{H} {tau * 1e-9} {M}", file=summary_file)  # save result

    # plot
    plt.plot(times, ms, label="Average measurement")
    plt.plot(times, magn(times, tau, M), label="Fit")
    plt.plot(times, np.zeros_like(times), color="grey", zorder=0)
    plt.title(f"Magnetization of thermally relaxed state with {H*1000} mT field with {samples} samples")
    plt.ylabel("Average horizontal magnetization"); plt.xlabel("Time (ns)")
    plt.legend()

    plt.show()

summary_file.close()

# ==================================================
# Summary

Hs, taus, Ms = np.loadtxt("Summary.txt", unpack=True)

plt.plot(Hs * 1000, taus * 1e9)
plt.title("Magnetization half lives for different H")
plt.ylabel("Half life (ns)"); plt.xlabel("Magnetic field strength (mT)")
plt.show()

plt.plot(Hs * 1000, 100 * np.sqrt(2) * Ms)
plt.title("Magnetization saturation for different H")
plt.ylabel("Magnetization (%)"); plt.xlabel("Magnetic field strength (mT)")
plt.ylim(0, 100)
plt.show()