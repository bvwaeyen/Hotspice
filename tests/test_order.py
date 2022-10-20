# A program to roughly determine the order parameter m_avg in function of temperature T of multiple geometries.
# Created 20/10/2022

import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

import hotspin
from hotspin.ASI import *

def get_mag_stats(mm: hotspin.core.Magnets, T, MCS: int, nsamples: int = 1):
    """Starts from a uniform grid at temperature T, it updates MCS*number_of_spins times and returns order parameter m_avg.
    If nsamples is more than 1, it will do this multiple times and return the statistical mean and standard deviation."""

    mm.T = T
    unique, counts = np.unique(mm.occupation, return_counts=True)
    number_of_spins = dict(zip(unique, counts))[1]  # counts the ammount of used spins in system
    m_array = np.zeros(nsamples)

    for sample in range(nsamples):
        mm.initialize_m(pattern="uniform")

        for step in range(int(number_of_spins * MCS)):  # on average every spin gets 1 update per MCS
            mm.update()

        m_array[sample] = mm.m_avg

    if nsamples==1:
        return m_array[0]

    return np.mean(m_array), np.std(m_array)

def sweep_order_temperature(filename: str, mm: hotspin.core.Magnets, T_array, MCS: int, nsamples: int = 1):
    """Sweeps over range of temperature and dumps order parameter statistics in file."""

    if not filename.endswith(".txt"):
        filename += ".txt"

    file = open(filename, "w")

    # header
    print(f"# {mm.nx}x{mm.ny} {type(mm).__name__} {'with' if mm.PBC else 'without'} PBC, a={mm.a}, " +
          f"{MCS} MC steps to relax{f' with {nsamples} samples per T' if nsamples>1 else ''}.", file=file)
    print(f"# T {'m_avg' if nsamples==1 else 'm_mean m_std'}", file=file)

    for T in T_array:
        output = get_mag_stats(mm, T, MCS, nsamples)  # the actual test

        if nsamples==1:
            print(T, output, file=file)
        else:
            print(T, output[0], output[1], file=file)

    file.close()

def plot_order_T(filename):
    """Plots data from file, shows plot and returns (fig, ax)"""

    if not filename.endswith(".txt"):
        filename += ".txt"

    file = open(filename, "r")
    header = file.readline()[2:]  # skip '# '
    output = np.loadtxt(filename, unpack=True)
    file.close()

    T, m = output[0], output[1]  # arrays of data

    fig, ax = plt.subplots()
    ax.plot(T, m, label="Average magnetization")
    if len(output) == 3: ax.plot(T, output[2], label="Standard deviation")  # standard deviation included
    ax.legend()
    ax.set_title("\n".join(wrap("Order parameter of " + header, 60)))     # long title can wrap around
    ax.set_xlabel("T[K]")

    return fig, ax

#-----------------------------------------------------------------------------------------------------------------------

a = 1e-06
n = 20
PBC=True
MCS = 5
samples = 5

T_array = np.arange(0, 1001, 10)
T_array[0] = 1  # don't do 0K

# List of ASI to do
mm_list = [IP_Ising(a, n, PBC=PBC), IP_SquareDiamond(a, n, PBC=PBC), IP_PinwheelDiamond(a, n, PBC=PBC),
           IP_Kagome(a, int(8/3 * n) - int(8/3 * 20)%4, PBC=PBC), IP_Triangle(a, int(8/3 * n) - int(8/3 * 20)%4, PBC=PBC)]

for mm in mm_list:
    filename = f"Rough {type(mm).__name__} order parameter sweep"
    print("Currently doing", filename)
    sweep_order_temperature(filename, mm, T_array, MCS, nsamples=samples)  # produce the data
    fig, ax = plot_order_T(filename)  # plot the data
    fig.savefig(filename+".png")  # save the plot

plt.show()