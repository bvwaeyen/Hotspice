# A program to determine the effect of the magnetic field strength at various temperatures at various times.
# Created 25/10/2022

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap

import hotspin


def sweep_HTorder(mm: hotspin.core.Magnets, inputter: hotspin.io.FieldInputter, filename: str, H_array, T_array, MaxMCsteps):

    if not filename.endswith(".txt"):
        filename += ".txt"

    file = open(filename, "w")

    # header
    print(f"# {mm.nx}x{mm.ny} {type(mm).__name__} {'with' if mm.PBC else 'without'} PBC, a={mm.a}, angle={inputter.angle/math.pi*180:.0f}°", file=file)
    print("# H, T, MCS, m", file=file)

    for H in H_array:
        inputter.magnitude = H                          # set magnetic field H
        for T in T_array:
            mm.T = T                                    # set temperature T
            mm.initialize_m(pattern="random")
            MCsteps0 = mm.MCsteps

            print(f"Running H = {H}, T = {T}")

            MCstepsDone = 0
            while MCstepsDone < MaxMCsteps:
                inputter.input_single(mm)               # update mm
                m = mm.m_avg                            # get order parameter avg_m
                MCstepsDone = mm.MCsteps - MCsteps0
                print(H, T, MCstepsDone, m, file=file)

    file.close()


def plot_HTorder(filename: str, ncols=1):

    if not filename.endswith(".txt"):
        filename += ".txt"

    file = open(filename, "r")
    header = file.readline()[2:]  # skip '# '
    output = np.loadtxt(filename)
    file.close()

    # Unpack data
    print("Unpacking data...")
    data_dict = {}
    for H, T, MCS, m in output:
        if H not in data_dict:
            data_dict[H] = {T: ([MCS], [m])}
        else:
            T_dict = data_dict[H]
            if T not in T_dict:
                T_dict[T] = ([MCS], [m])
            else:
                T_dict[T][0].append(MCS)
                T_dict[T][1].append(m)

    # PLot data
    print("Plotting data")
    nrows = math.ceil(len(data_dict)/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    flat_axs = axs.flatten()
    suptitle = "Order parameter m evolution for various magnetic field strengths H at different temperatures T\n" + header
    fig.suptitle(suptitle)

    cmap = cm.get_cmap('inferno')
    rgba = cmap(0.5)

    for i, (H, T_dict) in enumerate(data_dict.items()):
        ax = flat_axs[i]
        T_min, T_max = min(T_dict.keys()), max(T_dict.keys())

        for T, (MCS_list, m_list) in T_dict.items():
            color = cmap((T - T_min)/(T_max-T_min))
            ax.plot(MCS_list, m_list, label=f"T = {T}K", color=color)

        ax.set_title(f"H = {H:.2e}T")

    fig.text(0.5, 0.04, "Monte Carlo Steps (time)", ha='center')  # Common xlabel
    fig.text(0.04, 0.5, "Average magnetization m", va='center', rotation='vertical')  # Common ylabel

    handles, labels = flat_axs[0].get_legend_handles_labels()  # all the same legend anyway
    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.0, 0.5))

    plt.show()
    return fig, axs

#-----------------------------------------------------------------------------------------------------------------------

a = 1e-06
n = 25
PBC=False

angle = math.pi/4  # angle of magnetic field
MCS = 0.1  # Monte Carlo steps per input_single

H_array = np.arange(1.25, 4.25, 0.25) * 1e-5
T_array = np.arange(250, 750, 75)
MaxMCsteps = 5

filename = "even even more HTm"

mm = hotspin.ASI.IP_PinwheelDiamond(a, n, PBC=PBC, pattern="random", energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()))
datastream = hotspin.io.ConstantDatastream()  # constantly returns 1
inputter = hotspin.io.FieldInputter(datastream, angle=angle, n=MCS, frequency=0)

# sweep_HTorder(mm, inputter, filename, H_array, T_array, MaxMCsteps)
plot_HTorder(filename, ncols=4)