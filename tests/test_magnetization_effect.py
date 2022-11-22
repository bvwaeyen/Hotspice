# A program to determine the effect of the magnetic field strength at various temperatures at various times.
# Created 25/10/2022

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap

import pandas as pd

import hotspice


def sweep_HTorder(mm: hotspice.core.Magnets, inputter: hotspice.io.FieldInputter, H_array, T_array, MaxTimelike=0):
    """
    Sweep order parameter in time for different H and T combinations.
    @param mm:
    @param inputter: Made for inputter with ConstantDatastream. Every saved timestep is one input_single()
    @param H_array: list of H values to sweep
    @param T_array: list of T values to sweep
    @param MaxTimelike: the maximum timelike. Maximum mm.MCSteps for Glauber, mm.t for Néel.
    @return: Data
    """

    constants = {"Grid": f"{mm.nx}x{mm.ny}", "Geometry": type(mm).__name__, "a": mm.a, "PBC": mm.PBC,
                 "UPDATE_SCHEME": mm.params.UPDATE_SCHEME, "MaxTimelike": MaxTimelike, "angle": f"{inputter.angle/math.pi*180:.0f}°"}
    data_dict = {"H": [], "T": [], "TimeLike": [], "m_avg": []}

    for H in H_array:
        inputter.magnitude = H                          # set magnetic field H
        for T in T_array:
            mm.T = T                                    # set temperature T
            mm.initialize_m(pattern="random")

            print(f"Running H = {H}, T = {T}")
            TimeLikeStart = mm.MCsteps if mm.params.UPDATE_SCHEME == "Glauber" else mm.t
            TimeLike = TimeLikeStart
            while TimeLike - TimeLikeStart < MaxTimelike:
                inputter.input_single(mm)               # update mm
                m_avg = mm.m_avg                        # get order parameter avg_m
                TimeLike = mm.MCsteps if mm.params.UPDATE_SCHEME == "Glauber" else mm.t
                # save data
                data_dict["H"].append(H); data_dict["T"].append(T);
                data_dict["TimeLike"].append(TimeLike-TimeLikeStart); data_dict["m_avg"].append(m_avg)

    df = pd.DataFrame(data=data_dict)
    data = hotspice.utils.Data(df, constants)

    return data


def plot_HTorder(data: hotspice.utils.Data, ncols=1, outer="H"):
    """
    Plots m_avg for different H and T. outer can be "H" or "T". Subplots will be split in
    @param data: Data object containing information like from HTorder.
    @param ncols: number of colums for subplots
    @param outer: Can be "H" or "T". This decides the way plots are organized.
    @return: fig, ax
    """

    # Unpack data
    print("Unpacking data...")
    H_array, T_array, TimeLike_array, m_avg_array = data.get("H"), data.get("T"), data.get("TimeLike"), data.get("m_avg")
    H_unique, T_unique = np.unique(H_array), np.unique(T_array)
    data_dict = {"H": H_array, "T": T_array, "TimeLike": TimeLike_array, "m_avg": m_avg_array}
    df = pd.DataFrame(data_dict)

    # PLot data
    print("Plotting data")

    if outer == "H":
        outer_unique = H_unique
        inner_unique = T_unique
        inner = "T"
        cmap = cm.get_cmap('inferno')
    else:
        outer_unique = T_unique
        inner_unique = H_unique
        inner = "H"
        cmap = cm.get_cmap('viridis')

    nrows = math.ceil(len(outer_unique)/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    if nrows*ncols > 1:
        flat_axs = axs.flatten()
    else:
        flat_axs = [axs]
    cte = data.constants
    suptitle = f"Order parameter m {cte['UPDATE_SCHEME']} evolution for various magnetic field strengths H at different temperatures T\n" + \
               f"{cte['Grid']} {cte['Geometry']} {'with' if cte['PBC'] else 'without'} PBC, a={cte['a']}, angle={cte['angle']}"
    fig.suptitle(suptitle)

    for i, outer_value in enumerate(outer_unique):
        ax = flat_axs[i]
        ax.set_title(f"{outer} = {outer_value:.5g}")
        inner_min, inner_max = min(inner_unique), max(inner_unique)

        for inner_value in inner_unique:
            color = cmap((inner_value - inner_min) / (inner_max - inner_min))
            single_df = df.loc[(df[outer] == outer_value) & (df[inner] == inner_value)]
            ax.plot(single_df["TimeLike"], single_df["m_avg"], label=f"{inner} = {inner_value:.5g}", color=color)

    fig.text(0.5, 0.04, "Time", ha='center')  # Common xlabel
    fig.text(0.04, 0.5, "Average magnetization m", va='center', rotation='vertical')  # Common ylabel

    handles, labels = flat_axs[0].get_legend_handles_labels()  # all the same legend anyway
    fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.0, 0.5))

    plt.show()
    return fig, axs

#-----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    a = 300e-9
    n = 10
    PBC=False

    angle = math.pi/4  # angle of magnetic field
    H_array = np.arange(1, 10.5, 0.5) * 1e-4
    T_array = np.arange(0, 10001, 1000)
    T_array[0] = 1  # no 0K
    MaxMCsteps = 15
    MCS = 0.05  # Monte Carlo steps per input_single
    #MaxTime = 0.0001
    #frequency = 100/MaxTime  # frequency of input singles
    update_scheme = "Glauber"

    outer="T"
    ncols = 3

    filename = "HTm Glauber"

    mm = hotspice.ASI.IP_PinwheelDiamond(a, n, PBC=PBC, pattern="random", energies=(hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()))
    mm.params.UPDATE_SCHEME = update_scheme
    datastream = hotspice.io.ConstantDatastream()  # constantly returns 1
    inputter = hotspice.io.FieldInputter(datastream, angle=angle, n=MCS, frequency=0)

    data = sweep_HTorder(mm, inputter, H_array, T_array, MaxMCsteps)
    data.save(name=filename)
    plot_HTorder(data, ncols=ncols, outer=outer)

    """
    data = hotspice.utils.Data(pd.DataFrame({"test":["test"]}))
    data = data.load("hotspice_results/HTm Glauber _20221110135158.json")
    data.df = data.df.loc[data.df["TimeLike"] <= 12]
    plot_HTorder(data, ncols=int(np.sqrt(len(H_array))), outer="H")
    """