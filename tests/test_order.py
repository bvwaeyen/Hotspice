# A program to roughly determine the order parameter m_avg in function of temperature T of multiple geometries.
# Created 20/10/2022

if __name__ == "__main__": print("Importing...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
import time as time

import hotspice
from hotspice.ASI import *
if __name__ == "__main__": print("Everything imported")

def get_mag_stats(mm: hotspice.core.Magnets, T, MaxTimeLike, nsamples: int = 1):
    """Starts from a uniform grid at temperature T, it updates for some amount of timelike and returns order parameter m_avg.
    If nsamples is more than 1, it will do this multiple times and return the statistical mean and standard deviation."""

    mm.T = T
    m_array = np.zeros(nsamples)

    for sample in range(nsamples):
        mm.initialize_m(pattern="uniform")

        if mm.params.UPDATE_SCHEME == "Glauber":
            TimeLikeStart = mm.MCsteps
            TimeLike = TimeLikeStart
            while TimeLike - TimeLikeStart < MaxTimeLike:
                mm.update()
                TimeLike = mm.MCsteps

        elif mm.params.UPDATE_SCHEME == "Néel":
            TimeLikeStart = mm.t
            TimeLike = TimeLikeStart
            while TimeLike - TimeLikeStart < MaxTimeLike:
                mm.update(t_max=(MaxTimeLike-TimeLike))
                TimeLike = mm.t

        m_array[sample] = mm.m_avg

    if nsamples==1:
        return m_array[0]

    return np.mean(m_array), np.std(m_array)

def sweep_order_temperature(mm: hotspice.core.Magnets, T_array, MaxTimeLike, nsamples: int = 1, verbose=False):
    """Sweeps over range of temperature and returns data."""
    l = len(T_array)
    data_dict = {"T": T_array, "ComputationTime[s]": [0] * l}
    if samples == 1:
        data_dict["m_avg"] = [0] * l
    else:
        data_dict["m_mean"], data_dict["m_std"] = [0] * l, [0] * l

    constants = {"Grid": f"{mm.nx}x{mm.ny}", "Geometry": type(mm).__name__, "a": mm.a, "PBC": mm.PBC,
                 "UPDATE_SCHEME": mm.params.UPDATE_SCHEME, "nsamples": nsamples, "MaxTimeLike": MaxTimeLike}

    for i, T in enumerate(T_array):
        # the actual test
        if verbose and hotspice.utils.is_significant(i, l): hotspice.utils.log(f"Starting {i+1} of {l}: {100*(i+1)/l:.2f}%")
        start_time = time.time()
        output = get_mag_stats(mm, T, MaxTimeLike, nsamples=nsamples)  # Computing for specific T
        end_time = time.time()
        data_dict["ComputationTime[s]"][i] = end_time - start_time

        if nsamples==1:
            data_dict["m_avg"][i] = output
        else:
            data_dict["m_mean"][i], data_dict["m_std"][i] = output[0], output[1]

    df = pd.DataFrame(data=data_dict)
    data = hotspice.utils.Data(df, constants)

    return data

def plot_order_T(data: hotspice.utils.Data):
    """Plots data, shows plot and returns (fig, ax)"""

    # Elaborate title construction
    cte = data.constants
    title = f"Order parameter of {cte['Grid']} {cte['Geometry']} {'with' if cte['PBC'] else 'without'} PBC, a={cte['a']}"
    if cte["UPDATE_SCHEME"] == "Glauber":
        title += f", {cte['MaxTimeLike']}  MC steps to relax"
    elif cte["UPDATE_SCHEME"] == "Néel":
        title += f", {cte['MaxTimeLike']:.4f} sec to relax"
    if cte["nsamples"] > 1:
        title += f" with {cte['nsamples']} samples per T"
    title += "."
    title = "\n".join(wrap(title, 60))  # long title can wrap around

    # Actual plotting of data
    fig, ax = plt.subplots()

    T = data.get("T")
    if cte["nsamples"] > 1:
        m_mean, m_std = data.get("m_mean"), data.get("m_std")
        ax.plot(T, m_mean, label="Mean of average magnetization")
        ax.plot(T, m_std, label="Standard deviation")
    else:
        m_avg = data.get("m_avg")
        ax.plot(T, m_avg, label="Average magnetization")

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("T[K]")

    return fig, ax

#-----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    a = 300e-9
    n = 10
    PBC=False
    MaxTimeLike = 100
    samples = 5
    update_scheme = "Glauber"
    verbose = True

    filename = "Longer relax time"  # gets extended later on
    directory = "Glauber 10x10 PinwheelDiamond no PBC"

    T_array = np.arange(3000, 15050, 50)
    # T_array[0] = 1  # Don't start at 0K

    # List of ASI to do
    mm_list = [IP_PinwheelDiamond(a, n, PBC=PBC)]
    """
    mm_list = [IP_Ising(a, n, PBC=PBC), IP_Square (a, int(np.sqrt(2)*n), PBC=PBC), IP_SquareDiamond(a, n, PBC=PBC),
               IP_Pinwheel(a, int(np.sqrt(2)*n), PBC=PBC), IP_PinwheelDiamond(a, n, PBC=PBC),
               IP_Kagome(a, int(np.sqrt(8/3) * n) - int(np.sqrt(8/3) * 20)%4, PBC=PBC)]
    """

    for mm in mm_list:
        mm.params.UPDATE_SCHEME = update_scheme
        filename += f"{update_scheme} {type(mm).__name__} order parameter sweep"
        print("Currently doing", filename)
        data = sweep_order_temperature(mm, T_array, MaxTimeLike=MaxTimeLike, nsamples=samples, verbose=verbose)  # produce the data
        data.save(name=filename, dir=directory)  # save the data
        fig, ax = plot_order_T(data)  # plot the data
        fig.savefig(filename+".png")  # save the plot

    plt.show()
