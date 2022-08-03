import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from context import hotspin


def analysis_TAmetrics_Nk(filename: str, k_range=10, save=True, plot=True, verbose=True):
    ''' Loads a full TaskAgnosticExperiment dataframe (of a single run) and determines the
        task-agnostic metrics NL, MC and CP as a function of the number of recorded iterations N.
        This will allow to determine an optimum value of N before performing any sort of
        parameter sweep on an HPC cluster such that we do not use too much time.
    '''
    data_in = hotspin.utils.Data.load(filename)
    df_in = data_in.df
    k_range = np.asarray(k_range).reshape(-1) # Turn into 1D range

    # Use coarser N_range for higher N, because it takes O(N) time to determine NL, MC and CP:
    N_range = np.concatenate([np.arange(1, 100, 1), np.arange(100, 400, 5), np.arange(400, 600, 10), np.arange(600, 1001, 25)])
    N_range = N_range[np.where((N_range <= (df_in.shape[0] - 2)))] # Doesn't make much sense to take N or k larger
    k_range = k_range[np.where((k_range <= (df_in.shape[0] - 2)))] # than the number of samples in the dataframe
    N_grid, k_grid = np.meshgrid(N_range, k_range)
    NL, MC, CP = np.zeros_like(N_grid)*np.nan, np.zeros_like(N_grid)*np.nan, np.zeros_like(N_grid)*np.nan
    experiment = hotspin.experiments.TaskAgnosticExperiment.dummy()
    for index, _ in np.ndenumerate(N_grid):
        k = k_grid[index]
        N = N_grid[index]
        experiment.load_dataframe(df_in.iloc[:N+1])
        try:
            NL[index] = experiment.NL(k=k)
            MC[index] = experiment.MC(k=k)
            CP[index] = experiment.CP()
            if verbose: print("Performed iteration", index, "of", (k_range.size-1, N_range.size-1))
        except:
            if verbose: print("Failed in iteration", index, "of", (k_range.size-1, N_range.size-1))
    
    # BELOW HERE IS STILL UNDER CONSTRUCTION, ABOVE HERE SHOULD BE OK
    df = pd.DataFrame({"N": N_grid.reshape(-1), "k": k_grid.reshape(-1), "NL": NL.reshape(-1), "MC": MC.reshape(-1), "CP": CP.reshape(-1)})
    metadata = data_in.metadata | {
        "description": r"Metrics calculated for different N and k, based on a long TaskAgnosticExperiment, in order to determine an optimal value of N and k before starting a parameter sweep.",
        "original_filename": filename
    }
    constants = data_in.constants
    data = hotspin.utils.Data(df, metadata=metadata, constants=constants)
    if save: save = data.save(dir=os.path.dirname(filename), name=f"{os.path.splitext(os.path.basename(filename))[0]}_metrics", timestamp=False)
    if plot or save: analysis_TAmetrics_Nk_plot(df, save=save, show=plot)
    return data

def analysis_TAmetrics_Nk_plot(df: pd.DataFrame, save=False, show=True):
    k_vals = df["k"].nunique()
    N_grid = df["N"].values.reshape(-1, k_vals)
    k_grid = df["k"].values.reshape(-1, k_vals)
    NL_grid = df["NL"].values.reshape(-1, k_vals)
    MC_grid = df["MC"].values.reshape(-1, k_vals)
    CP_grid = df["CP"].values.reshape(-1, k_vals)

    hotspin.plottools.init_fonts()
    fig = plt.figure(figsize=(5, 5))
    if k_vals > 1: # use surface plot in 3D
        ax = fig.add_subplot(111, projection='3d')
        NL_surf = ax.plot_surface(N_grid, k_grid, NL_grid, color="C0", linewidth=0, antialiased=True, label='NL')
        MC_surf = ax.plot_surface(N_grid, k_grid, MC_grid, color="C1", linewidth=0, antialiased=True, label='MC')
        CP_surf = ax.plot_surface(N_grid, k_grid, CP_grid, color="C2", linewidth=0, antialiased=True, label='CP')
        fakeNLline = Line2D([0],[0], linestyle="none", color="C0", marker='o')
        fakeMCline = Line2D([0],[0], linestyle="none", color="C1", marker='o')
        fakeCPline = Line2D([0],[0], linestyle="none", color="C2", marker='o')
        ax.set_xlabel("N")
        ax.set_ylabel("k")
        ax.set_zlabel("metrics")
        ax.legend([fakeNLline, fakeMCline, fakeCPline], ['NL', 'MC', 'CP'], numpoints = 1)
    else:
        ax = fig.add_subplot(111)
        NL_line = ax.plot(N_grid, NL_grid, color="C0", label='NL')
        MC_line = ax.plot(N_grid, MC_grid, color="C1", label='MC')
        CP_line = ax.plot(N_grid, CP_grid, color="C2", label='CP')
        ax.set_xlabel("N")
        ax.set_ylabel("metrics")
        ax.legend()
    plt.gcf().tight_layout()
    if save:
        if not isinstance(save, str):
            save = f"results/analysis_TAmetrics/NL-MC-CP_N{df['N'].min()}-{df['N'].max()}_k{df['k'].min()}-{df['k'].max()}.pdf"
        hotspin.plottools.save_plot(save, ext='.pdf')
    if show: plt.show()

if __name__ == "__main__":
    save = True
    # analysis_TAmetrics_Nk("results/TaskAgnosticExperiment/PinwheelASI_N=1000_test.json", k_range=[10, 20])
    # analysis_TAmetrics_Nk("results/TaskAgnosticExperiment/SquareASI_N=1000_testsmol.json", k_range=[10], save=save)
    analysis_TAmetrics_Nk("results/TaskAgnosticExperiment/Sweep/ComputationInASI_extmag0.015or0.017T.json", k_range=[10], save=False)
