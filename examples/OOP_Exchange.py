r""" This file tests the correspondence between theory and simulation for a
    two-dimensional square Ising model, with exchange interactions only,
    by observing striped phases as a function of the relative strength $\delta$
    between the exchange and dipolar interaction at low temperature.
"""
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.special import ellipk

import os
os.environ["HOTSPICE_USE_GPU"] = "True"
import hotspice
import _example_plot_utils as epu


def run(T_range = np.linspace(0.9, 1.1, 21), N: int = 100, size: int = 800, scheme="Metropolis", outdir_name=None):
    """ Runs the simulation and saves it to a timestamped folder.
        `T_range` specifies the examined temperature range in multiples of T_c.
        At each step, `N` Monte Carlo steps per site are performed.
    """
    ## Sanitize input
    N = int(N)
    
    a = 1 # Choose large spacing to get many simultaneous Metropolis switches, exchange energy does not depend on `a` anyway
    T_c = 1 # T_c = 2*J/hotspice.kB/np.log(1+np.sqrt(2)), but we turn the roles around:
    J = T_c*hotspice.kB*np.log(1 + np.sqrt(2))/2
    
    N_saved_iters = int(np.ceil(N/2)) # Half of N iters gets saved
    observable_shape = (T_range.size, N_saved_iters)
    m_avg = np.empty(observable_shape)
    NNcorr_avg = np.empty(observable_shape)
    
    ## Simulate
    mm = hotspice.ASI.OOP_Square(a, size, energies=[hotspice.ExchangeEnergy(J=J)], pattern='uniform', PBC=True, T=1, params=hotspice.SimParams(UPDATE_SCHEME=scheme))
    for i, T in enumerate(T_range):
        print(f"[{i+1}/{T_range.size}] T = {T:.2f}*T_c...")
        mm.T = T*T_c
        mm.progress(t_max=np.inf, MCsteps_max=N/2)
        for j in range(N_saved_iters):
            mm.progress(t_max=np.inf, MCsteps_max=1)
            m_avg[i,j] = mm.m_avg
            NNcorr_avg[i,j] = mm.correlation_NN()
    
    ## Save
    real_outdir = hotspice.utils.save_results(parameters={"size": size, "N": N, "a": a, "T_c": T_c, "J": J, "PBC": mm.PBC},
                                data={"T_range": T_range, "m_avg": m_avg, "NNcorr_avg": NNcorr_avg},
                                outdir=outdir_name)
    print(f'Saved data to {real_outdir}')
    plot(real_outdir)


def plot(data_dirs=None, plot_args: list[dict] = None):
    """ (Re)plots the figures in the `data_dir` folder. If `data_dir` is not specified, the newest directory is used.
            If data_dir is a list of filepaths, then the data in those folders gets combined into a single figure.
            The params and data of the first item in the list are used for reference whenever ambiguity is possible.
        `plot_args` is a list of the same length as data_dir, containing dictionaries of plotting arguments.
            Valid plot_args keys: `label`, `marker`, `markersize`, `open_symbols`
            Example plot_args: [{"label": "Néel", "marker": "o", "open_symbols": True}]
    """
    ## Load data
    if data_dirs is None: data_dirs = epu.get_last_outdir()
    if isinstance(data_dirs, (str, Path)): data_dirs = [data_dirs] # From now on we assume data_dirs is a list of strings.
    l_params, l_data = list(map(list, zip(*[hotspice.utils.load_results(data_dir) for data_dir in data_dirs])))
    if plot_args is None: plot_args = [{} for _ in data_dirs]
    elif isinstance(plot_args, dict): plot_args = [plot_args]*len(data_dirs)
    
    # Theory
    params, data = l_params[0], l_data[0]
    J = params.get('J', params.get('T_c')*hotspice.kB*np.log(1 + np.sqrt(2))/2)
    T_range = data.get('T_range')
    T_lim = [min(T_range), max(T_range)]
    T_theory = np.linspace(T_lim[0]-.005, T_lim[1]+.005, 1000)
    k = np.sinh(2*J/(hotspice.kB*T_theory))**-2 # Some symbol defined in "Exactly solved models in statistical mechanics" p.120 (7.12)
    with np.errstate(invalid='ignore'):
        m_theory = (1 - k**2)**(1/8)
        m_theory[np.isnan(m_theory)] = 0 # Above T_c
        NNcorr_theory = np.where(k < 1,
                                np.sqrt(1+k)*((1-k)/np.pi*ellipk(k) + .5),
                                np.sqrt(1+k)*((1-k)/np.pi/k*ellipk(1/k) + .5))

    ## Plot
    epu.init_style()
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(epu.page_width/2, 2.5))
    
    ax1: plt.Axes = axes[0,0]
    ax1.set_xlabel("Temperature $T/T_c$")
    ax1.set_ylabel("Magnetization\n" + r"$\langle M \rangle /M_0$")
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_xlim(*T_lim)
    ax1.set_ylim([-0.01, 1])
    epu.label_ax(ax1, 0, offset=(-0.55, 0))
    ax2: plt.Axes = axes[0,1]
    ax2.set_xlabel("Temperature $T/T_c$")
    ax2.set_ylabel("NN correlation\n" + r"$\langle S_i S_{i+1} \rangle$")
    ax2.set_yticks([0.5, 0.7, 0.9])
    ax2.set_xlim(*T_lim)
    ax2.set_ylim([0.5, 0.9])
    epu.label_ax(ax2, 1, offset=(-0.55, 0))
    for i, (params, data) in enumerate(zip(l_params, l_data)):
        label = plot_args[i].get('label', r"Hotspice")
        fmt = plot_args[i].get('marker', epu.marker_cycle[i])
        ms = plot_args[i].get('markersize', 5)
        open_symbols = plot_args[i].get('open_symbols', False)
        ax1.errorbar(T_range, np.abs(np.mean(data['m_avg'], axis=1)), yerr=np.std(data['m_avg'], axis=1), fmt=fmt, markersize=ms, markerfacecolor='none' if open_symbols else None, label=label)
        ax2.errorbar(T_range, np.mean(data['NNcorr_avg'], axis=1), yerr=np.std(data['NNcorr_avg'], axis=1), fmt=fmt, markersize=ms, markerfacecolor='none' if open_symbols else None, label=label)
    ax1.plot(T_theory, m_theory, color='black', label=r"Theory", zorder=1000)
    ax2.plot(T_theory, NNcorr_theory, color='black', label=r"Theory", zorder=1000)
    fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=1+len(l_data), columnspacing=1.5, handletextpad=0.5)
    fig.subplots_adjust(top=0.75, bottom=0.2, left=0.17, right=0.95, wspace=0.8)
    if len(data_dirs) > 1:
        outdir = os.path.commonpath(data_dirs) + f"/combined_{hotspice.utils.timestamp()}"
    else:
        outdir = data_dirs[0]
    real_outdir = hotspice.utils.save_results(figures={'OOP_Exchange': fig}, outdir=outdir, copy_script=False)
    print(f"Saved plot to {real_outdir}")


if __name__ == "__main__":
    """ EXPECTED RUNTIME: <1min per 10 steps of N. There are 1710 steps of N in total, so a few hours for the paper figure. """
    # Run for a range of Monte Carlo steps per site (N)
    scheme = hotspice.Scheme.METROPOLIS
    size = 800 # With GPU, there is no reason to take a value smaller than 800 here
    Ns = [10, 100, 1600] #! This can take a while with N=1600.
    for N in Ns:
        run(N=N, size=size, scheme=scheme, outdir_name=f"N_sweep/size={size}/N={N:.0f}")
    
    # Combine the various runs of N into a single plot (Hotspice paper Fig. 3a-b)
    outdir = os.path.splitext(hotspice.utils.get_caller_script())[0] + '.out' # Where things get saved by default
    plot_args = [{'label': f"N={N:.0f}"} for N in Ns]
    plot([f"{outdir}/N_sweep/size={size}/N={N:.0f}" for N in Ns], plot_args=plot_args)
