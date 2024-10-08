import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context import hotspice


def analysis_speed(mm: hotspice.Magnets, t_min: float = 1, n_min: int = 1, verbose: bool = False):
    """ In this analysis, the performance of Hotspice for the geometry in `mm` is measured,
        by simply calculating the number of attempted switches per second, real switches
        per second, and Monte Carlo steps per second, when calling `mm.update()` successively.
        @param mm [hotspice.Magnets]: the Magnets object with the desired size and parameters.
        @param t_min [float] (1): the minimal number of seconds during which the performance is monitored.
        @param n_min [int] (1): the minimal number of `mm.update()` calls whose performance is monitored.
    """
    hotspice.utils.free_gpu_memory()
    i, t0 = -1, time.perf_counter()
    while (i := i + 1) < n_min or time.perf_counter() - t0 < t_min: # Do this for at least `n_min` iterations and `t_min` seconds
        mm.update()
    dt = time.perf_counter() - t0
    if verbose: print(f"[{hotspice.utils.get_gpu_memory()['free']} free on GPU] Time required for {i} runs ({mm.switches} switches) of Magnets.select() on {mm.nx}x{mm.ny} grid: {dt:.3f}s.")
    return {'attempts/s': mm.attempted_switches/dt, 'switches/s': mm.switches/dt, 'MCsteps/s': mm.MCsteps/dt}


def analysis_speed_size(L_range, ASI_type: type[hotspice.Magnets] = hotspice.ASI.OOP_Square, save: bool = False, plot: bool = True, verbose: bool = False, **kwargs):
    L_range: np.ndarray = np.asarray(L_range)
    n = np.zeros_like(L_range)
    attempts_per_s = np.zeros_like(L_range, dtype='float')
    switches_per_s = np.zeros_like(L_range, dtype='float')
    MCsteps_per_s = np.zeros_like(L_range, dtype='float')
    # TODO: add initialization time (perhaps on the right axis? or 'initializations per second' which is a bit weird?)
    for i, L in enumerate(L_range):
        try:
            mm = ASI_type(1e-6, L, ny=L, **kwargs)
        except Exception:
            if verbose: print(f"Could not initialize {ASI_type} for L={L}.")
            continue
        x = analysis_speed(mm, verbose=verbose)
        n[i], attempts_per_s[i], switches_per_s[i], MCsteps_per_s[i] = mm.n, x['attempts/s'], x['switches/s'], x['MCsteps/s']
    nz = n.nonzero()
    L_range, n, attempts_per_s, switches_per_s, MCsteps_per_s = L_range[nz], n[nz], attempts_per_s[nz], switches_per_s[nz], MCsteps_per_s[nz]

    df = pd.DataFrame({"L": L_range, "n": n, 'attempts/s': attempts_per_s, 'switches/s': switches_per_s, 'MCsteps/s': MCsteps_per_s})
    metadata = {'description': r"Performance test for Hotspice, determining throughput as switches/s or a similar metric, for various simulation sizes."}
    constants = {'T': mm.T_avg, 'E_B': mm.E_B_avg, 'dx': mm.dx, 'dy': mm.dy, 'ASI_type': hotspice.utils.full_obj_name(mm), 'PBC': mm.PBC}
    data = hotspice.utils.Data(df, metadata=metadata, constants=constants)
    if save: save = data.save(dir=f"results/analysis_speed_size", name=f"{ASI_type.__name__}_size{L_range.min()}-{L_range.max()}_T{constants['T']:.0f}{'_PBC' if kwargs.get('PBC', False) else ''}")
    if plot or save: analysis_speed_size_plot(df, save=save, show=plot) # If statement not strictly necessary, just better for performance
    return data


def analysis_speed_size_plot(df: pd.DataFrame, save=False, show=True):
    hotspice.plottools.init_style()
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.plot(df['n'], df['attempts/s'], label="Samples / sec")
    ax1.plot(df['n'], df['switches/s'], label="Switches / sec")
    ax1.plot(df['n'], df['MCsteps/s'], label="MC steps / sec")
    ax1.set_xscale('log')
    ax1.set_xlim([df['n'].min(), df['n'].max()])
    ax1.set_yscale('log')
    ax1.set_xlabel("Number of spins")
    ax1.set_ylabel("Throughput [$s^{-1}$]")
    ax1.legend()

    ax2.set_xscale('log')
    ax2.set_xlim([df['L'].min(), df['L'].max()])
    ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax2.set_xticks(ticks[np.where((ticks >= ax2.get_xlim()[0]) & (ticks <= ax2.get_xlim()[1]))])
    ax2.xaxis.grid(linestyle=':')
    ax2.set_xlabel("Cells in x- and y-direction")

    plt.gcf().tight_layout()
    if save:
        if not isinstance(save, str):
            save = f"results/analysis_speed_size/ASIspeed_size{L_range.min()}-{L_range.max()}.pdf"
        hotspice.plottools.save_plot(save, ext='.pdf')
    if show: plt.show()


if __name__ == "__main__":
    L_range = np.concatenate([np.arange(1, 100, 1), np.arange(100, 400, 5), np.arange(400, 600, 10), np.arange(600, 1001, 25)]) # for GPU
    # L_range = np.concatenate([np.arange(1, 100, 1), np.arange(100, 251, 5)]) # for CPU
    # analysis_speed_size(L_range=L_range, save=True, plot=True, T=100, PBC=True, verbose=True)

    # plot = False
    # T = 100
    # analysis_speed_size(L_range=L_range, ASI_type=hotspice.ASI.OOP_Square, save=True, plot=plot, T=T, PBC=True, verbose=True)
    # analysis_speed_size(L_range=L_range, ASI_type=hotspice.ASI.IP_Ising, save=True, plot=plot, T=T, PBC=True, verbose=True)
    # analysis_speed_size(L_range=L_range, ASI_type=hotspice.ASI.IP_Pinwheel, save=True, plot=plot, T=T, PBC=True, verbose=True)
    # analysis_speed_size(L_range=L_range, ASI_type=hotspice.ASI.IP_Square, save=True, plot=plot, T=T, PBC=True, verbose=True)
    # analysis_speed_size(L_range=L_range, ASI_type=hotspice.ASI.IP_Kagome, save=True, plot=plot, T=T, PBC=True, verbose=True)
    # analysis_speed_size(L_range=L_range, ASI_type=hotspice.ASI.IP_Triangle, save=True, plot=plot, T=T, PBC=True, verbose=True)

    # openname = "results/analysis_speed_size/something.json"
    # analysis_speed_size_plot(hotspice.utils.Data.load(openname).df, save=openname)
