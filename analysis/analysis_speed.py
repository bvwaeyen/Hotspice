import os
import time
import warnings

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from context import hotspin


# TODO: WIP, need to determine which metric(s) to use for analyzing performance


def analysis_speed(L:int=400, Lx:int=None, Ly:int=None, t_min:float=1, n_min:int=1, ASI_type=hotspin.ASI.FullASI, verbose:bool=True, PBC:bool=True):
    ''' In this analysis, the multiple-magnet-selection algorithm of hotspin.Magnets.select() is analyzed.
        The spatial distribution is calculated by performing <n> runs of the select() method.
        Also the probability distribution of the distance between two samples is calculated,
        as well as the probablity distrbution of their relative positions.
            (note that this becomes expensive to calculate for large L)
        @param n [int] (10000): the number of times the select() method is executed.
        @param Lx, Ly [int] (400): the size of the simulation in x- and y-direction. Can also specify <L> for square domain.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    '''
    if Ly is None: Ly = L
    if Lx is None: Lx = L
    mm = ASI_type(Lx, 1e-6, T=100, ny=Ly, PBC=PBC)
    
    energy_calculations = 0
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < t_min or i < n_min: # Do this for at least <t_min> seconds and <n_min> iterations
        switches_prev = mm.switches
        mm.update()
        i += 1
        if mm.switches - switches_prev > mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF and mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF:
            # A convolution is responsible for the recent update
            energy_calculations += L*L
        else:
            energy_calculations += L*L*(mm.switches - switches_prev)

    t1 = time.perf_counter()
    switches_per_sec = mm.switches/(t1 - t0)
    energy_calculations_per_sec = energy_calculations/(t1 - t0)

    
    if verbose: print(f'Time required for {i} runs ({mm.switches} switches) of Magnets.select() on {L}x{L} grid: {t1-t0:.3f}s ({switches_per_sec:.2e}/sec).')

    return energy_calculations_per_sec, switches_per_sec

def analysis_speed_size(L_range, save: bool = False, ASI_type=hotspin.ASI.FullASI, show_plot: bool = True, **kwargs):
    energy_calculations_per_sec = np.zeros_like(L_range)
    switches_per_sec = np.zeros_like(L_range)
    for i, L in enumerate(L_range):
        energy_calculations_per_sec[i], switches_per_sec[i] = analysis_speed(L=L, ASI_type=ASI_type, **kwargs)
    
    hotspin.plottools.init_fonts()
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(L_range**2, energy_calculations_per_sec, label='Energy calculations / sec')
    ax1.plot(L_range**2, switches_per_sec, label='Switches / sec')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of spins')
    ax1.set_ylabel('Throughput')
    ax1.legend()
    # ax1.set_xlim([0, max_dist_bin])
    # ax1.set_xticks([r*n for n in range(scale+1)])
    # ax1.set_xticklabels(['0', 'r'] + [f'{n}r' for n in range(2, scale+1)])
    plt.gcf().tight_layout()
    if save:
        save_path = f"results/analysis_speed_size/{ASI_type.__name__}_size{L_range.min()}-{L_range.max()}{'_PBC' if kwargs.get('PBC', False) else ''}.pdf"
        hotspin.plottools.save_plot(save_path)
    if show_plot:
        plt.show()



if __name__ == "__main__":
    L_range = np.concatenate([np.arange(1,100,1), np.arange(100, 400, 5), np.arange(400, 600, 10), np.arange(600, 1001, 25)])
    analysis_speed_size(L_range=L_range, save=True, PBC=True)
    mm = hotspin.ASI.FullASI(40, 1)
