import os
import time
import warnings

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.spatial.distance import pdist, cdist

from context import hotspin


def calculate_any_neighbors(pos, shape, center:int=0):
    ''' @param pos [cp.array(2xN)]: the array of indices (row 0: y, row 1: x)
        @param shape [tuple(2)]: the maximum indices (+1) in y- and x- direction
        @param center [int]: if 0, then the full neighbor array is returned. If nonzero, only the 
            middle region (of at most <center> cells away from the middle) is returned.
        @return [cp.array2D]: an array representing where the other samples are relative to every
            other sample. To this end, each sample is placed in the middle of the array, and all
            positions where another sample exists are incremented by 1. As such, the array is
            point-symmetric about its middle.
    '''
    # Note that this function can entirely be replaced by
    # signal.convolve2d(arr, arr, mode='full')
    # but this is very slow for large, sparse arrays like we are dealing with here.
    final_array = cp.zeros((2*shape[0]-1)*(2*shape[1]-1))
    pairwise_distances = (pos.T[:,None,:] - pos.T).reshape(-1,2).T # The real distances as coordinates
    # But we need to bin them, so we have to increase everything so there are no nonzero elements, and flatten:
    pairwise_distances_flat = (pairwise_distances[0] + shape[0] - 1)*(2*shape[0]-1) + (pairwise_distances[1] + shape[1] - 1)
    pairwise_distances_flat_binned = cp.bincount(pairwise_distances_flat)
    final_array[:pairwise_distances_flat_binned.size] = pairwise_distances_flat_binned
    final_array = final_array.reshape((2*shape[0]-1, 2*shape[1]-1))
    return final_array if center == 0 else final_array[shape[0]-1-center:shape[0]+center, shape[1]-1-center:shape[1]+center]


def analysis_select_distribution(n:int=10000, L:int=400, r=16, show_plot:bool=True, save:bool=False, PBC:bool=True):
    ''' In this analysis, the multiple-magnet-selection algorithm of hotspin.Magnets.select() is analyzed.
        The spatial distribution is calculated by performing <n> runs of the select() method.
        Also the probability distribution of the distance between two samples is calculated,
        as well as the probablity distrbution of their relative positions.
            (note that this becomes expensive to calculate for large L)
        @param n [int] (10000): the number of times the select() method is executed.
        @param L [int] (400): the size of the simulation.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    '''
    if L < r*3: warnings.warn(f"Simulation of size {L}x{L} might be too small for r={r} cells!", stacklevel=2)
    mm = hotspin.ASI.FullASI(L, 1, PBC=PBC)
    INTEGER_BINS = False # If true, the bins are pure integers, otherwise they can be finer than this.
    ONLY_SMALLEST_DISTANCE = True # If true, only the distances to nearest neighbors are counted.
    scale: int = 3 if ONLY_SMALLEST_DISTANCE else 4 # The distances are stored up to scale*r, beyond that we dont keep in memory

    distances_binned = cp.zeros(n_bins := int(r*scale+1)) if INTEGER_BINS else cp.zeros(n_bins := 100)
    bin_width = r*scale/(n_bins-1)
    distance_bins = cp.linspace(0, r*scale-bin_width, n_bins)
    max_dist_bin = r*scale
    get_bin = lambda x: cp.clip(cp.floor(x/r*(n_bins/scale)), 0, n_bins-1).astype(int)
    
    t = time.perf_counter()
    total = 0 # Number of samples
    min_dist = cp.inf # Minimal distance between 2 samples in 1 mm.select() call
    field = cp.zeros_like(mm.m) # The number of times each given cell was chosen
    field_local = cp.zeros((2*r*scale+1, 2*r*scale+1)) # Basically distances_binned, but in 2D (distribution of neighbors around a choice)
    spectrum = cp.zeros_like(field)
    for _ in range(n):
        # from poisson_disc import Bridson_sampling
        # pos = cp.asarray(Bridson_sampling(dims=np.array([L, L]), radius=r, k=5).T, dtype=int) # POISSON DISC SAMPLING BRIDSON2007
        pos = mm.select(r) # MODIFY THIS LINE TO SELECT A SPECIFIC SELECTION ALGORITHM
        total += pos.shape[1]
        choices = cp.zeros_like(field)
        choices[pos[0], pos[1]] += 1
        field += choices
        if pos.shape[1] > 1: # if there is more than 1 sample
            if PBC:
                _, n_pos = pos.shape # The following approach is quite suboptimal, but it works :)
                all_pos = cp.zeros((2, n_pos*4), dtype=int)
                all_pos[:,:n_pos] = pos
                all_pos[0,n_pos:n_pos*2] = all_pos[0,:n_pos] + mm.nx
                all_pos[1,n_pos:n_pos*2] = all_pos[1,:n_pos]
                all_pos[0,n_pos*2:] = all_pos[0,:n_pos*2]
                all_pos[1,n_pos*2:] = all_pos[1,:n_pos*2] + mm.ny
            else:
                all_pos = pos
            if ONLY_SMALLEST_DISTANCE:
                dist_matrix = cp.asarray(cdist(all_pos.T.get(), all_pos.T.get()))
                dist_matrix[dist_matrix==0] = np.inf
                distances = cp.min(dist_matrix, axis=1)
            else:
                distances = cp.asarray(pdist(all_pos.T.get()))
            min_dist = min(min_dist, cp.min(distances))
            near_distances = distances[distances < max_dist_bin]
            if near_distances.size != 0:
                bin_counts = cp.bincount(get_bin(near_distances))
                distances_binned[:bin_counts.size] += bin_counts
            field_local += calculate_any_neighbors(all_pos, (mm.ny*(1+PBC), mm.nx*(1+PBC)), center=r*scale)
            spectrum += cp.log(cp.abs(cp.fft.fftshift(cp.fft.fft2(choices)))) # Not sure if this should be done always or only if more than 1 sample exists
        
    field_local[r*scale, r*scale] = 0 # set center value to zero
    t = time.perf_counter() - t
    
    print(f'Time required for {n} runs of this analysis: {t:.3f}s.')
    print(f'--- TEST RESULTS ---')
    print(f'Total number of samples: {total}')
    print(f'Empirical minimal distance between two samples in a single selection: {min_dist:.2f} (r={r})')
    
    cmap = cm.get_cmap('viridis').copy()
    cmap.set_under(color='black')
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(2, 2, 1)
    if ONLY_SMALLEST_DISTANCE:
        ax1.bar(distance_bins.get(), cp.cumsum(distances_binned/cp.sum(distances_binned)).get(), align='edge', width=bin_width)
        ax1.bar(distance_bins.get(), (distances_binned/cp.sum(distances_binned)).get()/bin_width, align='edge', width=bin_width)
        ax1.set_xlabel('Distance to nearest neighbor (binned)')
        ax1.set_title('Nearest-neighbor distances')
        ax1.legend(['Cum. prob.', 'Prob. dens.'])
    else:
        ax1.bar(distance_bins.get(), distances_binned.get(), align='edge', width=bin_width)
        ax1.set_xlabel('Distance to any other sample (binned)')
        ax1.set_title('Inter-sample distances')
        ax1.set_ylabel('# occurences')
    ax1.set_xlim([0, max_dist_bin])
    ax1.set_xticks([r*n for n in range(scale+1)])
    ax1.set_xticklabels(['0', 'r'] + [f'{n}r' for n in range(2, scale+1)])
    ax1.axvline(r, color='black', linestyle=':', linewidth=1, label=None)
    ax2 = fig.add_subplot(2, 2, 3)
    im2 = ax2.imshow(field.get(), vmin=1e-10, origin='lower', interpolation_stage='rgba', interpolation='none', cmap=cmap)
    ax2.set_title(f"# choices for each cell")
    plt.colorbar(im2, extend='min')
    ax3 = fig.add_subplot(2, 2, 2)
    im3 = ax3.imshow(field_local.get()/total, vmin=1e-10, extent=[-.5-r*scale, .5+r*scale, -.5-r*scale, .5+r*scale], interpolation_stage='rgba', interpolation='nearest', cmap=cmap)
    ax3.set_title(f'Prob. dens. of neighbors\naround any sample')
    ax3.add_patch(plt.Circle((0, 0), 0.707, linewidth=0.5, fill=False, color='white'))
    ax3.add_patch(plt.Circle((0, 0), r, linewidth=1, fill=False, color='white', linestyle=':'))
    plt.colorbar(im3, extend='min')
    ax4 = fig.add_subplot(2, 2, 4)
    freq = cp.fft.fftshift(cp.fft.fftfreq(mm.nx, d=1)).get() # use fftshift to get ascending frequency order
    im4 = ax4.imshow(spectrum.get()/total, extent=[-.5+freq[0], .5+freq[-1], -.5+freq[0], .5+freq[-1]], interpolation_stage='rgba', interpolation='none', cmap='gray')
    ax4.set_title(f'Periodogram')
    plt.colorbar(im4)
    plt.suptitle(f'{L}x{L} grid, r={r} cells: {n} runs ({total} samples)\nPBC {"en" if PBC else "dis"}abled')
    plt.gcf().tight_layout()
    if save:
        save_path = f"results/analysis_select_distribution/{type(mm).__name__}_{L}x{L}_r={r}{'_PBC' if PBC else ''}.pdf"
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname): os.makedirs(dirname)
        try:
            plt.savefig(save_path)
        except PermissionError:
            warnings.warn(f'Could not save to {save_path}, probably because the file is opened somewhere else.', stacklevel=2)
    if show_plot:
        plt.show()


def analysis_select_speed(n: int=10000, L:int=400, r=16):
    ''' Tests the speed of selecting magnets without any other analysis-related calculations in between.
        @param n [int] (10000): the number of times the select() method is executed.
        @param L [int] (400): the size of the simulation.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    '''
    mm = hotspin.ASI.FullASI(L, 1)
    t = time.perf_counter()
    for _ in range(n):
        mm.select(r)
    t = time.perf_counter() - t
    print(f'Time required for {n} runs of hotspin.Magnets.select(): {t:.3f}s.')


if __name__ == "__main__":
    # analysis_select_speed(L=400)
    analysis_select_distribution(L=400, n=5000, save=True, PBC=True)