import os
import time
import warnings

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, cdist

from context import hotspin


def calculate_any_neighbors(pos, shape):
    ''' @param pos [cp.array(2xN)]: the array of coordinates (row 0: y, row 1: x)
        @param shape [tuple(2)]: the maximum values in y- and x- direction
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
    return final_array.reshape((2*shape[0]-1, 2*shape[1]-1))


def test(n:int=10000, L:int=400, r=16, show_plot=True, save=False):
    ''' In this test, the multiple-magnet-selection algorithm of hotspin.Magnets.select() is tested.
        The spatial distribution is calculated by performing <n> runs of the select() method.
        Also the probability distribution of the distance between two samples is calculated,
        as well as the probablity distrbution of their relative positions.
            (note that this becomes expensive to calculate for large L)
        @param n [int] (10000): the number of times the select() method is executed.
        @param L [int] (400): the size of the simulation.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    '''
    if L < r*3: warnings.warn(f"Simulation of size L={L} might be too small for r={r}!")
    mm = hotspin.ASI.FullASI(L, 1)
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
    for _ in range(n):
        pos = mm.select(r) # MODIFY THIS LINE TO SELECT A SPECIFIC SELECTION ALGORITHM
        total += pos.shape[1]
        field[pos[0], pos[1]] += 1
        field_local += calculate_any_neighbors(pos, (mm.ny, mm.nx))[mm.ny-1-r*scale:mm.ny+r*scale, mm.nx-1-r*scale:mm.nx+r*scale] # Cropped to center
        if pos.shape[1] > 1: # if there is more than 1 sample
            if ONLY_SMALLEST_DISTANCE:
                dist_matrix = cp.asarray(cdist(pos.T.get(), pos.T.get()))
                dist_matrix[dist_matrix==0] = np.inf
                distances = cp.min(dist_matrix, axis=1)
            else:
                distances = cp.asarray(pdist(pos.T.get()))
            min_dist = min(min_dist, cp.min(distances))
            bin_counts = cp.bincount(get_bin(distances[distances < max_dist_bin]))
            distances_binned[:bin_counts.size] += bin_counts
    field_local[r*scale, r*scale] = 0 # set center value to zero
    t = time.perf_counter() - t
    
    print(f'Time required for {n} runs of this test: {t:.3f}s.')
    print(f'--- TEST RESULTS ---')
    print(f'Total number of samples: {total}')
    print(f'Empirical minimal distance between two samples in a single selection: {min_dist} (r={r})')
    
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(1, 3, 1)
    if ONLY_SMALLEST_DISTANCE:
        ax1.bar(distance_bins.get(), cp.cumsum(distances_binned).get()/total, align='edge', width=bin_width)
        ax1.bar(distance_bins.get(), distances_binned.get()/total, align='edge', width=bin_width)
        ax1.set_xlabel('Distance to nearest neighbor (binned)')
        ax1.legend(['Cumulative prob.', 'Probability'])
    else:
        ax1.bar(distance_bins.get(), distances_binned.get(), align='edge', width=bin_width)
        ax1.set_xlabel('Distance to any other sample (binned)')
        ax1.set_ylabel('# occurences')
    ax1.set_xlim([0, max_dist_bin])
    ax1.set_title('Nearest-neighbor distances')
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(field.get(), vmin=0, interpolation_stage='rgba', interpolation='antialiased')
    ax2.set_title(f"# choices in entire simulation")
    plt.colorbar(im2)
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(field_local.get(), extent=[-.5-r*scale, .5+r*scale, -.5-r*scale, .5+r*scale], vmin=0, interpolation_stage='rgba', interpolation='antialiased')
    ax3.set_title(f'Distribution of neighbors around a sample')
    plt.colorbar(im3)
    plt.suptitle(f'{L}x{L} grid, r={r}, {n} runs')
    plt.gcf().tight_layout()
    if save:
        save_path = f"results/test_select_distribution/{type(mm).__name__}_{L}x{L}_r={r}.pdf"
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname): os.makedirs(dirname)
        plt.savefig(save_path)
    if show_plot:
        plt.show()


def test_speed(n: int=10000, L:int=400, r=16):
    ''' Tests the speed of selecting magnets without any other test-related calculations in between.
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
    # test_speed(L=400)
    test(L=400, save=True)