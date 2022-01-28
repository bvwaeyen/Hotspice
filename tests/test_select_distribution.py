import time

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, cdist

from context import hotspin


def test(n:int=10000, L:int=400, r=16):
    ''' In this test, the multiple-magnet-selection algorithm of hotspin.Magnets.select() is tested.
        The spatial distribution is calculated by performing <n> runs of the select() method.
        Also the probability distribution of the distance between two samples is calculated.
            (note that this becomes expensive to calculate for large L)
        @param n [int] (10000): the number of times the select() method is executed.
        @param L [int] (400): the size of the simulation.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    '''
    mm = hotspin.ASI.FullASI(L, 1)
    INTEGER_BINS = False # If true, the bins are pure integers, otherwise they can be finer than this.
    ONLY_SMALLEST_DISTANCE = True # If true, only the distances to nearest neighbors are counted.
    scale: int = 3 if ONLY_SMALLEST_DISTANCE else 4 # The distances are stored up to scale*r, beyond that we dont keep in memory

    distances_binned = cp.zeros(n_bins := int(r*scale+1)) if INTEGER_BINS else cp.zeros(n_bins := 100)
    bin_width = r*scale/(n_bins-1)
    distance_bins = cp.linspace(0, r*scale-bin_width, n_bins)
    max_dist_bin = r*scale
    get_bin = lambda x: cp.clip(cp.floor(x/r*(n_bins/scale)), 0, n_bins-1).astype(int)
    field = cp.zeros_like(mm.xx)
    
    t = time.perf_counter()
    total = 0
    min_dist = cp.inf
    for _ in range(n):
        pos = mm.select(r) # MODIFY THIS LINE TO SELECT A SPECIFIC SELECTION ALGORITHM
        total += pos.shape[1]
        field[pos[0], pos[1]] += 1
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
    t = time.perf_counter() - t
    
    print(f'Time required for {n} runs of this test: {t:.3f}s.')
    print(f'--- TEST RESULTS ---')
    print(f'Total number of samples: {total}')
    print(f'Minimal distance between two samples in a single selection: {min_dist} (r={r})')
    
    fig = plt.figure(figsize=(7, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    if ONLY_SMALLEST_DISTANCE:
        ax1.bar(distance_bins.get(), cp.cumsum(distances_binned).get()/total, align='edge', width=bin_width)
        ax1.bar(distance_bins.get(), distances_binned.get()/total, align='edge', width=bin_width)
        ax1.set_xlabel('Distance to any other sample (binned)')
        ax1.legend(['Cumulative prob.', 'Probability'])
    else:
        ax1.bar(distance_bins.get(), distances_binned.get(), align='edge', width=bin_width)
        ax1.set_xlabel('Distance to nearest neighbor (binned)')
        ax1.set_ylabel('# occurences')
    ax1.set_xlim([0, max_dist_bin])
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(field.get(), vmin=0, interpolation_stage='rgba', interpolation='antialiased')
    ax2.set_title(f"# choices over {n} runs")
    plt.colorbar(im2)
    plt.suptitle(f'{L}x{L} grid, r={r}, {n} runs')
    plt.gcf().tight_layout()
    plt.show()


def test_speed(n: int=10000, L:int=400, r=16):
    ''' Tests the speed of selecting magnets without any other calculations in between.
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
    # test_speed()
    test(L=400)