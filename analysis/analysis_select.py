import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps, patches
from scipy.spatial import distance

from context import hotspice
if hotspice.config.USE_GPU:
    import cupy as xp
else:
    import numpy as xp


def calculate_any_neighbors(pos, shape, center: int = 0):
    """ @param pos [xp.array(2xN)]: the array of indices (row 0: y, row 1: x)
        @param shape [tuple(2)]: the maximum indices (+1) in y- and x- direction
        @param center [int]: if 0, then the full neighbor array is returned. If nonzero, only the 
            middle region (of at most `center` cells away from the middle) is returned.
        @return [xp.array2D]: an array representing where the other samples are relative to every
            other sample. To this end, each sample is placed in the middle of the array, and all
            positions where another sample exists are incremented by 1. As such, the array is
            point-symmetric about its middle.
    """
    # Note that this function can entirely be replaced by
    # signal.convolve2d(arr, arr, mode='full')
    # but this is very slow for large, sparse arrays like we are dealing with here.
    final_array = xp.zeros((2*shape[0]-1)*(2*shape[1]-1))
    pairwise_distances = (pos.T[:,None,:] - pos.T).reshape(-1,2).T # The real distances as coordinates
    # But we need to bin them, so we have to increase everything so there are no nonzero elements, and flatten:
    pairwise_distances_flat = (pairwise_distances[0] + shape[0] - 1)*(2*shape[1]-1) + (pairwise_distances[1] + shape[1] - 1)
    pairwise_distances_flat_binned = xp.bincount(pairwise_distances_flat)
    final_array[:pairwise_distances_flat_binned.size] = pairwise_distances_flat_binned
    final_array = final_array.reshape((2*shape[0]-1, 2*shape[1]-1))
    if center == 0:
        return final_array # No need to center it at all
    else:
        pad_x, pad_y = -(shape[1] - center - 1), -(shape[0] - center - 1) # How much padding(>0)/cropping(<0) needed to center the array
        final_array = final_array[:,-pad_x:-pad_x + 2*center+1] if pad_x < 0 else xp.pad(final_array, ((0,0), (pad_x,pad_x))) # x cropping/padding
        final_array = final_array[-pad_y:-pad_y + 2*center+1,:] if pad_y < 0 else xp.pad(final_array, ((pad_y,pad_y), (0,0))) # y cropping/padding
        return final_array


def analysis_select_distribution(n:int=10000, L:int=400, Lx:int=None, Ly:int=None, r:float=16, plot:bool=True, save:bool=False, PBC:bool=True, ASI_type:type[hotspice.Magnets]=None):
    """ In this analysis, the multiple-magnet-selection algorithm of `hotspice.Magnets.select()` is analyzed.
        The spatial distribution is calculated by performing `n` runs of the `select()` method.
        Also the probability distribution of the distance between two samples is calculated,
        as well as the probablity distrbution of their relative positions.
            (note that this becomes expensive to calculate for large `L`)
        @param n [int] (10000): the number of times the `select()` method is executed.
        @param Lx, Ly [int] (400): the size of the simulation in x- and y-direction. Can also specify `L` for square domain.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    """
    if Ly is None: Ly = L
    if Lx is None: Lx = L
    if ASI_type is None: ASI_type = hotspice.ASI.OOP_Square
    mm = ASI_type(1, Lx, ny=Ly, PBC=PBC)
    mm.params.UPDATE_SCHEME = hotspice.Scheme.METROPOLIS
    mm.params.MULTISAMPLING_SCHEME = 'grid'
    INTEGER_BINS = False # If true, the bins are pure integers, otherwise they can be finer than this.
    ONLY_SMALLEST_DISTANCE = True # If true, only the distances to nearest neighbors are counted.
    scale: int = 3 if ONLY_SMALLEST_DISTANCE else 4 # The distances are stored up to scale*r, beyond that we dont keep in memory

    n_bins = int(r*scale+1) if INTEGER_BINS else 99
    distances_binned = xp.zeros(n_bins)
    bin_width = r*scale/(n_bins-1)
    distance_bins = np.linspace(0, r*scale-bin_width, n_bins)
    max_dist_bin = r*scale
    
    t = time.perf_counter()
    total = 0 # Number of samples
    min_dist = xp.inf # Minimal distance between 2 samples in 1 mm.select() call
    field = xp.zeros_like(mm.m) # The number of times each given cell was chosen
    field_local = xp.zeros((2*r*scale+1, 2*r*scale+1)) # Basically distances_binned, but in 2D (distribution of neighbors around a choice)
    spectrum = xp.zeros_like(field)
    for _ in range(n):
        # from poisson_disc import Bridson_sampling
        # pos = xp.asarray(Bridson_sampling(dims=np.array([Ly, Lx]), radius=r, k=5).T, dtype=int) # POISSON DISC SAMPLING BRIDSON2007
        pos = mm.select(r=r)
        total += pos.shape[1]
        choices = xp.zeros_like(field)
        choices[pos[0], pos[1]] += 1
        field += choices
        if mm.PBC:
            _, n_pos = pos.shape # The following approach is quite suboptimal, but it works :)
            all_pos = xp.zeros((2, n_pos*4), dtype=int)
            all_pos[:,:n_pos] = pos
            all_pos[0,n_pos:n_pos*2] = all_pos[0,:n_pos] + mm.ny
            all_pos[1,n_pos:n_pos*2] = all_pos[1,:n_pos]
            all_pos[0,n_pos*2:] = all_pos[0,:n_pos*2]
            all_pos[1,n_pos*2:] = all_pos[1,:n_pos*2] + mm.nx
        else:
            all_pos = pos
        if all_pos.shape[1] > 1: # if there is more than 1 sample
            if ONLY_SMALLEST_DISTANCE:
                dist_matrix = xp.asarray(distance.cdist(hotspice.utils.asnumpy(all_pos.T), hotspice.utils.asnumpy(all_pos.T)))
                dist_matrix[dist_matrix==0] = np.inf
                distances = xp.min(dist_matrix, axis=1)
            else:
                distances = xp.asarray(distance.pdist(hotspice.utils.asnumpy(all_pos.T)))
            # if min_dist > (m := xp.min(distances)) and m < r:
            #     indices = xp.where(distances == m)[0]
            #     print(m, pos[:,indices])
            min_dist = min(min_dist, xp.min(distances))
            near_distances = distances[distances < max_dist_bin]
            if near_distances.size != 0:
                bin_counts = xp.bincount(xp.clip(xp.floor(near_distances/r*(n_bins/scale)), 0, n_bins-1).astype(int))
                distances_binned[:bin_counts.size] += bin_counts
            field_local += calculate_any_neighbors(all_pos, (mm.ny*(1+mm.PBC), mm.nx*(1+mm.PBC)), center=r*scale)
            spectrum += xp.log(xp.abs(xp.fft.fftshift(xp.fft.fft2(choices)))) # Not sure if this should be done always or only if more than 1 sample exists
        
    field_local[r*scale, r*scale] = 0 # set center value to zero
    t = time.perf_counter() - t
    
    print(f"Time required for {n} runs of this analysis: {t:.3f}s.")
    print(f"--- ANALYSIS RESULTS ---")
    print(f"Total number of samples: {total}")
    print(f"Empirical minimal distance between two samples in a single selection: {min_dist:.2f} (r={r})")
    
    cmap = colormaps['viridis'].copy()
    cmap.set_under(color='black')
    hotspice.plottools.init_style()
    fig = plt.figure(figsize=(8, 7))

    # PLOT 1: HISTOGRAM OF (NEAREST) NEIGHBORS
    ax1 = fig.add_subplot(2, 2, 1)
    if ONLY_SMALLEST_DISTANCE:
        color_left = 'C1'
        ax1.set_xlabel("Distance to nearest neighbor (binned)")
        data1_left = hotspice.utils.asnumpy(distances_binned/xp.sum(distances_binned))/(bin_width/r)
        ax1.fill_between(distance_bins, data1_left, step='post', edgecolor=color_left, facecolor=color_left, alpha=0.7)
        ax1.set_ylabel("Probability density [$r^{-1}$]", color=color_left)
        ax1.tick_params(axis='y', labelcolor=color_left)

        ax1_right = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color_right = 'C0'
        ax1_right.set_ylabel("Cumulative probability", color=color_right)
        data1_right = hotspice.utils.asnumpy(xp.cumsum(distances_binned/xp.sum(distances_binned)))
        ax1_right.bar(distance_bins, data1_right, align='edge', width=bin_width, color=color_right)
        ax1_right.tick_params(axis='y', labelcolor=color_right)
        ax1.set_zorder(ax1_right.get_zorder() + 1)
        ax1.patch.set_visible(False)

        ax1.set_ylim([0, ax1.get_ylim()[1]])
        ax1_right.set_ylim([0, ax1_right.get_ylim()[1]]) # might want to set this to [0, 1]
    else:
        ax1.bar(distance_bins, hotspice.utils.asnumpy(distances_binned), align='edge', width=bin_width)
        ax1.set_xlabel("Distance to any other sample (binned)")
        ax1.set_title("Inter-sample distances")
        ax1.set_ylabel("# occurences")
    ax1.set_xlim([0, max_dist_bin])
    ax1.set_xticks([r*n for n in range(scale+1)])
    ax1.set_xticklabels(["0", "r"] + [f"{n}r" for n in range(2, scale+1)])
    ax1.axvline(r, color='black', linestyle=':', linewidth=1, label=None)

    # PLOT 2: PROBABILITY DENSITY OF NEIGHBORS AROUND ANY SAMPLE
    ax2 = fig.add_subplot(2, 2, 2)
    data2 = hotspice.utils.asnumpy(field_local)/total
    im2 = ax2.imshow(data2, vmin=1e-10, vmax=max(2e-10, np.max(data2)), extent=[-.5-r*scale, .5+r*scale, -.5-r*scale, .5+r*scale], interpolation_stage='rgba', interpolation='nearest', cmap=cmap)
    ax2.set_title("Prob. dens. of neighbors\naround any sample")
    ax2.add_patch(plt.Circle((0, 0), 0.707, linewidth=0.5, fill=False, color='white'))
    ax2.add_patch(patches.Ellipse((0, 0), 2*r/xp.min(mm.dx), 2*r/xp.min(mm.dy), linewidth=1, fill=False, color='white', linestyle=':'))
    plt.colorbar(im2, extend='min')

    # PLOT 3: PROBABILITY OF CHOOSING EACH CELL
    ax3 = fig.add_subplot(2, 2, 3) # TODO: add x and y histograms to the sides of this
    data3 = hotspice.utils.asnumpy(field)
    im3 = ax3.imshow(data3, vmin=1e-10, origin='lower', interpolation_stage='rgba', interpolation='none', cmap=cmap)
    ax3.set_title("# choices for each cell")
    plt.colorbar(im3, extend='min')

    # PLOT 4: PERIODOGRAM
    ax4 = fig.add_subplot(2, 2, 4)
    freq = hotspice.utils.asnumpy(xp.fft.fftshift(xp.fft.fftfreq(mm.nx, d=1))) # use fftshift to get ascending frequency order
    data4 = hotspice.utils.asnumpy(spectrum)/total
    im4 = ax4.imshow(data4, extent=[-.5+freq[0], .5+freq[-1], -.5+freq[0], .5+freq[-1]], interpolation_stage='rgba', interpolation='none', cmap='gray')
    ax4.set_title("Periodogram")
    plt.colorbar(im4)
    
    # TODO: show histogram of how many samples are generated for each call of select()
    # TODO: show one representative (or the worst i.e. least samples maybe? or lowest distance?) call of select() spatially for visual inspection (might not be necessary since we have the other plots already)
    plt.suptitle(f"{Lx}x{Ly} grid, r={r} cells: {n} runs ({total} samples)\nPBC {'en' if mm.PBC else 'dis'}abled")
    plt.gcf().tight_layout()
    if save:
        save_path = f"results/analysis_select_distribution/{type(mm).__name__}_{mm.params.MULTISAMPLING_SCHEME}_{Lx}x{Ly}_r={r}{'_PBC' if mm.PBC else ''}"
        hotspice.plottools.save_plot(save_path, ext='.pdf')
    if plot:
        plt.show()


def analysis_select_speed(n: int=10000, L:int=400, r=16, PBC:bool=True, params:hotspice.SimParams=None):
    """ Tests the speed of selecting magnets without any other analysis-related calculations in between.
        @param n [int] (10000): the number of times the `select()` method is executed.
        @param L [int] (400): the size of the simulation.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    """
    mm = hotspice.ASI.OOP_Square(1, L, PBC=PBC, params=params)
    samples = 0
    t = time.perf_counter()
    for _ in range(n):
        samples += mm.select(r=r).shape[1]
    t = time.perf_counter() - t
    print(f"Time required for {n} runs of hotspice.Magnets.select() on {L}x{L} grid: {t:.3f}s ({samples} samples).")

# TODO: write function to analyze sampling performance (analysis_select_speed/num_samples) as a function of parameters between different samplers
# e.g. occupation sparsity, nx & ny for constant r or vice versa (or maybe 2D imshow changing nx=ny and r?)

if __name__ == "__main__":
    save = False
    PBC = True
    # analysis_select_speed(L=400, n=400, PBC=PBC)
    # analysis_select_distribution(Lx=296, Ly=200, n=10000, r=16, ASI_type=hotspice.ASI.OOP_Square, save=save, PBC=PBC)
    analysis_select_distribution(Lx=315, Ly=296, n=10000, r=16, ASI_type=hotspice.ASI.IP_Pinwheel, save=save, PBC=PBC)
