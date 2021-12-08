import ctypes
import math
import matplotlib

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cupyx.scipy import signal
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import MultiCursor

from .core import Magnets

ctypes.windll.shcore.SetProcessDpiAwareness(2) # (For Windows 10/8/7) this makes the matplotlib plots smooth on high DPI screens
matplotlib.rcParams["image.interpolation"] = 'none' # 'none' works best for large images scaled down, 'nearest' for the opposite

# TODO: organize these functions better

class Average:
    # Use dtype='float, because if mask would be int then output of convolve2d is also int instead of float
    POINT = cp.array([[1]], dtype='float')
    CROSS = cp.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype='float')
    SQUARE = cp.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype='float')
    HEXAGON = cp.array([[0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 1],
                        [0, 1, 0, 1, 0]], dtype='float')
    TRIANGLE = cp.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]], dtype='float')


# Below here are some graphical functions (plot magnetization profile etc.)
def _get_averaged_extent(mm: Magnets, avg):
    ''' Returns the extent that can be used in imshow when plotting an averaged quantity. '''
    avg = _resolve_avg(mm, avg)
    mask = _get_avg_mask(mm, avg=avg)
    if mm.PBC:
        movex, movey = 0.5*mm.dx, 0.5*mm.dy
    else:
        movex, movey = mask.shape[1]/2*mm.dx, mask.shape[0]/2*mm.dy # The averaged imshow should be displaced by this much
    return [mm.x_min-mm.dx+movex,mm.x_max-movex+mm.dx,mm.y_min-mm.dy+movey,mm.y_max-movey+mm.dy]

def _resolve_avg(mm: Magnets, avg):
    # TODO: do this better
    ''' If avg is str then determine if it is valid, otherwise auto-determine which averaging method is appropriate. '''
    if isinstance(avg, str):
        assert avg in ['point', 'cross', 'square', 'hexagon', 'triangle'], "Unsupported averaging mask: %s" % avg
    else: # It is something which can be truthy or falsy
        avg = mm._get_appropriate_avg() if avg else 'point'
    return avg

def _get_avg_mask(mm: Magnets, avg=None):
    ''' Returns the raw averaging mask as a 2D array. Note that this obviously does not include
        any removal of excess zero-rows etc., that is the task of get_m_angles(mm).
        Note that this returns a CuPy array for performance reasons, since this is a 'hidden' _function anyway.
    '''
    avg = _resolve_avg(mm, avg)
    if avg == 'point':
        return Average.POINT
    elif avg == 'cross': # cross ⁛
        return Average.CROSS
    elif avg == 'square': # square □
        return Average.SQUARE
    elif avg == 'hexagon':
        return Average.HEXAGON
    elif avg == 'triangle':
        return Average.TRIANGLE

def get_m_polar(mm: Magnets, m=None, avg=True):
    '''
        Returns the magnetization angle and magnitude (can be averaged using the averaging method specified by <avg>).
        If the local average magnetization is zero, the corresponding angle is NaN.
        If there are no magnets to average around a given cell, then the angle and magnitude are both NaN.
        @param m [2D array] (mm.m): The magnetization profile that should be averaged.
        @param avg [str|bool] (True): can be any of True, False, 'point', 'cross', 'square', 'triangle', 'hexagon':
            True: automatically determines the appropriate averaging method.
            False|'point': no averaging at all, just calculates the angle of each individual spin.
            'cross': averages the spins north, east, south and west of each position.
            'square': averages the 8 nearest neighbors of each cell.
            'triangle': averages the three magnets connected to a corner of a hexagon in the kagome geometry.
            'hexagon:' averages each hexagon in kagome ASI, or each star in triangle ASI.
        @return [(2D np.array, 2D np.array)]: a tuple containing two arrays, namely the (averaged) magnetization
            angle and magnitude, respecively, for each relevant position in the simulation.
            Angles lay between 0 and 2*pi, magnitudes between 0 and mm.Msat.
            !! This does not necessarily have the same shape as <m> !!
    '''
    if m is None: m = mm.m
    avg = _resolve_avg(mm, avg)

    if mm.in_plane:
        x_comp = cp.multiply(m, mm.orientation[:,:,0])
        y_comp = cp.multiply(m, mm.orientation[:,:,1])
    else:
        x_comp = m
        y_comp = cp.zeros_like(m)
    mask = _get_avg_mask(mm, avg=avg)
    if mm.PBC:
        magnets_in_avg = signal.convolve2d(cp.abs(m), mask, mode='same', boundary='wrap')
        x_comp_avg = signal.convolve2d(x_comp, mask, mode='same', boundary='wrap')/magnets_in_avg
        y_comp_avg = signal.convolve2d(y_comp, mask, mode='same', boundary='wrap')/magnets_in_avg
    else:
        magnets_in_avg = signal.convolve2d(cp.abs(m), mask, mode='valid', boundary='fill')
        x_comp_avg = signal.convolve2d(x_comp, mask, mode='valid', boundary='fill')/magnets_in_avg
        y_comp_avg = signal.convolve2d(y_comp, mask, mode='valid', boundary='fill')/magnets_in_avg
    angles_avg = cp.arctan2(y_comp_avg, x_comp_avg) % (2*math.pi)
    magnitudes_avg = cp.sqrt(x_comp_avg**2 + y_comp_avg**2)*mm.Msat
    useless_angles = cp.where(cp.logical_and(cp.isclose(x_comp_avg, 0), cp.isclose(y_comp_avg, 0)), cp.NaN, 1) # No well-defined angle
    useless_magnitudes = cp.where(magnets_in_avg == 0, cp.NaN, 1) # No magnet (the NaNs here will be a subset of useless_angles)
    angles_avg *= useless_angles
    magnitudes_avg *= useless_magnitudes
    if avg == 'triangle':
        angles_avg = angles_avg[1::2,1::2]
        magnitudes_avg = magnitudes_avg[1::2,1::2]
    elif avg == 'hexagon': # Only keep the centers of hexagons, throw away the rest
        angles_avg = angles_avg[::2,::2]
        magnitudes_avg = magnitudes_avg[::2,::2]
        ixx, iyy = cp.meshgrid(cp.arange(0, angles_avg.shape[1]), cp.arange(0, angles_avg.shape[0])) # DO NOT REMOVE THIS, THIS IS NOT THE SAME AS mm.ixx, mm.iyy!
        NaN_mask = (ixx + iyy) % 2 == 1 # These are not the centers of hexagons, so dont draw these
        angles_avg[NaN_mask] = cp.NaN
        magnitudes_avg[NaN_mask] = cp.NaN
    return angles_avg, magnitudes_avg

def polar_to_rgb(mm: Magnets, angles=None, magnitudes=None, m=None, avg=True, fill=False, autoscale=True):
    ''' Returns the rgb values for the polar coordinates defined by angles and magnitudes. 
        TAKES CUPY ARRAYS AS INPUT, YIELDS NUMPY ARRAYS AS OUTPUT
        @param angles [2D cp.array()] (None): The averaged angles.
    '''
    if angles is None or magnitudes is None:
        angles, magnitudes = get_m_polar(mm, m=m, avg=avg)
        if autoscale:
            avgmethod = _resolve_avg(mm, avg) # TODO: make this independent of config
            if mm.config in ['pinwheel', 'square'] and avgmethod in ['cross', 'square']:
                magnitudes *= math.sqrt(2)*.999
            elif mm.config in ['kagome', 'triangle'] and avgmethod in ['hexagon', 'triangle']:
                magnitudes *= 1.5*.999
    assert angles.shape == magnitudes.shape, "polar_to_hsv() did not receive angle and magnitude arrays of the same shape."
    
    # Normalize to ranges between 0 and 1 and determine NaN-positions
    angles = angles/2/math.pi
    magnitudes = magnitudes/mm.Msat
    NaNangles = cp.isnan(angles)
    NaNmagnitudes = cp.isnan(magnitudes)
    # Create hue, saturation and value arrays
    hue = cp.zeros_like(angles)
    saturation = cp.ones_like(angles)
    value = cp.zeros_like(angles)
    # Situation 1: angle and magnitude both well-defined (an average => color (hue=angle, saturation=1, value=magnitude))
    affectedpositions = cp.where(cp.logical_and(cp.logical_not(NaNangles), cp.logical_not(NaNmagnitudes)))
    hue[affectedpositions] = angles[affectedpositions]
    value[affectedpositions] = magnitudes[affectedpositions]
    # Situation 2: magnitude is zero, so angle is NaN (zero average => black (hue=anything, saturation=anything, value=0))
    affectedpositions = cp.where(cp.logical_and(NaNangles, magnitudes == 0))
    value[affectedpositions] = 0
    # Situation 3: magnitude is NaN, so angle is NaN (no magnet => white (hue=0, saturation=0, value=1))
    affectedpositions = cp.where(cp.logical_and(NaNangles, NaNmagnitudes))
    saturation[affectedpositions] = 0
    value[affectedpositions] = 1
    # Create the hsv matrix with correct axes ordering for matplotlib.color.hsv_to_rgb:
    hsv = np.array([hue.get(), saturation.get(), value.get()]).swapaxes(0, 2).swapaxes(0, 1)
    if fill: hsv = fill_neighbors(hsv, cp.logical_and(NaNangles, NaNmagnitudes))
    rgb = hsv_to_rgb(hsv)
    return rgb

def show_m(mm: Magnets, m=None, avg=True, show_energy=True, fill=False):
    ''' Shows two (or three if <show_energy> is True) figures displaying the direction of each spin: one showing
        the (locally averaged) angles, another quiver plot showing the actual vectors. If <show_energy> is True,
        a third and similar plot, displaying the interaction energy of each spin, is also shown.
        @param m [2D array] (mm.m): the direction (+1 or -1) of each spin on the geometry. Default is the current
            magnetization profile. This is useful if some magnetization profiles have been saved manually, while 
            mm.update() has been called since: one can then pass these saved profiles as the <m> parameter to
            draw them onto the geometry stored in <mm>.
        @param avg [str|bool] (True): can be any of True, False, 'point', 'cross', 'square', 'triangle', 'hexagon'.
        @param show_energy [bool] (True): if True, a 2D plot of the energy is shown in the figure as well.
        @param fill [bool] (False): if True, empty pixels are interpolated if all neighboring averages are equal.
    '''
    avg = _resolve_avg(mm, avg)
    if m is None: m = mm.m
    show_quiver = mm.m.size < 1e5 and mm.in_plane # Quiver becomes very slow for more than 100k cells, so just dont show it then
    averaged_extent = _get_averaged_extent(mm, avg)
    full_extent = [mm.x_min-mm.dx/2,mm.x_max+mm.dx/2,mm.y_min-mm.dy/2,mm.y_max+mm.dx/2]

    num_plots = 1
    num_plots += 1 if show_energy else 0
    num_plots += 1 if show_quiver else 0
    axes = []
    fig = plt.figure(figsize=(3.5*num_plots, 3))
    ax1 = fig.add_subplot(1, num_plots, 1)
    im = polar_to_rgb(mm, m=m, avg=avg, fill=fill)
    im1 = ax1.imshow(im, cmap='hsv' if mm.in_plane else 'gray', origin='lower', vmin=0, vmax=2*cp.pi,
                        extent=averaged_extent) # extent doesnt work perfectly with triangle or kagome but is still ok
    c1 = plt.colorbar(im1)
    c1.ax.get_yaxis().labelpad = 30
    c1.ax.set_ylabel(f"Averaged magnetization angle [rad]\n('{avg}' average{', PBC' if mm.PBC else ''})", rotation=270, fontsize=12)
    axes.append(ax1)
    if show_quiver:
        ax2 = fig.add_subplot(1, num_plots, 2, sharex=ax1, sharey=ax1)
        ax2.set_aspect('equal')
        nonzero = mm.m.get().nonzero()
        quiverscale = mm._get_plotting_params()['quiverscale']
        ax2.quiver(mm.xx.get()[nonzero], mm.yy.get()[nonzero], 
                cp.multiply(m, mm.orientation[:,:,0]).get()[nonzero], cp.multiply(m, mm.orientation[:,:,1]).get()[nonzero],
                pivot='mid', scale=quiverscale*mm.a, headlength=17, headaxislength=17, headwidth=7, units='xy') # units='xy' makes arrows scale correctly when zooming
        ax2.set_title(r'$m$')
        axes.append(ax2)
    if show_energy:
        ax3 = fig.add_subplot(1, num_plots, num_plots, sharex=ax1, sharey=ax1)
        im3 = ax3.imshow(mm.E_int.get(), origin='lower',
                            extent=full_extent)
        plt.colorbar(im3)
        ax3.set_title(r'$E_{int}$')
        axes.append(ax3)
    multi = MultiCursor(fig.canvas, axes, color='black', lw=1, linestyle='dotted', horizOn=True, vertOn=True) # Assign to variable to prevent garbage collection
    plt.gcf().tight_layout()
    plt.show()

def show_history(mm: Magnets, y_quantity=None, y_label=r'Average magnetization'):
    ''' Plots <y_quantity> (default: average magnetization (mm.history.m)) and total energy (mm.history.E)
        as a function of either the time or the temperature: if the temperature (mm.history.T) is constant, 
        then the x-axis will represent the time (mm.history.t), otherwise it represents the temperature.
        @param y_quantity [1D array] (mm.m): The quantity to be plotted as a function of T or t.
        @param y_label [str] (r'Average magnetization'): The y-axis label in the plot.
    '''
    if y_quantity is None:
        y_quantity = mm.history.m
    if cp.all(cp.isclose(mm.history.T, mm.history.T[0])):
        x_quantity, x_label = mm.history.t, 'Time [a.u.]'
    else:
        x_quantity, x_label = mm.history.T, 'Temperature [a.u.]'
    assert len(y_quantity) == len(x_quantity), "Error in show_history: <y_quantity> has different length than %s history." % x_label.split(' ')[0].lower()

    fig = plt.figure(figsize=(4, 6))
    ax1 = fig.add_subplot(211)
    ax1.plot(x_quantity, y_quantity)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax2 = fig.add_subplot(212)
    ax2.plot(x_quantity, mm.history.E)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Total energy [a.u.]')
    plt.gcf().tight_layout()
    plt.show()

def get_AFMness(mm: Magnets, AFM_mask=None):
    ''' Returns the average AFM-ness of mm.m at the current time step, normalized to 1.
        For a perfectly uniform configuration this is 0, while for random it is 0.375.
        Note that the boundaries are not taken into account for the normalization, so the
        AFM-ness will often be slightly lower than the ideal values mentioned above.
        @param AFM_mask [2D array] (None): The mask used to determine the AFM-ness. If not
            provided explicitly, it is determined automatically based on mm.config.
        @return [float]: The average normalized AFM-ness.
    '''
    AFM_mask = mm._get_AFMmask() if AFM_mask is None else cp.asarray(AFM_mask)
    AFM_ness = cp.mean(cp.abs(signal.convolve2d(mm.m, AFM_mask, mode='same', boundary='wrap' if mm.PBC else 'fill')))
    return float(AFM_ness/cp.sum(cp.abs(AFM_mask))/cp.sum(mm.mask)*mm.m.size)

def fill_neighbors(hsv, replaceable, fillblack=True): # TODO: make this cupy if possible
    ''' THIS FUNCTION ONLY WORKS FOR GRIDS WHICH HAVE A CHESS-LIKE OCCUPATION OF THE CELLS! (cross ⁛)
        THIS FUNCTION OPERATES ON HSV VALUES, AND RETURNS HSV AS WELL!!! NOT RGB HERE!
        The 2D array <replaceable> is True at the positions of hsv which can be overwritten by this function.
        The 3D array <hsv> has the same first two dimensions as <replaceable>, with the third dimension having size 3 (h, s, v).
        Then this function overwrites the replaceables with the surrounding values at the nearest neighbors (cross neighbors ⁛),
        but only if all those neighbors are equal. This is useful for very large simulations where each cell
        occupies less than 1 pixel when plotted: by removing the replaceables, visual issues can be prevented.
        @param fillblack [bool] (True): If True, white pixels next to black pixels are colored black regardless of other neighbors.
        @return [2D np.array]: The interpolated array.
    '''
    hsv = hsv.get() if type(hsv) == cp.ndarray else np.asarray(hsv)
    replaceable = replaceable if type(replaceable) == cp.ndarray else cp.asarray(replaceable)

    # Extend arrays a bit to fill NaNs near boundaries as well
    a = np.insert(hsv, 0, hsv[1], axis=0)
    a = np.insert(a, 0, a[:,1], axis=1)
    a = cp.append(a, a[-2].reshape(1,-1,3), axis=0)
    a = cp.append(a, a[:,-2].reshape(-1,1,3), axis=1)

    N = a[:-2, 1:-1, :]
    E = a[1:-1, 2:, :]
    S = a[2:, 1:-1, :]
    W = a[1:-1, :-2, :]
    equal_neighbors = cp.logical_and(cp.logical_and(cp.isclose(N, E), cp.isclose(E, S)), cp.isclose(S, W))
    equal_neighbors = cp.logical_and(cp.logical_and(equal_neighbors[:,:,0], equal_neighbors[:,:,1]), equal_neighbors[:,:,2])

    result = cp.where(cp.repeat(cp.logical_and(replaceable, equal_neighbors)[:,:,cp.newaxis], 3, axis=2), N, cp.asarray(hsv))

    if fillblack:
        blacks = cp.where(cp.logical_and(cp.logical_or(cp.logical_or(N[:,:,2] == 0, E[:,:,2] == 0), cp.logical_or(S[:,:,2] == 0, W[:,:,2] == 0)), replaceable))
        result[blacks[0], blacks[1], 2] = 0

    return result.get()