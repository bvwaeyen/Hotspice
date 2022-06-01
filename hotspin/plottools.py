import ctypes
import math
import matplotlib
import os
import warnings

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cupyx.scipy import signal
from enum import Enum
from matplotlib import cm, colors, patches, widgets

from .core import Magnets

ctypes.windll.shcore.SetProcessDpiAwareness(2) # (For Windows 10/8/7) this makes the matplotlib plots smooth on high DPI screens
matplotlib.rcParams["image.interpolation"] = 'none' # 'none' works best for large images scaled down, 'nearest' for the opposite


class Average(Enum):
    def __new__(cls, *args, **kwargs): # To allow specifying extra properties like <mask> while retaining enumeration
          obj = object.__new__(cls)
          obj._value_ = len(cls.__members__) + 1
          return obj
    def __init__(self, mask):
          self.mask = cp.asarray(mask, dtype='float')
    
    POINT = [[1]]
    CROSS = [[0, 1, 0], # For Pinwheel and Square ASI
             [1, 0, 1],
             [0, 1, 0]]
    SQUARE = [[1, 1, 1], # Often not very useful, just the same as CROSS in most cases
              [1, 0, 1],
              [1, 1, 1]]
    SQUAREFOUR = [[1, 1, 1], # Because nearly all ASIs have cells with 4 neighbors, this can come in handy
                  [1, 4, 1],
                  [1, 1, 1]]
    HEXAGON = [[0, 1, 0, 1, 0], # For Kagome ASI
               [1, 0, 0, 0, 1],
               [0, 1, 0, 1, 0]]
    TRIANGLE = [[0, 1, 0], # For Triangle ASI
                [1, 0, 1],
                [0, 1, 0]]

    @classmethod
    def resolve(cls, avg, mm: Magnets=None):
        ''' <avg> can be any of [str], [bool-like], or [Average]. This function will
            then return the [Average] instance that is most appropriate.
        '''
        match avg:
            case Average():
                return avg
            case str():
                for average in Average:
                    if average.name.upper() == avg.upper():
                        return average
                raise ValueError(f"Unsupported averaging mask: {avg}")
            case _ if avg: # If avg is not str() or Average(), but is still truthy
                return Average.resolve(mm._get_appropriate_avg()) if mm is not None else Average.SQUARE
            case _: # If avg is falsy
                return Average.POINT


# Below here are some graphical functions (plot magnetization profile etc.)
def _get_averaged_extent(mm: Magnets, avg):
    ''' Returns the extent (in meters) that can be used in imshow when plotting an averaged quantity. '''
    avg = Average.resolve(avg, mm)
    mask = avg.mask
    if mm.PBC:
        movex, movey = 0.5*mm.dx, 0.5*mm.dy
    else:
        movex, movey = mask.shape[1]/2*mm.dx, mask.shape[0]/2*mm.dy # The averaged imshow should be displaced by this much
    return [mm.x_min-mm.dx+movex,mm.x_max-movex+mm.dx,mm.y_min-mm.dy+movey,mm.y_max-movey+mm.dy] # [m]

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
            Angles lay between 0 and 2*pi, magnitudes between 0 and mm.moment.
            !! This does not necessarily have the same shape as <m> !!
    '''
    if m is None: m = mm.m
    avg = Average.resolve(avg, mm)

    if mm.in_plane:
        x_comp = cp.multiply(m, mm.orientation[:,:,0])
        y_comp = cp.multiply(m, mm.orientation[:,:,1])
    else:
        x_comp = m
        y_comp = cp.zeros_like(m)
    mask = avg.mask
    if mm.PBC:
        magnets_in_avg = signal.convolve2d(cp.abs(m), mask, mode='same', boundary='wrap')
        x_comp_avg = signal.convolve2d(x_comp, mask, mode='same', boundary='wrap')/magnets_in_avg
        y_comp_avg = signal.convolve2d(y_comp, mask, mode='same', boundary='wrap')/magnets_in_avg
    else:
        magnets_in_avg = signal.convolve2d(cp.abs(m), mask, mode='valid', boundary='fill')
        x_comp_avg = signal.convolve2d(x_comp, mask, mode='valid', boundary='fill')/magnets_in_avg
        y_comp_avg = signal.convolve2d(y_comp, mask, mode='valid', boundary='fill')/magnets_in_avg
    angles_avg = cp.arctan2(y_comp_avg, x_comp_avg) % (2*math.pi)
    magnitudes_avg = cp.sqrt(x_comp_avg**2 + y_comp_avg**2)*mm.moment
    useless_angles = cp.where(cp.isclose(x_comp_avg, 0) & cp.isclose(y_comp_avg, 0), cp.NaN, 1) # No well-defined angle
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
        NaN_occupation = (ixx + iyy) % 2 == 1 # These are not the centers of hexagons, so dont draw these
        angles_avg[NaN_occupation] = cp.NaN
        magnitudes_avg[NaN_occupation] = cp.NaN
    return angles_avg, magnitudes_avg

def get_hsv(mm: Magnets, angles=None, magnitudes=None, m=None, avg=True, fill=False, autoscale=True):
    ''' Returns the hsv values for the polar coordinates defined by angles [rad] and magnitudes [A/m]. 
        TAKES CUPY ARRAYS AS INPUT, YIELDS NUMPY ARRAYS AS OUTPUT
        @param angles [2D cp.array()] (None): The averaged angles.
    '''
    if angles is None or magnitudes is None:
        angles, magnitudes = get_m_polar(mm, m=m, avg=avg)
        if autoscale and mm.in_plane:
            s = cp.sign(mm.orientation[:,:,0])
            mask = Average.resolve(avg).mask
            n = signal.convolve2d(mm.occupation, mask, mode='same', boundary='wrap' if mm.PBC else 'fill')
            max_mag_x = signal.convolve2d(mm.orientation[:,:,0]*s, mask, mode='same', boundary='wrap' if mm.PBC else 'fill')
            max_mag_y = signal.convolve2d(mm.orientation[:,:,1]*s, mask, mode='same', boundary='wrap' if mm.PBC else 'fill')
            max_mean_magnitude = cp.sqrt(max_mag_x**2 + max_mag_y**2)/n
            ny, nx = magnitudes.shape
            shape = max_mean_magnitude.shape
            max_mean_magnitude = max_mean_magnitude[(shape[0]-ny)//2:(shape[0]-ny)//2+ny, (shape[1]-nx)//2:(shape[1]-nx)//2+nx]
            magnitudes = magnitudes/max_mean_magnitude*.9999 # times .9999 to prevent rounding errors yielding values larger than 1
    assert angles is not None and magnitudes is not None, "Angles and/or magnitudes could not be resolved."
    if angles.shape != magnitudes.shape: raise ValueError(f"Angle and/or magnitude arrays have different shape: {angles.shape} and {magnitudes.shape}.")
    
    # Normalize to ranges between 0 and 1 and determine NaN-positions
    angles = angles/2/math.pi
    magnitudes = magnitudes/mm.moment
    NaNangles = cp.isnan(angles)
    NaNmagnitudes = cp.isnan(magnitudes)
    # Create hue, saturation and value arrays
    hue = cp.zeros_like(angles)
    saturation = cp.ones_like(angles)
    value = cp.zeros_like(angles)
    # Situation 1: angle and magnitude both well-defined (an average => color (hue=angle, saturation=1, value=magnitude))
    affectedpositions = cp.where(~NaNangles & ~NaNmagnitudes)
    hue[affectedpositions] = angles[affectedpositions]
    value[affectedpositions] = magnitudes[affectedpositions]
    # Situation 2: magnitude is zero, so angle is NaN (zero average => black (hue=anything, saturation=anything, value=0))
    affectedpositions = cp.where(NaNangles & (magnitudes == 0))
    value[affectedpositions] = 0
    # Situation 3: magnitude is NaN, so angle is NaN (no magnet => white (hue=0, saturation=0, value=1))
    affectedpositions = cp.where(NaNangles & NaNmagnitudes)
    saturation[affectedpositions] = 0
    value[affectedpositions] = 1
    # Create the hsv matrix with correct axes ordering for matplotlib.colors.hsv_to_rgb:
    hsv = np.array([hue.get(), saturation.get(), value.get()]).swapaxes(0, 2).swapaxes(0, 1)
    if fill: hsv = fill_neighbors(hsv, NaNangles & NaNmagnitudes, mm=mm)
    return hsv

def get_rgb(*args, **kwargs):
    return colors.hsv_to_rgb(get_hsv(*args, **kwargs))

def show_m(mm: Magnets, m=None, avg=True, show_energy=True, fill=True, overlay_quiver=False, color_quiver=True, figure=None):
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
        @param overlay_quiver [bool] (False): if True, the quiver plot is shown overlaid on the color average plot.
        @param color_quiver [bool] (True): if True, the quiver plot arrows are colored according to their angle.
        @param figure [matplotlib.Figure] (None): if specified, that figure is used to redraw this show_m().
    '''
    avg = Average.resolve(avg, mm)
    if m is None: m = mm.m
    show_quiver = mm.m.size < 1e5 and mm.in_plane # Quiver becomes very slow for more than 100k cells, so just dont show it then
    averaged_extent = _get_averaged_extent(mm, avg)
    full_extent = [mm.x_min-mm.dx/2,mm.x_max+mm.dx/2,mm.y_min-mm.dy/2,mm.y_max+mm.dx/2]

    num_plots = 1
    num_plots += 1 if show_energy else 0
    num_plots += (0 if overlay_quiver else 1) if show_quiver else 0
    axes = []
    if figure is not None and not plt.isinteractive(): init_interactive()
    if not plt.get_fignums(): figure = None
    fig = plt.figure(figsize=(3.5*num_plots, 3)) if figure is None else figure
    if figure is not None: fig.clear()
    ax1 = fig.add_subplot(1, num_plots, 1)
    im = get_rgb(mm, m=m, avg=avg, fill=fill)
    cmap = cm.get_cmap('hsv')
    if mm.in_plane:
        im1 = ax1.imshow(im, cmap='hsv', origin='lower', vmin=0, vmax=2*cp.pi,
                        extent=averaged_extent, interpolation='antialiased', interpolation_stage='rgba') # extent doesnt work perfectly with triangle or kagome but is still ok
        c1 = plt.colorbar(im1)
        c1.ax.get_yaxis().labelpad = 30
        c1.ax.set_ylabel(f"Averaged magnetization angle [rad]\n('{avg.name.lower()}' average{', PBC' if mm.PBC else ''})", rotation=270, fontsize=10)
    else:
        r0, g0, b0, _ = cmap(.5) # Value at angle 'pi' (-1)
        r1, g1, b1, _ = cmap(0) # Value at angle '0' (1)
        cdict = {'red':   [[0.0,  r0, r0], # x, value_left, value_right
                   [0.5,  0.0, 0.0],
                   [1.0,  r1, r1]],
         'green': [[0.0,  g0, g0],
                   [0.5, 0.0, 0.0],
                   [1.0,  g1, g1]],
         'blue':  [[0.0,  b0, b0],
                   [0.5,  0.0, 0.0],
                   [1.0,  b1, b1]]}
        newcmap = colors.LinearSegmentedColormap('OOP_cmap', segmentdata=cdict, N=256)
        im1 = ax1.imshow(im, cmap=newcmap, origin='lower', vmin=-1, vmax=1,
                         extent=averaged_extent, interpolation='antialiased', interpolation_stage='rgba')
        c1 = plt.colorbar(im1)
        c1.ax.get_yaxis().labelpad = 30
        c1.ax.set_ylabel(f"Averaged magnetization\n('{avg.name.lower()}' average{', PBC' if mm.PBC else ''})", rotation=270, fontsize=10)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    axes.append(ax1)
    if show_quiver:
        if overlay_quiver:
            ax2 = ax1
        else:
            ax2 = fig.add_subplot(1, num_plots, 2, sharex=ax1, sharey=ax1)
            ax2.set_aspect('equal')
            ax2.set_title(r'$m$')
            ax2.set_xlabel('x [m]')
            ax2.set_ylabel('y [m]')
            axes.append(ax2)
        nonzero = mm.m.get().nonzero()
        mx, my = cp.multiply(m, mm.orientation[:,:,0]).get()[nonzero], cp.multiply(m, mm.orientation[:,:,1]).get()[nonzero]
        ax2.quiver(mm.xx.get()[nonzero], mm.yy.get()[nonzero], mx, my,
                color=(cmap((np.arctan2(my, mx)/2/np.pi) % 1) if color_quiver else 'black'),
                pivot='mid', scale=1.1/mm._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7, units='xy') # units='xy' makes arrows scale correctly when zooming
    if show_energy:
        ax3 = fig.add_subplot(1, num_plots, num_plots, sharex=ax1, sharey=ax1)
        im3 = ax3.imshow(np.where(mm.m.get() != 0, mm.E.get(), np.nan), origin='lower',
                            extent=full_extent, interpolation='antialiased', interpolation_stage='rgba')
        c3 = plt.colorbar(im3)
        c3.ax.get_yaxis().labelpad = 15
        c3.ax.set_ylabel("Local energy [J]", rotation=270, fontsize=10)
        ax3.set_title(r'$E_{int}$')
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('y [m]')
        axes.append(ax3)
    multi = widgets.MultiCursor(fig.canvas, axes, color='black', lw=1, linestyle='dotted', horizOn=True, vertOn=True) # Assign to variable to prevent garbage collection
    plt.gcf().tight_layout()
    if figure is None:
        plt.show()
        if plt.isinteractive(): update_interactive(fig)
    else:
        update_interactive(fig)
    return fig

def show_lattice(mm: Magnets, nx: int = 3, ny: int = 3, fall_off: float = 1, scale: float = .8, save: bool = False):
    ''' Shows a minimalistic rendition of the lattice on which the spins are placed.
        @param mm [Magnets]: an instance of the ASI class whose lattice should be plotted.
        @param nx, ny [int] (3): the number of unit cells that will be shown.
        @param fall_off [float] (1): how many unit cells it takes for the opacity to drop from 1 to 0 near the edge.
        @param scale [float] (.8): how long the shown ellipses are, as a multiple of the distance between nearest neighbors.
        @param save [bool] (False): if True, the figure is saved as "results/lattices/<ASI_name>_<nx>x<ny>.pdf
    '''
    nx, ny = nx*mm.unitcell.x+1, ny*mm.unitcell.y+1
    if mm.nx < nx or mm.ny < ny:
        raise ValueError(f"Lattice of {type(mm).__name__} is too small: ({mm.nx}x{mm.ny})<({nx}x{ny}).")

    occupation = mm.occupation[:ny,:nx]
    positions_x = mm.xx[np.where(occupation)].get()
    positions_y = mm.yy[np.where(occupation)].get()
    xmin, xmax, ymin, ymax = positions_x.min(), positions_x.max(), positions_y.min(), positions_y.max()
    ux, uy = mm.unitcell.x*mm.dx, mm.unitcell.y*mm.dy

    figsize = 3
    fig = plt.figure(figsize=(figsize, figsize*uy*ny/ux/nx))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_axis_off()
    size = mm._get_closest_dist()*scale
    if mm.in_plane:
        ox = mm.orientation[:,:,0][np.where(occupation)].get()
        oy = mm.orientation[:,:,1][np.where(occupation)].get()
        angles = np.arctan2(oy, ox)*180/math.pi
    else:
        angles = np.zeros_like(positions_x)

    for i in range(positions_x.size):
        px, py = positions_x[i], positions_y[i]
        edgedist = min([(px-xmin)/ux, (xmax-px)/ux, (py-ymin)/uy, (ymax-py)/uy, fall_off])/fall_off # Normalized distance to edge of figure (between 0 and 1)
        alpha = max(edgedist, 0.1)
        ax.add_artist(patches.Ellipse((px, py), size, (size/2 if mm.in_plane else size), angle=angles[i], alpha=alpha, ec=None))
    ax.set_xlim(xmin-size/2, xmax+size/2)
    ax.set_ylim(ymin-size/2, ymax+size/2)

    plt.gcf().tight_layout()
    if save: save_plot(f'results/lattices/{type(mm).__name__}_{nx//mm.unitcell.x:.0f}x{ny//mm.unitcell.y:.0f}.pdf')
    plt.show()

def show_history(mm: Magnets, *, y_quantity=None, y_label=r'Average magnetization'):
    ''' Plots <y_quantity> (default: average magnetization (mm.history.m)) and total energy (mm.history.E)
        as a function of either the time or the temperature: if the temperature (mm.history.T) is constant, 
        then the x-axis will represent the time (mm.history.t), otherwise it represents the temperature.
        @param y_quantity [1D array] (mm.m): The quantity to be plotted as a function of T or t.
        @param y_label [str] (r'Average magnetization'): The y-axis label in the plot.
    '''
    if y_quantity is None:
        y_quantity = mm.history.m
    if cp.all(cp.isclose(mm.history.T, mm.history.T[0])):
        x_quantity, x_label = mm.history.t, 'Time [s]'
    else:
        x_quantity, x_label = mm.history.T, 'Temperature [K]'
    if len(y_quantity) != len(x_quantity): raise ValueError(f"y_quantity has different length than history {x_label.split(' ')[0].lower()} array.")

    fig = plt.figure(figsize=(4, 6))
    ax1 = fig.add_subplot(211)
    ax1.plot(x_quantity, y_quantity)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax2 = fig.add_subplot(212)
    ax2.plot(x_quantity, mm.history.E)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Total energy [J]')
    plt.gcf().tight_layout()
    plt.show()

def get_AFMness(mm: Magnets, AFM_mask=None):
    ''' Returns the average AFM-ness of mm.m at the current time step, normalized to 1.
        For a perfectly uniform configuration this is 0, while for random it is 0.375.
        Note that the boundaries are not taken into account for the normalization, so the
        AFM-ness will often be slightly lower than the ideal values mentioned above.
        @param AFM_mask [2D array] (None): The mask used to determine the AFM-ness. If not
            provided explicitly, it is determined automatically based on the type of ASI.
        @return [float]: The average normalized AFM-ness.
    '''
    AFM_mask = mm._get_AFMmask() if AFM_mask is None else cp.asarray(AFM_mask)
    AFM_ness = cp.mean(cp.abs(signal.convolve2d(mm.m, AFM_mask, mode='same', boundary='wrap' if mm.PBC else 'fill')))
    return float(AFM_ness/cp.sum(cp.abs(AFM_mask))/cp.sum(mm.occupation)*mm.m.size)

def fill_neighbors(hsv, replaceable, mm=None, fillblack=False, fillwhite=False): # TODO: this is quite messy because we are working with color here instead of angles/magnitudes
    ''' THIS FUNCTION ONLY WORKS FOR GRIDS WHICH HAVE A CHESS-LIKE OCCUPATION OF THE CELLS! (cross ⁛)
        THIS FUNCTION OPERATES ON HSV VALUES, AND RETURNS HSV AS WELL!!! NOT RGB HERE!
        The 2D array <replaceable> is True at the positions of hsv which can be overwritten by this function.
        The 3D array <hsv> has the same first two dimensions as <replaceable>, with the third dimension having size 3 (h, s, v).
        Then this function overwrites the replaceables with the surrounding values at the nearest neighbors (cross neighbors ⁛),
        but only if all those neighbors are equal. This is useful for very large simulations where each cell
        occupies less than 1 pixel when plotted: by removing the replaceables, visual issues can be prevented.
        @param fillblack [bool] (False): If True, white pixels next to black pixels are colored black regardless of other neighbors.
        @param fillwhite [bool] (False): If True, white pixels are colored in using the Average.SQUAREFULL rule.
        @return [2D np.array]: The interpolated array.
    '''
    if hsv.shape[0] < 2 or hsv.shape[1] < 2: return hsv
    hsv = hsv.get() if isinstance(hsv, cp.ndarray) else np.asarray(hsv)
    replaceable = cp.asarray(replaceable, dtype='bool')

    # Extend arrays a bit to fill NaNs near boundaries as well
    a = np.insert(hsv, 0, hsv[1], axis=0) # TODO: make this cupy once cupy supports the .insert function
    a = np.insert(a, 0, a[:,1], axis=1)
    a = cp.append(a, a[-2].reshape(1,-1,3), axis=0)
    a = cp.append(a, a[:,-2].reshape(-1,1,3), axis=1)

    N = a[:-2, 1:-1, :]
    E = a[1:-1, 2:, :]
    S = a[2:, 1:-1, :]
    W = a[1:-1, :-2, :]
    equal_neighbors = cp.isclose(N, E) & cp.isclose(E, S) & cp.isclose(S, W)
    equal_neighbors = equal_neighbors[:,:,0] & equal_neighbors[:,:,1] & equal_neighbors[:,:,2]

    result = cp.where(cp.repeat((replaceable & equal_neighbors)[:,:,cp.newaxis], 3, axis=2), N, cp.asarray(hsv))

    if fillblack:
        blacks = cp.where(replaceable & ((N[:,:,2] == 0) | (E[:,:,2] == 0) | (S[:,:,2] == 0) | (W[:,:,2] == 0)))
        result[blacks[0], blacks[1], 2] = 0
    if fillwhite: # If any NaNs remain, they will be white, and we might want to fill them as well
        whites = cp.where((result[:,:,1] == 0) & (result[:,:,2] == 1))
        substitution_colors = cp.asarray(get_hsv(mm, fill=False, avg=Average.SQUAREFOUR))
        result[whites[0], whites[1], :] = substitution_colors[whites[0], whites[1], :]

    return result.get()


def init_fonts(backend=True):
    ''' Sets various parameters for consistent plotting across all Hotspin scripts.
        This should be called before instantiating any subplots.
        This should not be called directly by any function in hotspin.plottools itself,
        only by higher-level scripts (e.g. examples, analyses, tests...) which can
        then decide for themselves whether or not to use these standardized settings.
        @param backend [bool] (True): if True, the tkinter backend is activated. This
            backend is preferred since it allows consistent updating of interactive plots.
    '''
    if backend:
        try:
            matplotlib.use("TkAgg") # tkinter backend is preferred by Hotspin
        except ImportError:
            warnings.warn(f"Could not activate 'TkAgg' backend for Hotspin (using {matplotlib.get_backend()} instead).", stacklevel=2)
    
    # Call this function before instantiating subplots!
    smol = 10
    medium = 11
    chonk = 12

    plt.rc('font', size=smol)          # controls default text sizes
    plt.rc('axes', titlesize=medium)     # fontsize of the axes title
    plt.rc('axes', labelsize=smol)   # fontsize of the x and y labels
    plt.rc('xtick', labelsize=smol)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=smol)    # fontsize of the tick labels
    plt.rc('legend', fontsize=smol)    # legend fontsize
    plt.rc('figure', titlesize=chonk)  # fontsize of the figure title


def init_interactive():
    ''' Call this once before starting to build an interactive (i.e. real-time updatable) plot. '''
    plt.ion()

def update_interactive(figure=None):
    ''' Update all the visuals of the most recent figure so it is up-to-date. '''
    # Interactive functions in hotspin.plottools probably already call this function by themselves.
    fig = plt.gcf() if figure is None else figure
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def close_interactive(figure=None):
    if figure is not None:
        plt.close(figure)
    else:
        plt.close()


def save_plot(save_path: str, ext=None):
    ''' <save_path> is a full relative pathname, usually something like
        "results/<test_or_experiment_name>/<relevant_params=...>.pdf"
    '''
    if ext is not None: # Then a specific extension was requested, to override the one in save_path
        original_ext = os.path.splitext(save_path)[1]
        if not original_ext[0].isdigit(): # Then probably the original 'extension' is actually real, not just the long part after a random decimal point in the string
            save_path = os.path.splitext(save_path)[0]
        save_path += '.' + ext.removeprefix('.') # This slightly convoluted way allows <ext> to be e.g. ".pdf" but also just "pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        plt.savefig(save_path)
    except PermissionError:
        warnings.warn(f'Could not save to {save_path}, probably because the file is opened somewhere else.', stacklevel=2)


if __name__ == "__main__":
    print(Average.TRIANGLE.mask)