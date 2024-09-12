""" This file creates plots of the 12 most important ASIs that are standard in Hotspice.
    The figure "ASIs.pdf" in the output directory is used in the Hotspice paper as Figure 1.
    The other figures generated in the output directory show the lattices individually.
"""

import matplotlib
matplotlib.use("Agg")

import hotspice
import matplotlib.pyplot as plt
import numpy as np

xp = hotspice.xp
from hotspice.utils import asnumpy
from matplotlib import lines, patches
from typing import Type

import _example_plot_utils as epu
FIGSIZE = 5 #! DO NOT CHANGE, FONT SIZES DON'T SCALE. This is a standard multiplier for the width of each lattice plot.

def get_lattice_fig(ASI_type: Type[hotspice.Magnets], radius_a: float = 5, fade_width: float = None, scale: float = .8, a: tuple = (0,0,0), ax: plt.Axes = None, **kwargs):
    """ Shows a minimalistic rendition of the lattice on which the spins are placed.
        Includes the unit cell and the lattice parameter `a`.
        NOTE: This function returns the figure or axes object, so does NOT save to a file or show the figure.
        @param ASI_type: an ASI class (not an instance!) whose lattice should be plotted.
        @param radius_a [float] (5): the viewport will be `2*radius_a` by `2*radius_a`, centered on the center of a unit cell.
        @param fade_width [float] (`radius_a/3`): how many units of `a` it takes for the opacity to drop from 1 to 0 near the edge.
        @param scale [float] (.8): how long the shown ellipses are, as a multiple of the distance between nearest neighbors.
        @param a [tuple(3)]: tuple of 3 floats, indicating (start_ix, start_iy, angle) of the line indicating the definition
            of the lattice parameter `a`, relative to the first element of the highlighted unitcell.
    """
    ALPHA_BY_ELLIPSE_BBOX = False
    
    mm = ASI_type(a=1, n=10*radius_a) # 10 should be safe for all builtin lattices
    if fade_width is None: fade_width = radius_a / 3
    
    unitcell_x = xp.sum(mm.dx[:mm.unitcell.x])
    unitcell_y = xp.sum(mm.dy[:mm.unitcell.y])
    
    ## Determine highlighted unit cell
    start_ix = (mm.nx // mm.unitcell.x)//2*mm.unitcell.x  # Lowest x-index in the central unitcell
    start_iy = (mm.ny // mm.unitcell.y)//2*mm.unitcell.y  # Lowest y-index in the central unitcell
    rect_range_x = (mm.x[start_ix] - mm.dx[start_ix-1]/2, mm.x[start_ix+mm.unitcell.x] - mm.dx[start_ix+mm.unitcell.x-1]/2)
    rect_range_y = (mm.y[start_iy] - mm.dy[start_iy-1]/2, mm.y[start_iy+mm.unitcell.y] - mm.dy[start_iy+mm.unitcell.y-1]/2)
    center_x = np.mean(rect_range_x)
    center_y = np.mean(rect_range_y)

    occupied_indices = xp.where(mm.occupation)
    positions_x = asnumpy(mm.xx[occupied_indices])
    positions_y = asnumpy(mm.yy[occupied_indices])

    if ax is None:
        fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
        ax = fig.add_axes([0,0,1,1])
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.margins(0)
    if mm.in_plane:
        ox = asnumpy(mm.orientation[:,:,0][occupied_indices])
        oy = asnumpy(mm.orientation[:,:,1][occupied_indices])
        angles = np.arctan2(oy, ox)*180/np.pi
    else:
        angles = np.zeros_like(positions_x)

    rect = patches.Rectangle((rect_range_x[0], rect_range_y[0]), unitcell_x, unitcell_y, linewidth=1, ec=None, fc='gray', alpha=0.3)
    ax.add_patch(rect)

    size = mm._get_closest_dist()*scale
    xmin, xmax, ymin, ymax = center_x - radius_a, center_x + radius_a, center_y - radius_a, center_y + radius_a
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for i in range(positions_x.size):
        px, py = positions_x[i], positions_y[i]
        if not (xmin < px < xmax) or not (ymin < py < ymax): continue # Only include magnets that are entirely inside the viewport
        color = 'k' if (rect_range_x[0] < px < rect_range_x[1]) and (rect_range_y[0] < py < rect_range_y[1]) else 'C0'
        ellipse = patches.Ellipse((px, py), size, (size/2 if mm.in_plane else size), angle=angles[i], fc=color, ec=None)
        if ALPHA_BY_ELLIPSE_BBOX:
            bbox = ellipse.get_corners()
            exmin, exmax, eymin, eymax = np.min(bbox[:,0]), np.max(bbox[:,0]), np.min(bbox[:,1]), np.max(bbox[:,1]) # Ellipse orthogonal bounding box
            edgedist = min(exmin - xmin, xmax - exmax, eymin - ymin, ymax - eymax) # Exact distance to edge of figure
        else:
            edgedist = float(max(min(px-xmin, xmax-px, py-ymin, ymax-py) - size/2, 0)) # Underestimation of distance to edge of figure (>= 0)
        ellipse.set_alpha(np.clip(edgedist/fade_width, 0, 1))
        ax.add_artist(ellipse)
    
    ## Add indication of lattice parameter a
    a1, a2, aa = mm.x[start_ix + a[0]], mm.y[start_iy + a[1]], 0 if len(a) == 2 else a[2]
    a_kwargs = dict(color='r', linewidth=4)
    d = 1.2 # Size of arrows
    ax.annotate('', xy=(a1, a2), xytext=(a1+np.cos(aa), a2+np.sin(aa)), arrowprops=dict(arrowstyle='<|-|>', mutation_scale=3*d*FIGSIZE, shrinkA=0, shrinkB=0, joinstyle='miter', capstyle='butt') | a_kwargs)
    ax.annotate('', xy=(a1, a2), xytext=(a1+np.cos(aa), a2+np.sin(aa)), arrowprops=dict(arrowstyle='|-|', mutation_scale=1.25*d*FIGSIZE, shrinkA=0, shrinkB=0, joinstyle='miter', capstyle='butt') | a_kwargs | dict(linewidth=2*d))
    
    try: return fig
    except: return ax


def show_all_lattices():
    """ This creates Figure 1 from the Hotspice paper. """
    ## Create individual plots
    scale_OOP = 0.75
    ASIs = {
        hotspice.ASI.IP_Pinwheel_Diamond: dict(radius_a=2.7, a=(-1,0), name="Pinwheel\n(diamond)"),
        hotspice.ASI.IP_Pinwheel_LuckyKnot: dict(radius_a=3, a=(0,1,-np.pi/4), name="Pinwheel\n(lucky-knot)"),
        hotspice.ASI.IP_Square_Closed: dict(radius_a=2.7, a=(0,1), name="Square\n(closed)"),
        hotspice.ASI.IP_Square_Open: dict(radius_a=3, a=(0,1,-np.pi/4), name="Square\n(open)"),
        hotspice.ASI.IP_Ising: dict(radius_a=3.5, a=(0,0)),
        hotspice.ASI.IP_Triangle: dict(radius_a=2.45, scale=1, a=(0,1)),
        hotspice.ASI.IP_Kagome: dict(radius_a=2.15, a=(0,1)),
        hotspice.ASI.IP_Cairo: dict(radius_a=4, scale=1, a=(5,4,np.pi-hotspice.ASI.IP_Cairo.BETA)),
        hotspice.ASI.OOP_Square: dict(radius_a=3.5, scale=scale_OOP),
        hotspice.ASI.OOP_Triangle: dict(radius_a=3.5, scale=scale_OOP, a=(0,1)),
        hotspice.ASI.OOP_Honeycomb: dict(radius_a=4, scale=scale_OOP, a=(0,1,np.pi/2)),
        hotspice.ASI.OOP_Cairo: dict(radius_a=4, scale=scale_OOP, a=(2,3))
        }
    
    figs = {ASI.__name__: get_lattice_fig(ASI, **kwargs) for ASI, kwargs in ASIs.items()}
    hotspice.utils.save_results(figures=figs, timestamped=False)
    for fig in figs.values(): plt.close(fig)
    
    ## Combine all plots
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(6*FIGSIZE, 2.5*FIGSIZE))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, wspace=0.05, hspace=0.25)
    for i, (ASI, kwargs) in enumerate(ASIs.items()):
        ax: plt.Axes = axes.T.flat[i]
        get_lattice_fig(ASI, ax=ax, **kwargs)
        name = kwargs.get('name', ' '.join(ASI.__name__.split('_')[1:]).lower().capitalize())
        ax.set_title(name, fontdict={'fontsize': 28})
        epu.label_ax(ax, i, fontsize=32, offset=(0.01, 0)) # To move some labels up, use: form="(%s)" + ("\n" if i < 4 else "")
    # Add in-plane and out-of-plane titles
    textkwargs = dict(fontsize=36, weight='bold', rotation='horizontal', ha='center', va='center')
    linekwargs = dict(color='k', linestyle="-", linewidth=6, markeredgewidth=6, marker="|", markersize=32, solid_capstyle='butt')
    cutoff, y = 2/3, 0.96
    fig.text((1 + cutoff + 0.01)/2, y, "Out-of-plane", **textkwargs)
    fig.add_artist(lines.Line2D([cutoff + 0.01, 0.995], [1 - (1-y)*2]*2, **linekwargs))
    fig.text((cutoff - 0.01)/2, y, "In-plane", **textkwargs)
    fig.add_artist(lines.Line2D([0.005, cutoff - 0.01], [1 - (1-y)*2]*2, **linekwargs))
    hotspice.utils.save_results(figures={"ASIs": fig}, timestamped=False)


if __name__ == "__main__":
    """ EXPECTED RUNTIME: <30s, since no dynamics are simulated, only figures are plotted. """
    show_all_lattices()