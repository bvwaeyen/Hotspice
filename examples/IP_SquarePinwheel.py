r""" This file tests if Hotspice can correctly reproduce the transition
    angle between pinwheel and square FM/AFM ordering.
    Based on the paper
        Mac\^edo, R., Macauley, G. M., Nascimento, F. S., & Stamps, R. L. (2018).
        Apparent ferromagnetism in the pinwheel artificial spin ice.
        Physical Review B, 98(1), 014437.
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal

import hotspice
import _example_plot_utils as epu


def count_AFM_cells(mm: hotspice.Magnets):
    edges = np.logical_or(np.logical_or(mm.ixx == 0, mm.ixx == mm.nx - 1), np.logical_or(mm.iyy == 0, mm.iyy == mm.ny - 1))
    valid = np.where(np.logical_and(np.logical_not(edges), mm.ixx % 2 == mm.iyy % 2)) # All vertices and centers of squares
    mx = np.multiply(mm.m, mm.orientation[:,:,0])
    my = np.multiply(mm.m, mm.orientation[:,:,1])
    ## Count AFM vertices
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mx_avg = signal.convolve2d(mx, mask[::-1,::-1], mode="same", boundary="wrap" if mm.PBC else "fill")
    my_avg = signal.convolve2d(my, mask[::-1,::-1], mode="same", boundary="wrap" if mm.PBC else "fill")
    mx_valid = mx_avg[valid]
    my_valid = my_avg[valid]
    AFM = np.sum(np.logical_and(np.isclose(mx_valid, 0), np.isclose(my_valid, 0)))
    total = valid[0].size
    return AFM/total


def run(N: float = 10, size: int = 51, monopoles: list[float] = None, PBC: bool = False):
    """ Runs the simulation and saves it to a timestamped folder.
        `T_range` specifies the examined temperature range in multiples of T_c.
        At each step, `N` calls of `Magnets.update()` are performed.
        The returned N1, N2, N3 and N4 are vertex fractions.
    """
    size = int(size/2)*2 if PBC else int(size) # For PBC to fit nicely, an even `size` is needed
    moment = 4.5e-17 # Paper value
    E_B = hotspice.utils.eV_to_J(15)
    init_pattern = "random"
    if monopoles is None: monopoles = [0]
    monopoles = np.asarray(monopoles).astype(float)
    
    angles = np.array([0, 5, 30, 32, 34, 36, 38, 40, 42, 44, 45]) #  np.arange(46)
    d_paper = 170e-9
    AFM_fraction = np.zeros((monopoles.size, angles.size, N))
    
    ## Simulate
    for k, monopole in enumerate(monopoles):
        monopole_d = 220e-9*monopole
        if monopole: energyDD = hotspice.energies.DiMonopolarEnergy(d=monopole_d)
        else: energyDD = hotspice.energies.DipolarEnergy()
        for j in range(N):
            for i, angle in enumerate(angles):
                print(f"[sample {j}] {angle=}°, {monopole_d=}...")
                a = d_paper*2*np.cos(np.deg2rad(angle))
                mm = hotspice.ASI.IP_Square(a, size, PBC=PBC, moment=moment, T=300, 
                                            energies=[energyDD],E_B=E_B, m_perp_factor=0.4,
                                            angle=np.deg2rad(angle))
                mm.params.UPDATE_SCHEME = hotspice.Scheme.METROPOLIS
                mm.initialize_m(init_pattern)
                T0 = -np.min(mm.E)/hotspice.kB*0.3
                mm.T = T0
                while mm.T_avg > 300: # To approach the ground state by lowering the temperature gradually
                    mm.T *= 0.999
                    mm.update(Q=1)
                AFM_fraction[k,i,j] = count_AFM_cells(mm)
    
    ## Save
    hotspice.utils.save_results(parameters={"size": size, "N": N, "PBC": mm.PBC, "m_perp_factor": mm.m_perp_factor, "monopoles": monopoles, "a": a, "monopole_d": monopole_d, "moment": moment, "scheme": mm.params.UPDATE_SCHEME.name, "init_pattern": init_pattern},
                                data={"angles": angles, "AFM_fraction": AFM_fraction})
    plot()


def inset_ax(ax: plt.Axes, fig: plt.Figure = None, ASI_type: hotspice.ASI.IP_ASI = hotspice.ASI.IP_Square):
    if fig is None: fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height # Width and height of ax in inches
    inset_height = height # Height of inset ax in inches
    on_left = (ASI_type == hotspice.ASI.IP_Square)
    inset_ax: plt.Axes = inset_axes(ax, width=inset_height, height=inset_height, loc='upper left' if on_left else 'upper right',
                        bbox_to_anchor=(0, 1) if on_left else (1, 1), bbox_transform=ax.transAxes, borderpad=0)
    # inset_ax.set_aspect('equal', adjustable='datalim')
    inset_ax.axis('off')
    inset_ax.patch.set_alpha(0) # Transparent background
    
    def draw_ellipse(x, y, l, angle):
        inset_ax.add_artist(Ellipse((x, y), l, 4*l/11, angle=angle, color='gray', alpha=0.9))
    
    def draw_lattice(x, y, r, n=2, angle=0):
        for i in range(1-n, 2+n):
            for j in range(-n, 1+n):
                if i % 2 == j % 2:
                    draw_ellipse(x + (i-1)*r, y + j*r, l=1.4*r, angle=angle + (i % 2)*90)
    
    l, y, n = 0.04, 0.75, 2
    if on_left:
        draw_lattice(0.15, y, l, n=n, angle=0)
    else:
        draw_lattice(0.85, y, l, n=n, angle=45)


def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = epu.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    AFM_fraction = data["AFM_fraction"]
    monopoles = list(np.asarray(params['monopoles']).reshape(-1))
    
    ## We need two axes, because Matplotlib does not natively support broken axes.
    epu.init_style()
    min1, max1, min2, max2 = 0, 6.1, 28.5, 45
    ratio = (max2 - min2)/(max1 - min1)
    fig, axes = plt.subplots(nrows=1, ncols=2, width_ratios=(1,ratio), sharey=True, facecolor='w', squeeze=True, figsize=(epu.page_width/2*0.56, 2.1))
    (ax1, ax2) = axes
    for ax in (ax1, ax2):
        for k, monopole in enumerate(monopoles):
            label = "Dipole" if not monopole else ("Dumbbell" if np.isclose(monopole, 1) else f"Dumbbell $d={monopole:.1f}$")
            ax.errorbar(data['angles'], np.mean(AFM_fraction[k,:,:], axis=1), yerr=np.std(AFM_fraction[k,:,:], axis=1), fmt=f"{epu.marker_cycle[k]}-", label=label)
        ax.axhline(0.5, color="grey", linestyle=":")
        ax.set_yticks([0, 0.5, 1])
        ax.set_facecolor('none')
        ticksevery = 5
    ax1.set_xticks([i for i in np.arange(min1 - (min1 % ticksevery), max1+.1, ticksevery)])
    ax2.set_xticks([i for i in np.arange(min2 - (min2 % ticksevery), max2+.1, ticksevery)])
    
    ax2.axvline(np.rad2deg(np.arcsin(1/np.sqrt(3))), color='C0', linestyle=':')

    ax1.set_xlim([min1, max1])
    ax2.set_xlim([min2, max2])
    fig.supxlabel(r"Magnet rotation angle $\alpha$ [°]", fontsize=10, x=0.565, y=0.03)
    ax1.set_ylabel("AFM fraction", labelpad=-0.05)

    ## Create the look of broken axes.
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.tick_params(left=False)
    ax1.tick_params(right=False)

    d = .015 # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', lw=1, clip_on=False)
    ax1.plot((1-d*ratio, 1+d*ratio), (-d*2, +d*2), **kwargs)
    ax1.plot((1-d*ratio, 1+d*ratio), (1-d*2, 1+d*2), **kwargs)
    kwargs.update(transform=ax2.transAxes) # switch to the bottom axes
    ax2.plot((-d, +d), (1-d*2, 1+d*2), **kwargs)
    ax2.plot((-d, +d), (-d*2, +d*2), **kwargs)

    inset_ax(ax=ax1, fig=fig, ASI_type=hotspice.ASI.IP_Square)
    inset_ax(ax=ax2, fig=fig, ASI_type=hotspice.ASI.IP_Pinwheel)
    epu.label_ax(ax1, 1, offset=(-0.1/(0.56*0.75)*(1+ratio), 0.1))
    fig.legend(*ax.get_legend_handles_labels(), bbox_to_anchor=(0.18, 0.22), loc="lower left", borderaxespad=0.3, ncol=1, columnspacing=1, handletextpad=0.5, fontsize=9).set_zorder(-1)
    fig.subplots_adjust(top=0.85, bottom=0.22, left=0.18, right=0.95, wspace=0.05)
    hotspice.utils.save_results(figures={'Pinwheel_angle': fig}, outdir=data_dir, copy_script=False)


if __name__ == "__main__":
    """ EXPECTED RUNTIME: ≈20min for N=10. """
    run(monopoles=[0, 1], N=10)
    # epu.replot_all(plot)
