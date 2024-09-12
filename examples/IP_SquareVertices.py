r""" This file tests if Hotspice can correctly reproduce the vertex count
    in IP square ASI as a function of magnetostatic interaction strength.
    Based on the paper
        Wang, R., Nisoli, C., Freitas, R. et al. Artificial 'spin ice' in a
        geometrically frustrated lattice of nanoscale ferromagnetic islands.
        Nature 439, 303-306 (2006). https://doi.org/10.1038/nature04447
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal

import hotspice
import _example_plot_utils as epu


def count_square_vertices(mm: hotspice.Magnets):
    vertices = np.where(np.logical_and(mm.ixx % 2 == 0, mm.iyy % 2 == 0))
    ## Count Type 3 and Type 4 vertices
    mask34 = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])
    convolution34 = signal.convolve2d(mm.m, mask34[::-1,::-1], mode="same", boundary="wrap")
    N_in = convolution34[vertices] # Counts the number of in-pointing magnets
    N3 = np.sum(np.abs(N_in) == 2)
    N4 = np.sum(np.abs(N_in) == 4)
    ## Count Type 1 and Type 2 vertices
    mask12 = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
    convolution12 = signal.convolve2d(mm.m, mask12[::-1,::-1], mode="same", boundary="wrap")
    N_flower = convolution12[vertices] # Is +4 or -4 at a flower vertex state (Type 1)
    N1 = np.sum(np.logical_and(np.abs(N_in) == 0, np.abs(N_flower) == 4))
    N2 = np.sum(np.logical_and(np.abs(N_in) == 0, np.abs(N_flower) != 4))
    total = N1 + N2 + N3 + N4
    assert total == vertices[0].size # Sanity check: do the masks work as expected?
    return N1/total, N2/total, N3/total, N4/total


def run(spacings=np.linspace(300, 800, 6), N: float = 10, size: int = 40, monopoles: bool = False):
    """ Runs the simulation and saves it to a timestamped folder.
        At each step, `N` Monte Carlo steps per spin are performed.
        The returned N1, N2, N3 and N4 are vertex fractions between 0 and 1.
    """
    spacings = np.asarray(spacings)
    size = int(size/2)*2 # For PBC to fit nicely, an even `size` is needed
    moment = 3e-17
    init_pattern = "random"

    N1 = np.zeros((spacings.size, N))
    N2, N3, N4 = np.zeros_like(N1), np.zeros_like(N1), np.zeros_like(N1)
    ## Simulate
    for j in range(N):
        energyDD = hotspice.energies.DiMonopolarEnergy(d=220e-9) if monopoles else hotspice.energies.DipolarEnergy()
        energyZ = hotspice.energies.ZeemanEnergy()
        mm = hotspice.ASI.IP_Square(1e-6, size, PBC=True, moment=moment, T=300, energies=[energyDD, energyZ],
                                        E_B=hotspice.utils.eV_to_J(10), m_perp_factor=0)
        mm.params.UPDATE_SCHEME = hotspice.Scheme.METROPOLIS
        for i, spacing in enumerate(spacings):
            mm.dx = spacing/2
            mm.dy = spacing/2
            print(f"[a={spacing*1e9:.0f}nm][sample {j}]...")
            mm.initialize_m(init_pattern)
            for factor in np.linspace(0.3, 0.2, 20): # To approach the ground state by lowering the temperature gradually
                mm.T = -np.min(mm.E)/hotspice.kB*factor
                if mm.T_avg < 300: break
                mm.progress(MCsteps_max=100, Q=np.inf)
            mm.T = 300
            mm.progress(MCsteps_max=100, Q=np.inf)
            N1[i,j], N2[i,j], N3[i,j], N4[i,j] = count_square_vertices(mm)

    ## Save
    hotspice.utils.save_results(parameters={"size": size, "N": N, "PBC": mm.PBC, "m_perp_factor": mm.m_perp_factor, "monopoles": monopoles, "moment": moment, "scheme": mm.params.UPDATE_SCHEME.name, "init_pattern": init_pattern},
                                data={"spacings": spacings, "N1": N1, "N2": N2, "N3": N3, "N4": N4})
    plot()


def inset_ax(ax: plt.Axes, fig: plt.Figure = None, w=0.6, on_left: bool = False, single_icon: bool = True):
    if fig is None: fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height # Width and height of ax in inches
    inset_width = w*width # Width of inset ax in inches
    inset_ax: plt.Axes = inset_axes(ax, width=inset_width, height=inset_width, loc='upper left' if on_left else 'upper right',
                        bbox_to_anchor=(0, 1) if on_left else (1, 1), bbox_transform=ax.transAxes, borderpad=0)
    inset_ax.axis('off')
    inset_ax.patch.set_alpha(0) # Transparent background
    
    def draw_arrow(x, y, dx, dy, color='k'):
        x1, x2, y1, y2 = x - dx/2, x + dx/2, y - dy/2, y + dy/2
        l = np.sqrt(dx*dx + dy*dy)
        fancyarrowkwargs = dict(posA=(x1, y1), posB=(x2, y2), color=color, lw=1, joinstyle='miter', linestyle='-', zorder=1, shrinkA=0.5, shrinkB=0)
        inset_ax.add_artist(FancyArrowPatch(**fancyarrowkwargs, arrowstyle='-|>', mutation_scale=120*l))
    
    def draw_state(x, y, r, state: tuple[bool, bool, bool, bool]=None, name=None, fontsize=None, color='k'):
        """ `state` is 0 if the magnet points towards the center. Order of magnets: (0°, 90°, 180°, 270°). """
        if state is not None:
            for a in range(4):
                l = 0.8
                dx, dy = r*np.cos(np.pi/2*a), r*np.sin(np.pi/2*a)
                x0, y0 = x + dx*(1-l/2), y + dy*(1-l/2)
                sign = 1 if state[a] else -1
                draw_arrow(x0, y0, dx*l*sign, dy*l*sign, color=color)
        
        if name is not None:
            fontsize = height*72*r*0.9 # Inches to points is factor 72
            inset_ax.text(x=x, y=y+1.1*r, s=name, color=color, ha="center", va="bottom", fontsize=fontsize)
    
    if single_icon:
        n = 4
        d = 1/(n+1) # Center-to-center distance between state plots
        r = d/2*0.7
        x = [(2*i+1)*d/2 if on_left else 1 - (2*(n-i)-1)*d/2 for i in range(n)]
        y = 1 - 2*r
        draw_state(x[0], y, r, (0, 1, 0, 1), name="Type 1", color='C0')
        draw_state(x[1], y, r, (1, 1, 0, 0), name="Type 2", color='C1')
        draw_state(x[2], y, r, (1, 0, 0, 0), name="Type 3", color='C2')
        draw_state(x[3], y, r, (0, 0, 0, 0), name="Type 4", color='C3')
    else:
        n = 11
        d = 1/(n+1)
        r = d/2*0.9
        x = [(2*i+1)*d/2 if on_left else 1 - (2*(n-i)-1)*d/2 for i in range(n)]
        y = 1 - 4*r
        draw_state(x[0], y, r, (0, 1, 0, 1), color='C0')
        draw_state(x[0], y-d, r, (1, 0, 1, 0), color='C0')
        draw_state(x[0], y, 2*r, None, color='C0', name="Type 1")
        
        draw_state(x[2], y, r, (1, 1, 0, 0), color='C1')
        draw_state(x[3], y, r, (0, 1, 1, 0), color='C1')
        draw_state(x[2], y-d, r, (0, 0, 1, 1), color='C1')
        draw_state(x[3], y-d, r, (1, 0, 0, 1), color='C1')
        draw_state(np.mean(x[2:4]), y, 2*r, None, color='C1', name="Type 2")
        
        draw_state(x[5], y, r, (1, 0, 0, 0), color='C2')
        draw_state(x[6], y, r, (0, 1, 0, 0), color='C2')
        draw_state(x[7], y, r, (0, 0, 1, 0), color='C2')
        draw_state(x[8], y, r, (0, 0, 0, 1), color='C2')
        draw_state(x[5], y-d, r, (0, 1, 1, 1), color='C2')
        draw_state(x[6], y-d, r, (1, 0, 1, 1), color='C2')
        draw_state(x[7], y-d, r, (1, 1, 0, 1), color='C2')
        draw_state(x[8], y-d, r, (1, 1, 1, 0), color='C2')
        draw_state(np.mean(x[5:9]), y, 2*r, None, color='C2', name="Type 3")
        
        draw_state(x[10], y, r, (0, 0, 0, 0), color='C3')
        draw_state(x[10], y-d, r, (1, 1, 1, 1), color='C3')
        draw_state(x[10], y, 2*r, None, color='C3', name="Type 4")


def plot(data_dir=None, use_inset: bool = True):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = epu.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)

    ## Main axes
    epu.init_style()
    OOP_Exchange_figheight = 2.5 #! KEEP THIS SYNCED WITH OOP_Exchange.py!
    OOP_Dipolar_figheight = 2.1 #! KEEP THIS SYNCED WITH OOP_Dipolar.py!
    IP_SquareVertices_figheight = 2.3 if use_inset else (OOP_Exchange_figheight + OOP_Dipolar_figheight)
    fig, axes = plt.subplots(nrows=2-use_inset, ncols=1, squeeze=False, figsize=(epu.page_width/4*(1+use_inset), IP_SquareVertices_figheight))
    ax: plt.Axes = axes[-1,0]
    limit_frac = [1/8, 1/4, 1/2, 1/8]
    for i in range(4):
        ax.errorbar(data['spacings']*1e9, np.mean(data[f"N{i+1}"], axis=1), yerr=np.std(data[f"N{i+1}"], axis=1), fmt=f"{epu.marker_cycle[i]}-", label=f"Type {i+1}")
        # Draw horizontal lines indicating limit
        n_prev = limit_frac[:i].count(limit_frac[i])
        n_occur = limit_frac.count(limit_frac[i])
        spacing = 2
        ax.axhline(limit_frac[i], linestyle=(n_prev*(1+spacing), (1, spacing + (n_occur-1)*(1+spacing))), color=f"C{i}")
    ax.set_yticks([0, 1/8, 1/4, 1/2], labels=["0", "1/8", "1/4", "1/2"])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Lattice spacing $a$ [nm]")
    ax.set_ylabel("Vertex fraction")

    ## Inset axes
    if use_inset:
        axin: plt.Axes = ax.inset_axes(bounds=[axin_x := 0.55, axin_y := 0.56, 1 - axin_x - 0.02, 1 - axin_y - 0.08])
        lw = 1
        for i in range(1, 5):
            axin.errorbar(data['spacings']*1e9, np.mean(data[f"N{i}"], axis=1) - limit_frac[i-1], yerr=np.std(data[f"N{i}"], axis=1), markersize=3, lw=lw, fmt=f"{epu.marker_cycle[i-1]}-", label=f"Type {i}")
        axin.set_title("Excess vertex fraction", fontsize=6, fontdict=dict(weight='bold'), pad=-5)
        axin.axhline(0, color='k', linestyle=':', lw=lw)
        axin.tick_params(labelsize=8, size=2)
    else:
        axin = axes[0,0]
        for i in range(1, 5):
            axin.errorbar(data['spacings']*1e9, np.mean(data[f"N{i}"], axis=1) - limit_frac[i-1], yerr=np.std(data[f"N{i}"], axis=1), fmt=f"{epu.marker_cycle[i-1]}-", label=f"Type {i}")
        axin.set_ylabel("Excess\nvertex fraction", labelpad=-10)
        axin.axhline(0, color='k', linestyle=':')
        epu.label_ax(axin, 2, offset=(-0.32, 0))
    axin.tick_params('x', labelbottom=False)
    axin.set_yticks([-0.5, 0, 0.5], labels=["-1/2", "0", "1/2"])
    axin.set_ylim([-.55, .55])

    ## Finish plot
    epu.label_ax(ax, 2 if use_inset else 3, offset=(-0.2 if use_inset else -0.32, 0))
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4 if use_inset else 2, columnspacing=1.5, handletextpad=0.5)
    ratio = OOP_Exchange_figheight/IP_SquareVertices_figheight
    if use_inset:
        fig.subplots_adjust(top=0.75*ratio, bottom=0.2*ratio, left=0.17, right=0.95)
    else:
        ax.set_xticks([300, 500, 700, 900])
        fig.subplots_adjust(top=1 - (1-0.75)*ratio, bottom=0.2*ratio, left=0.23, right=0.95)
    ticks = ax.get_xticks()
    axin.set_xticks(ticks[np.where((ticks > axin.get_xlim()[0]) & (ticks < axin.get_xlim()[1]))])
    inset_ax(ax, fig=fig, w=0.5 if use_inset else 1, on_left=use_inset, single_icon=True)
    hotspice.utils.save_results(figures={'IP_SquarePinwheel': fig}, outdir=data_dir, copy_script=False)


if __name__ == "__main__":
    """ EXPECTED RUNTIME: ≈10min for N=10. """
    run(spacings=1e-9*np.array([320, 360, 400, 440, 480, 560, 680, 880]), N=10, monopoles=False)
    # epu.replot_all(plot)
