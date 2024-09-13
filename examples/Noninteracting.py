r""" This file tests if Hotspice can correctly reproduce the tanh
    (as a function of external field) when interactions are disabled.
"""
import matplotlib.pyplot as plt
import numpy as np

import hotspice
import _example_plot_utils as epu


def run(N: float = 10, size: int = 1000, monopoles: bool = False):
    """ Tests if the hyperbolic tangent is found when interactions are disabled.
        At each field magnitude, `N` Monte Carlo steps per spin are performed.
    """
    T = 300
    moment = 1.1e-15

    energyZ = hotspice.energies.ZeemanEnergy(magnitude=0, angle=np.pi/4) # 45° to equally affect all magnets
    mm = hotspice.ASI.IP_Square(1, size, moment=moment, energies=[energyZ], pattern="random", T=T, m_perp_factor=0)
    mm.params.UPDATE_SCHEME = hotspice.Scheme.METROPOLIS

    alpha = np.linspace(0, 3, 11)

    mm.initialize_m('uniform', angle=np.pi/4)
    M_0 = mm.m_avg
    M_x = np.zeros_like(alpha)
    B_vals = alpha*hotspice.kB*T/moment*np.sqrt(2) # sqrt(2) stronger field because 45° field angle
    for i, B in enumerate(B_vals):
        print(f"[{i+1}/{alpha.size}] alpha={alpha[i]:.1f}...")
        energyZ.magnitude = B
        mm.progress(MCsteps_max=N)
        M_x[i] = mm.m_avg/M_0
        
    ## Save
    hotspice.utils.save_results(parameters={"size": size, "N": N, "PBC": mm.PBC, "m_perp_factor": mm.m_perp_factor, "monopoles": monopoles, "T": T, "moment": moment, "scheme": mm.params.UPDATE_SCHEME.name},
                                data={"alpha": alpha, "M_avg": M_x})
    plot()


def plot(data_dir=None, use_inset: bool = True):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = epu.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
        
    ## Main axes
    epu.init_style()
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(epu.page_width/2*0.42, 2.1))
    ax: plt.Axes = axes[0,0]

    ax.scatter(data["alpha"], data["M_avg"], label="Hotspice", zorder=2)
    alpha = np.linspace(0, 3, 1000)
    ax.plot(alpha, np.tanh(alpha), label="Theory", color="black", zorder=1)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 3)
    ax.set_xlabel(r"$\mu B/k_BT$")
    ax.set_ylabel(r"$\langle M \rangle/M_0$", labelpad=-.05)
    ax.set_yticks([0, 0.5, 1])

    ## Finish plot
    epu.label_ax(ax, 0, offset=(-0.1/(0.42*0.7), 0.1))
    ax.legend(ncol=1, loc="lower right", fontsize=9, borderaxespad=0.3)
    fig.subplots_adjust(top=.85, bottom=0.22, left=0.25, right=0.95)
    hotspice.utils.save_results(figures={'Noninteracting': fig}, outdir=data_dir, copy_script=False)


if __name__ == "__main__":
    """ EXPECTED RUNTIME: <1min. """
    run()
    # epu.replot_all(plot)
