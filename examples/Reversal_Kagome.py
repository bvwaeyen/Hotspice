""" This file tests the correspondence between experiment and simulation for a
    kagome artifical spin ice, which is initialized in a uniform state and
    reversed using an external field.
    Based on the paper
        Mengotti, E., Heyderman, L. J., Rodríguez, A. F. et al. Real-space
        observation of emergent magnetic monopoles and associated Dirac strings
        in artificial kagome spin ice. Nature Physics, 7(1), 68-74, 2011.
"""

import matplotlib.pyplot as plt
import numpy as np

import hotspice
import _example_plot_utils as epu
from hotspice.utils import asnumpy


def run(size: int = 30, angle: float = np.pi/2 - 3.6*np.pi/180,
        E_B: float = hotspice.utils.eV_to_J(120), E_B_std: float = 0.05):
    a = 1e-6 # Gives 500nm NN center-to-center distance
    d = 470e-9  # Must be <577.35e-9, otherwise neighbors touch
    moment = 1.1e-15
    
    _N = 2000 # Number of `mm.update()` steps between `B_max` and `-B_max`.
    B_max = 0.06 # [T] The hysteresis occurs between `-B_max` -> `B_max` -> `-B_max`.
    B_fields = np.linspace(B_max, -B_max, _N) # One sweep to opposite fields (N steps)
    B_fields = np.append(B_fields, np.flip(B_fields)) # Sweep both ways (2*N steps)

    fields = [-0.85, -0.92, -0.99, -1.06] # As fraction of the coercive field
    states_monopole, states_dipole = [], []
    for monopoles in [False, True]:
        if monopoles: energyDD = hotspice.energies.DiMonopolarEnergy(d=d)
        else: energyDD = hotspice.energies.DipolarEnergy()
        energyZ = hotspice.ZeemanEnergy(angle=angle)
        mm = hotspice.ASI.IP_Kagome(a, size, moment=moment, energies=[energyDD, energyZ],
                                    E_B=E_B, E_B_std=E_B_std, m_perp_factor=0)
        mm.params.UPDATE_SCHEME = hotspice.Scheme.NEEL
        
        # Perform the reversal
        mm.initialize_m("uniform", angle=angle)
        M_S = mm.m_avg_y
        M_y = np.zeros_like(B_fields)
        for i, B in enumerate(B_fields):
            energyZ.magnitude = B
            mm.progress()
            M_y[i] = mm.m_avg_y/M_S

        # Identify the critical field magnitude
        for i, _ in enumerate(M_y): # Identify the critical field magnitude
            if (M_y[i] <= 0 < M_y[i + 1]) or (M_y[i] >= 0 > M_y[i + 1]):
                B_C = np.abs(B_fields[i] - (M_y[i]*(B_fields[i+1] - B_fields[i]))
                            /(M_y[i+1] - M_y[i]))
                break

        # Save states at certain field magnitudes
        for B in fields:
            energyZ.magnitude = B*B_C
            mm.progress()
            (states_monopole if monopoles else states_dipole).append(mm.m.copy())
    
    ## Save
    real_outdir = hotspice.utils.save_results(parameters={"size": size, "angle": angle, "a": a, "E_B": E_B, "E_B_std": E_B_std, "PBC": mm.PBC, "scheme": mm.params.UPDATE_SCHEME},
                                data={"fields": fields, "states_monopole": states_monopole, "states_dipole": states_dipole, "mm": mm}) # Note: mm is meant to get the orientations etc. from, so is same for monopole and dipole.
    print(f"Saved results to {real_outdir}")
    plot()


def vertex(mm: hotspice.Magnets, skip, m=None):
    if m is None: m = mm.m
    initial, type_I, type_II, inside, wrong, final = 0, 0, 0, 0, 0, 0
    red_x, red_y, blue_x, blue_y = [], [], [], []
    for y in range(len(m)):
        if skip <= y <= len(m)-skip-1:
            for x in range(len(m[y])):
                if np.abs(m[y,x]) == 1 and mm.angles[y,x] == np.pi/2 and skip <= x <= len(m[y])-skip-1:
                    self, up_left, up_right, low_left, low_right = m[y,x]*np.sign(mm.angles[y,x]), m[y-1,x-1]*np.sign(mm.angles[y-1,x-1]), m[y-1,x+1]*np.sign(mm.angles[y-1,x+1]), m[y+1,x-1]*np.sign(mm.angles[y+1,x-1]), m[y+1,x+1]*np.sign(mm.angles[y+1,x+1])
                    if (self == 1 and up_left == 1 and up_right == 1):
                        initial += 1
                    elif (self == 1 and up_left == -1 and up_right == 1) or (self == 1 and up_left == 1 and up_right == -1):
                        type_I += 1
                        red_x.append(mm.xx[y,x]*10**6)
                        red_y.append((mm.yy[y,x] - 288.6751346e-9)*10**6)
                    elif (self == -1 and up_left == 1 and up_right == 1):
                        type_II += 1
                        blue_x.append(mm.xx[y,x]*10**6)
                        blue_y.append((mm.yy[y,x] - 288.6751346e-9)*10**6)
                    elif (self == -1 and up_left == -1 and up_right == 1) or (self == -1 and up_left == 1 and up_right == -1):
                        inside += 1
                    elif (self == 1 and up_left == -1 and up_right == -1):
                        wrong += 1
                    else:
                        final += 1

                    if (self == 1 and low_left == 1 and low_right == 1):
                        initial += 1
                    elif (self == 1 and low_left == -1 and low_right == 1) or (self == 1 and low_left == 1 and low_right == -1):
                        type_I += 1
                        blue_x.append(mm.xx[y,x]*10**6)
                        blue_y.append((mm.yy[y,x] + 288.6751346e-9)*10**6)
                    elif (self == -1 and low_left == 1 and low_right == 1):
                        type_II += 1
                        red_x.append(mm.xx[y,x]*10**6)
                        red_y.append((mm.yy[y,x] + 288.6751346e-9)*10**6)
                    elif (self == -1 and low_left == -1 and low_right == 1) or (self == -1 and low_left == 1 and low_right == -1):
                        inside += 1
                    elif (self == 1 and low_left == -1 and low_right == -1):
                        wrong += 1
                    else:
                        final += 1
    total = initial + type_I + type_II + inside + wrong + final
    return initial/total*100, type_I/total*100, type_II/total*100, inside/total*100, wrong/total*100, final/total*100, red_x, red_y, blue_x, blue_y, total


def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = epu.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    mm: hotspice.Magnets = data['mm']
    
    ## Plot
    epu.init_style()
    fig = plt.figure(figsize=(6, 3))
    for row, monopole in enumerate([False, True]):
        states = data['states_monopole'] if monopole else data['states_dipole']
        for i, B in enumerate(data['fields']):
            m = states[i]
            ax = fig.add_subplot(2, 4, 1 + i + 4*monopole, facecolor='gray', aspect='equal')
            
            # Make axes invisible
            ax.set_xticks([])
            ax.set_yticks([])
            plt.setp(ax.spines.values(), visible=False)
            
            # Populate axes
            if row == 0: ax.set_title(f"$B={abs(B):.2f} B_C$", fontsize=10)
            if i == 0:
                ax.set_ylabel("Dumbbell" if monopole else "Dipole")
            unit = 'µ'
            unit_factor = hotspice.utils.SIprefix_to_mul(unit)
            nonzero = m.nonzero()
            m_old = states[0]
            mx, my = asnumpy(np.multiply(m, mm.orientation[:, :, 0])[nonzero]), asnumpy(np.multiply(m, mm.orientation[:, :, 1])[nonzero])
            ax.quiver(asnumpy(mm.xx[nonzero]) / unit_factor, asnumpy(mm.yy[nonzero]) / unit_factor, mx / unit_factor, my / unit_factor,
                    color=np.where(m[nonzero] == m_old[nonzero], "black", "white"),
                    pivot='mid', scale=1.1 / mm._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7,
                    units='xy')  # units='xy' makes arrows scale correctly when zooming
            initial, type_I, type_II, inside, wrong, final, red_x, red_y, blue_x, blue_y, _ = vertex(mm, 1, m=m)
            ax.scatter(red_x, red_y, color="red", s=5)
            ax.scatter(blue_x, blue_y, color="blue", s=5)
    fig.tight_layout()
    
    real_outdir = hotspice.utils.save_results(figures={'Reversal_Kagome': fig}, outdir=data_dir, copy_script=False)
    print(f"Saved plot to {real_outdir}")


if __name__ == "__main__":
    """ EXPECTED RUNTIME: ≈30s. """
    run()
    # epu.replot_all(plot)
