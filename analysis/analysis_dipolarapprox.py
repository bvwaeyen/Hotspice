import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['HOTSPICE_USE_GPU'] = 'False'

from context import hotspice
xp = hotspice.xp


def analysis_dipolarapprox_OOP(a: float = 200e-9, d: float = 170e-9):
    ''' `a` is the center-to-center distance between magnets,
        `d` is the diameter of the magnets.
    '''
    mm = hotspice.ASI.OOP_Square(a, 30, major_axis=d)
    dipolar: hotspice.energies.DipolarEnergy = mm.get_energy('dipolar')
    rrx = xp.maximum(mm.xx - mm.xx[0,0], 0) # Taken from `DipolarEnergy.initialize()`
    rry = xp.maximum(mm.yy - mm.yy[0,0], 0)
    rr = hotspice.utils.mirror4(xp.sqrt(rrx**2 + rry**2))
    unique_distances = xp.unique(rr)[1:]
    interactions = xp.zeros_like(unique_distances)
    for i, distance in enumerate(unique_distances): # Can only do this check for OOP due to the symmetry
        kernelvalues = dipolar.kernel_unitcell[0][xp.where(rr == distance)]
        assert xp.all(xp.isclose(kernelvalues, kernelvalues[0])), "OOP kernel values are non-unique for given distance."
        interactions[i] = kernelvalues[0] # All kernelvalues are the same anyway
    
    return unique_distances, interactions

def plot_diff_OOP():
    for i, magnet_size in enumerate([0, 90e-9, 180e-9, 200e-9]):
        unique_distances, interactions = analysis_dipolarapprox_OOP(a=(a:=200e-9), d=magnet_size)
        plt.scatter(unique_distances, interactions, marker="o", color=f"C{i}", label=f"{magnet_size*1e9:.0f} nm")
        plt.axvline(magnet_size, color=f"C{i}")
        # Plot the expected 1/r³ behavior with correction term
        r = np.linspace(np.min(unique_distances), np.max(unique_distances), 1000)
        rinv5_expectation = 1/r**3 + 9/16*magnet_size*magnet_size/r**5
        plt.plot(r, rinv5_expectation/rinv5_expectation[-1]*interactions[-1], color=f"C{i}", linestyle=':', label=f"Theory for {magnet_size*1e9:.0f} nm")

    plt.plot(r, r[-1]**3/r**3*interactions[-1], color='k', linestyle=':', label="Theory $1/r^3$")
    plt.xlim([0, 4*a])
    plt.ylim(bottom=0)
    plt.xlabel("Center-to-center distance $a$")
    plt.ylabel("Relative interaction")
    plt.legend(title="Magnet size")
    plt.show()


if __name__ == "__main__":
    plot_diff_OOP()