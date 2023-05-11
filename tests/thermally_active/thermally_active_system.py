"""
Created 07/05/2023
Use 'from thermally_active_system import *'
A set of parameters to yield a good thermally active system.
<E_B> = 25 meV, but a 5% randomness is added as <random_distr> to make <E_B_distr>. Reroll randomness for different samples!
Only domain walls should be switchable, not bulk.
Only H should not be enough, T should help with overcoming the barrier: thermally assisted switching.
E_H < E_B
E_H + E_T > E_B ish
E_H + E_T < E_B + E_d

E_H = E_T = E_d = 25 meV and E_B = 2 * 25 meV = 50 meV
probably fulfills these requirements
"""

import hotspice
import numpy as np

# PARAMETERS
T = 300  # about room temperature;  this yields ~25.85 eV
E_B = hotspice.utils.eV_to_J(25e-3)  # same as E_T
moment = hotspice.utils.eV_to_J(2.5)  # 400 times smaller than usual ; to get zeeman energy of order 2.5meV when H = 1 mT

# dipolar energy scales with 1e-7 * moment²/a³ (with extra factor for summation over all spins)
a = 22e-9  # to get 25eV dipolar energy AFTER THERMAL RELAXATION.
# Does not matter too much. As long as E_d > "0" every requirement is (probably) met.

n = 51  # probably need large systems for thermally active regime? odd number is more symmetric, but not PBC friendly
update_scheme = "Néel"  # Need to do stuff with time scales
PBC = False  # works and is more physical
pattern = "uniform"

# Random distribution still recommended. Reroll for new samples!
randomness = 0.05
random_distr = np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))
E_B_distr = E_B * random_distr

mm = hotspice.ASI.IP_Pinwheel(a, n, E_B=E_B_distr, T=T, moment=moment, pattern=pattern, PBC=PBC)
mm.params.UPDATE_SCHEME = update_scheme

H = 1e-3  # suggested magnetic field
dt = 2e-9  # around 2 ns is thermal half life and close to magnetic half life
input_angle = 7. * np.pi/180.  # angle is probably unnecessary, but can't hurt? Normally used to break symmetry ties


if __name__ == "__main__":
    dt, dMCsteps = mm.progress(t_max=15e-9, MCsteps_max=np.inf)
    print(f"Progressing {dt * 1e9} ns took {dMCsteps} MCsteps")
    print(f"Magnetization is {mm.m_avg_x}")
    hotspice.plottools.show_m(mm)
