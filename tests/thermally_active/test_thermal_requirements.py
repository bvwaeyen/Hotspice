"""
Created 07/05/2023
A program to determine if the following requirements for the system are actually met.
- The magnetic field alone should not do anything.
- Magnetic field + thermal energy should work: thermally assisted switching
- Edges and weak spots should switch first

TODO thermal and magnetic half-times should be the same.
"""
import mmap

import hotspice.plottools
from thermally_active_system import *  # mm is system
import matplotlib.pyplot as plt

MCsteps_max = 200

# ==================================================
# check average E_d after thermal relaxation

check_E_d = False
if check_E_d:
    print("Checking average E_d value after thermal relaxation")
    dt, dMCsteps = mm.progress(t_max=np.inf, MCsteps_max=MCsteps_max)
    print(f"progressed {dt}s within {dMCsteps}")
    hotspice.plottools.show_m(mm)  # check if okay

    E_d = mm.get_energy("dipolar").E
    E_d = hotspice.utils.J_to_eV(E_d) * 1000  # in meV

    E_avg = np.average(E_d[E_d != 0.0])
    print(f"{E_avg} meV")

# ==================================================
# check H does flips everything

check_H = False
if check_H:
    print("Checking effect of H field")
    mm.initialize_m("uniform")

    zeeman = mm.get_energy("zeeman")
    if zeeman is None:
        zeeman = hotspice.ZeemanEnergy(0, 0)
        mm.add_energy(zeeman)
    zeeman.set_field(magnitude=H, angle=np.pi)

    dt, dMCsteps = mm.progress(t_max=np.inf, MCsteps_max=MCsteps_max)
    print(f"progressed {dt}s within {dMCsteps}")
    hotspice.plottools.show_m(mm)  # check if okay


# ==================================================
# check if H does nothing without T

check_H_T = False
if check_H_T:
    print("Checking H field without T (should do nothing)")
    mm.initialize_m("uniform")

    zeeman = mm.get_energy("zeeman")
    if zeeman is None:
        zeeman = hotspice.ZeemanEnergy(0, 0)
        mm.add_energy(zeeman)
    zeeman.set_field(magnitude=H, angle=np.pi)

    mm.T = 1  # almost no T

    dt, dMCsteps = mm.progress(t_max=np.inf, MCsteps_max=MCsteps_max)
    print(f"progressed {dt}s within {dMCsteps}")
    hotspice.plottools.show_m(mm)  # check if okay
