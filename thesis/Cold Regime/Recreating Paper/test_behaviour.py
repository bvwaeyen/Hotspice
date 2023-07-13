# Created 10/07/2023 to test the cold regime behaviour of pinwheel ASI. I expect nothing or avalanches.
# Is it okay to use Néel in very low temperatures? ... I should probably look at the official paper for that :(
# Should I test relax or minimize? I have bad experience with those

import hotspice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def mm_to_rgb(mm):
    avg_angle = np.angle(complex(mm.m_avg_x, mm.m_avg_y))
    hsv = [avg_angle % (2*np.pi) / (2*np.pi), 1., max(0., min(mm.m_avg * np.sqrt(2), 1.))]
    return hsv_to_rgb(hsv)


# PARAMETERS
n = 21  # 220 actual spins
T = 1  # almost deterministic
t_max = 1  # 1 second
update_scheme = "Néel"  # better for low temperatures
moment = 3e-16  # from Jonathan Maes, comes from ratio of flatspin alpha and hotspice a
E_B = hotspice.utils.eV_to_J(110)  # from Jonathan Maes
E_B = E_B * np.ones((n, n))*np.random.normal(1, 0.05, size=(n, n))  # for randomness  # TODO


angle = 7 * np.pi/180  # used to be almost symmetric
zeeman = hotspice.ZeemanEnergy(angle = np.pi + angle)  # backwards

# TODO: this is very detailed
a_min, a_max, da = 20, 800, 10  # nm
a_array = np.arange(a_min, a_max+da, da) * 1e-9  # 20nm to 1000nm, paper has ~215nm to 1000nm
B_min, B_max, dB = 60, 110, 0.5  # mT
B_array = np.arange(B_min, B_max+dB, dB) * 1e-3

rgb = np.zeros((a_array.size, B_array.size, 3), dtype=float)  # empty rbg grid

for a_index, a in enumerate(a_array):  # lattice parameters
    print(f"Computing a {a_index + 1}/{a_array.size}")
    # new a new ASI
    mm = hotspice.ASI.IP_Pinwheel(a, n, T=T, moment=moment, E_B=E_B)
    mm.params.UPDATE_SCHEME = update_scheme
    mm.add_energy(zeeman)

    for B_index, B in enumerate(B_array):  # magnetic field magnitude of input
        # new B new zeeman
        zeeman.set_field(magnitude=B)

        mm.initialize_m("uniform")  # reset
        mm.progress(t_max=t_max)  # progress

        rgb[a_index, B_index][:] = mm_to_rgb(mm)


matplotlib.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(5, 3.75))
extent = [B_min - 0.5*dB, B_max + 0.5*dB, a_max + 0.5*da, a_min - 0.5*da]  # with "upper" and units in mind
ax.imshow(rgb, origin="upper", extent=extent, aspect="auto")
ax.set_xlabel("B (mT)"); ax.set_ylabel("a (nm)")
plt.tight_layout()
plt.show()