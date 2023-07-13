import hotspice
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
a = 124e-9  # 124 nm  seems very close together
a *= 2  # otherwise too bound
# n = 101  # 5100 actual spins
n = 50
V = 4.4e-22  # 220nm length; 80nm width; 25nm thickness;  Gets some randomness later on
# moment = 3e-16  # from Jonathan Maes, comes from ratio of flatspin alpha and hotspice a
# Using V and the default Msat, I get 3.52e-16, so slightly more
T = 1  # deterministic
E_B = hotspice.utils.eV_to_J(110)  # from Jonathan Maes from other paper
PBC = True  # I was unable to get clocking results with a boundary

# randomness
randomness = 0.10  # seems to need more than 5% randomness
random_distr = np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))
E_B = E_B * random_distr  # in E_B
# V = V * random_distr  # what about volume randomness? that could be more physical after all?

mm = hotspice.ASI.IP_Pinwheel(a, n, V=V, E_B=E_B, T=T, pattern="uniform", PBC=PBC)
mm.params.UPDATE_SCHEME = "Néel"  # better at low temperatures and high E_B; should maybe even use relax() ?

# middle island
# mm.m[int(n/2 - n/8) : int(n/2 + n/8 + 1), int(n/2 - n/8) : int(n/2 + n/8 + 1)] *= -1

# flip bottom half
# mm.m[0:n//2, :] = -1 * mm.m[0:n//2, :]

# flip left half
# mm.m[:, 0:n//2] = -1 * mm.m[:, 0:n//2]

# flip diagonal half of it
# UT = np.triu(mm.m, 1)  # without diagonal
# LT = np.tril(mm.m)  # with diagonal
# mm.m = (UT - LT)


H = 56e-3  # TODO: figure out magnetic field strength
angle = (45/2 + 180) * np.pi/180  # clocking angles at 22.5°, -22.5° and flipped 180°
zeeman = hotspice.ZeemanEnergy(magnitude=H, angle=angle)
mm.add_energy(zeeman)


bits = 5 * [1] + 5 * [0]

def bit_to_angles(bit):
    if bit:  # 1
        return (np.pi/8, - np.pi/8)
    return (9/8 * np.pi, 7/8 * np.pi)  # 0

angles = [bit_to_angles(bit) for bit in bits]

fig, axs = plt.subplots(2, len(bits)//2)
flat_axs = axs.flatten()
avg = hotspice.plottools.Average.resolve(True, mm)  # average for rgb

for i in range(len(bits)):
    zeeman.set_field(magnitude=H, angle=angles[i][0])
    mm.progress(t_max=1.)
    zeeman.set_field(magnitude=H, angle=angles[i][1])
    mm.progress(t_max=1.)

    print(f"Inserted bit {bits[i]} with angles {angles[i]}")
    rgb = hotspice.plottools.get_rgb(mm, avg=avg, fill=True)
    flat_axs[i].imshow(rgb)
    flat_axs[i].set_title("".join([str(bit) for bit in bits[:i+1]]))
    flat_axs[i].axis("off")

plt.show()

# realtime = hotspice.plottools.RealTime(mm, E_B_range=None, T_range=None, moment_range=None, H_range=(50e-3, 60e-3))
# realtime.run()