import matplotlib.pyplot as plt
import numpy as np


def E(E_B, ext_field, angle):
    """ Adds an external field (period 2π) to the energy barrier (period π). """
    return lambda x: -E_B*(np.cos(2*x)-1)/2 - ext_field*(np.cos(x - angle)-1)/2


def plot(E_B, ext_field, angle):
    if ext_field < 0: ext_field, angle = -ext_field, angle + np.pi
    angle = angle % (2*np.pi)

    x = np.linspace(0, 2*np.pi, 200)
    y = E(E_B, ext_field, angle)(x)
    maxima = np.where((y > np.roll(y, 1)) & (y > np.roll(y, -1)))
    minima = np.where((y < np.roll(y, 1)) & (y < np.roll(y, -1)))

    p,  = plt.plot(x, y, label=f"{E_B} + {ext_field} ({angle*180/np.pi:.0f}°)")
    plt.scatter(np.concatenate([x[maxima], x[minima]]), np.concatenate([y[maxima], y[minima]]), color=p.get_color())
    plt.axvline(angle, linewidth=1, color=p.get_color())


def show():
    plt.xlim([0, np.pi*2])
    plt.ylim(0, plt.ylim()[1])
    # plt.xticks([i*np.pi/2 for i in range(5)], ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.xticks([i*np.pi/2 for i in range(5)], ['0°', '90°', '180°', '270°', '360°'])
    for x in [i*np.pi/2 for i in range(5)]: plt.axvline(x, color='#CCC', linestyle=':')
    plt.xlabel("Magnetization angle [°]")
    plt.ylabel("Energy [a.u.]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example for pinwheel ASI sublattices in a global 30° external field (cfr. pinwheel reversal test)
    plot(1, -1, -np.pi/2 + np.pi/12)
    plot(1, -1, np.pi/12)
    show()