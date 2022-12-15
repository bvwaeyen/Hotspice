import math

import matplotlib.pyplot as plt
import numpy as np


def E(E_B, ext_field, angle=0):
    """ Adds an external field (period 2π) to the energy barrier (period π).
        If angle=0, ext_field is equal to Delta_E (energy diff. between m angle at 0 and at π).
    """
    return lambda theta: -E_B*(np.cos(2*theta)-1)/2 - ext_field*(np.cos(theta - angle)-1)/2


def plot(E_B, ext_field, angle, verbose=True):
    if ext_field < 0: ext_field, angle = -ext_field, angle + np.pi
    angle = angle % math.tau

    E_func = E(E_B, ext_field, angle)
    x = np.linspace(0, math.tau, 1024+1)
    y = E_func(x)
    maxima = np.where((y > np.roll(y, 1)) & (y > np.roll(y, -1)))
    minima = np.where((y < np.roll(y, 1)) & (y < np.roll(y, -1)))

    p,  = plt.plot(x, y, label=f"{E_B} + {ext_field} ({angle*180/np.pi:.0f}°)")
    plt.scatter(np.concatenate([x[maxima], x[minima]]), np.concatenate([y[maxima], y[minima]]), color=p.get_color())
    plt.axvline(angle, linewidth=1, color=p.get_color())

    if verbose:
        print(f"For external field at {angle*180/np.pi:.0f}°:")
        # The most accurate measure: using the analytical minima and maxima to calculate the effective energy barrier
        print(u"∞" + f"-state E_b = {np.min(y[maxima]) - np.max(y[minima]):.2e} J")
        # An accurate discrete measure: assume that maxima and minima occur at 0°, 90°, 180°, 270° exactly
        print(f"4-state E_b = {min(E_func(np.pi/2), E_func(-np.pi/2)) - max(E_func(0), E_func(np.pi)):.2e} J") 
        # Inaccurate, but the best if one only has access to the two stable states: E_b = E_B + (E(0°)-E(180°))/2
        print(f"2-state E_b = {max((delta_E := -abs(E_func(0) - E_func(np.pi))), E_B + delta_E/2):.2e} J")
    return E_func


def show():
    plt.xlim([0, np.pi*2])
    plt.ylim(0, plt.ylim()[1])
    # plt.xticks([i*np.pi/2 for i in range(5)], ["0", "π/2", "π", "3π/2", "2π"])
    plt.xticks([i*np.pi/2 for i in range(5)], ["0°", "90°", "180°", "270°", "360°"])
    for x in [i*np.pi/2 for i in range(5)]: plt.axvline(x, color='#CCC', linestyle=':')
    plt.xlabel("Magnetization angle [°]")
    plt.ylabel("Energy [a.u.]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example for pinwheel ASI sublattices in a global 30° external field (cfr. pinwheel reversal test)
    E_B, ext_field, ext_angle = 1, -1, -np.pi/6
    E1 = plot(E_B, ext_field, -np.pi/4 + ext_angle)
    E2 = plot(E_B, ext_field, np.pi/4 + ext_angle)

    show()