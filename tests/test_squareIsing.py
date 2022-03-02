import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from context import hotspin
from examples import examplefunctions as ef

# The formulae in this file come from https://en.wikipedia.org/wiki/Square_lattice_Ising_model.

kB = 1.380649e-23
J = kB*300
T_c = 2*J/kB/math.log(1+math.sqrt(2))

def get_m_theory(T_range):
    m_theory_range = np.linspace(np.min(T_range), np.max(T_range), 1000)
    with np.errstate(invalid='ignore'):
        m_theory = (1-np.sinh(2*(T_c/2*kB*math.log(1+math.sqrt(2)))/kB/m_theory_range)**-4)**.125
    m_theory[np.isnan(m_theory)] = 0
    return m_theory_range, m_theory


def test_magnetization(T_steps=30, N=1000, verbose=False, plot=True):
    mm = hotspin.ASI.FullASI(800, 1, energies=[hotspin.ExchangeEnergy(J=J)], PBC=True) # 800, 1e-6, N=300, T_steps = 100 yields best result until now
    mm.initialize_m('uniform')
    # ef.animate_quenching(mm, T_low=690, T_high=690, n_sweep=4000, pattern='uniform')

    T_range = np.linspace(T_c*.9, T_c*1.1, T_steps)
    m_avg = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
        if verbose: print(f'({i+1}/{T_steps}) T={T:.0f} K...')
        mm.T = T
        for _ in range(N): mm.update()
        m_avg[i] = abs(mm.m_avg)
        # hotspin.plottools.show_m(mm)

    if plot:
        plt.scatter(T_range, m_avg)
        plt.plot(*get_m_theory(T_range), color='black')
        plt.legend(['Hotspin', 'Theory'])
        plt.show()
    return T_range, m_avg

def test_N_influence(*args, plot=True, **kwargs):
    ''' Tests the influence of <N> (the number of iterations per value of T) on m_avg.
        Any arguments (not 'N') passed to this function are passed through to test_magnetization().
    '''
    N_range = [1000, 2000, 4000]
    for N in N_range:
        T_range, m_avg = test_magnetization(*args, N=N, plot=False, **kwargs)
        if plot: plt.scatter(T_range, m_avg)

    if plot:
        plt.plot(*get_m_theory(T_range), color='black')
        plt.legend([f'Hotspin N={N}' for N in N_range] + ['Theory'])
        plt.show()

# TODO: does re-initializing at each T help?


if __name__ == "__main__":
    test_N_influence(T_steps=10, verbose=True)
