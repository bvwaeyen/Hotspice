import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from context import hotspin
from examples import examplefunctions as ef

kB = 1.380649e-23
J = kB*300
T_c = 2*J/kB/math.log(1+math.sqrt(2))
T_lim = [0.9, 1.1] # Relative to T_c
a = 1 # Lattice spacing, chosen large to get many simultaneous switches, it doesn't matter for exchange energy anyway

def get_m_theory(T_min, T_max):
    m_theory_range = np.linspace(T_min, T_max, 1000)
    with np.errstate(invalid='ignore'):
        m_theory = (1-np.sinh(2*(1/2*kB*math.log(1+math.sqrt(2)))/kB/m_theory_range)**-4)**.125 # J. M. D. Coey, Magnetism and Magnetic Materials (6.33)
    m_theory[np.isnan(m_theory)] = 0
    return m_theory_range, m_theory


def test_magnetization(T_steps=30, N=1000, verbose=False, plot=True):
    mm = hotspin.ASI.FullASI(800, a, energies=[hotspin.ExchangeEnergy(J=J)], PBC=True, pattern='uniform')

    T_range = np.linspace(T_lim[0], T_lim[1], T_steps)
    m_avg = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
        if verbose: print(f'[{i+1}/{T_steps}] N={N}, T = {T:.2f}*T_c (= {T*T_c:.0f} K)...')
        mm.T = T*T_c
        for _ in range(N): mm.update()
        m_avg[i] = abs(mm.m_avg)

    if plot:
        hotspin.plottools.init_fonts()
        plt.scatter(T_range, m_avg)
        plt.plot(*get_m_theory(T_lim[0]-.005, T_lim[1]+.005), color='black')
        plt.legend(['Hotspin', 'Theory'])
        plt.show()
    return T_range, m_avg

def test_N_influence(*args, plot=True, save=False, **kwargs):
    ''' Tests the influence of <N> (the number of iterations per value of T) on m_avg.
        Any arguments (not 'N') passed to this function are passed through to test_magnetization().
    '''
    if plot:
        hotspin.plottools.init_fonts()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
    
    N_range = [1000, 2000, 4000]
    for N in N_range:
        T_range, m_avg = test_magnetization(*args, N=N, plot=False, **kwargs)
        if plot: ax.scatter(T_range, m_avg)

    if plot:
        ax.plot(*get_m_theory(T_lim[0]-.005, T_lim[1]+.005), color='black')
        ax.legend([f'N={N}' for N in N_range] + ['Theory'])
        ax.set_xlabel('Temperature $T/T_c$')
        ax.set_ylabel('Magnetization $\\langle M\\rangle /M_0$')
        ax.set_xlim([T_lim[0]-.005, T_lim[1]+.005])
        ax.set_ylim([-0.01, 1])
        plt.gcf().tight_layout()
        if save:
            hotspin.plottools.save_plot(f"results/test_squareIsing/J={J/kB:.0f}kB_Tsweep{len(T_range)}_Nsweep{len(N_range)}_a={a:.2g}.pdf")
        plt.show()


if __name__ == "__main__":
    test_N_influence(T_steps=21, verbose=True, save=True)
