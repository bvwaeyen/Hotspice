import math
import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context import hotspin
from examples import examplefunctions as ef

kB = 1.380649e-23
J = kB*300
T_c = 2*J/kB/math.log(1+math.sqrt(2))
T_lim = [0.9, 1.1] # Relative to T_c
a = 1 # Lattice spacing, chosen large to get many simultaneous switches, it doesn't matter for exchange energy anyway
size = 800

def get_m_theory(T_min, T_max):
    m_theory_range = np.linspace(T_min, T_max, 1000)
    with np.errstate(invalid='ignore'):
        m_theory = (1-np.sinh(2*(1/2*kB*math.log(1+math.sqrt(2)))/kB/m_theory_range)**-4)**.125 # J. M. D. Coey, Magnetism and Magnetic Materials (6.33)
    m_theory[np.isnan(m_theory)] = 0
    return m_theory_range, m_theory


def test_magnetization(T_steps=21, N=1000, verbose=False, plot=True, reverse=False):
    ''' Performs a sweep of the temperature in <T_steps> steps. At each step, <N> Magnets.update() calls are performed.
        @param reverse [bool] (False): if True, the temperature steps are in descending order, otherwise ascending.
    '''
    mm = hotspin.ASI.FullASI(size, a, energies=[hotspin.ExchangeEnergy(J=J)], PBC=True, pattern=('random' if reverse else 'uniform'))

    T_range = np.linspace(T_lim[0], T_lim[1], T_steps)
    if reverse: T_range = np.flip(T_range)
    m_avg = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
        if verbose: print(f'[{i+1}/{T_steps}] N={N}, T = {T:.2f}*T_c (= {T*T_c:.0f} K)...')
        mm.T = T*T_c
        for _ in range(N): mm.update()
        m_avg[i] = abs(mm.m_avg)
    
    data = pd.DataFrame({"T": T_range, "m_avg": m_avg})
    if plot: test_magnetization_plot(data)
    return data

def test_magnetization_plot(data: pd.DataFrame, save=False):
    hotspin.plottools.init_fonts()
    plt.scatter(data['T'], data['m_avg'])
    plt.plot(*get_m_theory(T_lim[0]-.005, T_lim[1]+.005), color='black')
    plt.legend(['Hotspin', 'Theory'])
    plt.show()

def test_N_influence(*args, plot=True, save=True, **kwargs):
    ''' Tests the influence of <N> (the number of iterations per value of T) on m_avg.
        Any arguments (not 'N') passed to this function are passed through to test_magnetization().
    '''
    data = pd.DataFrame()
    
    N_range = [1000, 2000, 4000]
    for N in N_range:
        local_data = test_magnetization(*args, N=N, plot=False, **kwargs)
        local_data["N"] = N
        data = pd.concat([data, local_data])

    Tsweep_direction = 'reverse' if kwargs.get('reverse', False) else ''
    savename = f"results/test_squareIsing/J={J/kB:.0f}kB_Tsweep{data['T'].nunique()}{Tsweep_direction}_Nsweep{data['N'].nunique()}_a={a:.2g}_{size}x{size}"
    if plot: test_N_influence_plot(data, save=savename)
    if save: hotspin.plottools.save_data(data, f"{savename}.csv")
    
    return data

def test_N_influence_plot(data: pd.DataFrame, save=False):
    ''' If <save> is bool, the filename is automatically generated. If <save> is str, it is used as filename. '''
    hotspin.plottools.init_fonts()
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(111)
    
    T_lim = [data["T"].min(), data["T"].max()]
    ascending = data["T"].iloc[0] < data["T"].iloc[-1]

    for N, local_data in data.groupby("N"):
        ax.scatter(local_data["T"], local_data["m_avg"], label=f'N={N}')

    ax.plot(*get_m_theory(T_lim[0]-.005, T_lim[1]+.005), color='black', label='Theory')
    ax.legend()
    ax.set_xlabel('Temperature $T/T_c$')
    ax.set_ylabel('Magnetization $\\langle M\\rangle /M_0$')
    ax.set_xlim([T_lim[0]-.005, T_lim[1]+.005])
    ax.set_ylim([-0.01, 1])
    plt.gcf().tight_layout()
    if save:
        if type(save) is str:
            savename = os.path.splitext(save)[0] + '.pdf'
        else:
            savename = f"results/test_squareIsing/Tsweep{data['T'].nunique()}{'' if ascending else 'reverse'}_Nsweep{data['N'].nunique()}.pdf"
        hotspin.plottools.save_plot(f"{savename}.pdf")
    plt.show()


if __name__ == "__main__":
    test_N_influence(T_steps=1, verbose=True, save=True)
