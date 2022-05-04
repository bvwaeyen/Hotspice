''' This file tests the correspondence between theory and simulation for a
    two-dimensional square Ising model, with only exchange interactions,
    by observing the average magnetization as a function of temperature.
'''

import math

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context import hotspin


class test_squareIsing:
    def __init__(self, **kwargs):
        self.J = hotspin.kB*300 # This does not matter much, just dont take it too large to prevent float errors
        self.T_lim = [0.9, 1.1] # Relative to T_c
        self.a = 1 # Lattice spacing, chosen large to get many simultaneous switches, it doesn't matter for exchange energy anyway
        self.size = kwargs.get('size', 800) # Large to get the most statistically ok behavior

    def test(self, *args, **kwargs):
        self.data = self.test_N_influence(*args, **kwargs)

    @property
    def T_c(self):
        return 2*self.J/hotspin.kB/math.log(1+math.sqrt(2))

    def test_magnetization(self, T_steps=21, N=1000, verbose=False, plot=True, reverse=False):
        ''' Performs a sweep of the temperature in <T_steps> steps. At each step, <N> Magnets.update() calls are performed.
            The final half of these <N> update calls are recorded, from which and their average/stdev of m_avg calculated.
            @param reverse [bool] (False): if True, the temperature steps are in descending order, otherwise ascending.
        '''
        self.mm = hotspin.ASI.FullASI(self.size, self.a, energies=[hotspin.ExchangeEnergy(J=self.J)], PBC=True, pattern=('random' if reverse else 'uniform'))

        T_range = np.linspace(self.T_lim[0], self.T_lim[1], T_steps)
        if reverse: T_range = np.flip(T_range)
        m_avg = np.zeros_like(T_range)
        m_std = np.zeros_like(T_range)
        for i, T in enumerate(T_range):
            if verbose: print(f"[{i+1}/{T_steps}] N={N}, T = {T:.2f}*T_c (= {T*self.T_c:.0f} K)...")
            self.mm.T = T*self.T_c
            averages = []
            for n in range(N):
                self.mm.update()
                if n > N/2: averages.append(abs(self.mm.m_avg))
            m_avg[i] = np.mean(averages)
            m_std[i] = np.std(averages)

        data = pd.DataFrame({"T": T_range, "m_avg": m_avg, "m_std": m_std})
        if plot: test_squareIsing.test_magnetization_plot(data)
        return data
    
    def test_N_influence(self, *args, plot=True, save=True, **kwargs):
        ''' Tests the influence of <N> (the number of iterations per value of T) on m_avg.
            Any arguments (not 'N') passed to this function are passed through to test_magnetization().
        '''
        data = pd.DataFrame()

        N_range = [1000, 2000, 4000]
        for N in N_range:
            local_data = self.test_magnetization(*args, N=N, plot=False, **kwargs)
            local_data["N"] = N
            data = pd.concat([data, local_data])

        Tsweep_direction = 'reverse' if kwargs.get('reverse', False) else ''
        savename = f"results/test_squareIsing/Tsweep{data['T'].nunique()}{Tsweep_direction}_Nsweep{data['N'].nunique()}_{self.mm.nx}x{self.mm.ny}"
        if save: hotspin.utils.save_data(data, f"{savename}.csv")
        if plot: test_squareIsing.test_N_influence_plot(data, save=save*savename)

        return data

    @staticmethod
    def test_magnetization_plot(data: pd.DataFrame):
        T_lim = [data["T"].min(), data["T"].max()]

        hotspin.plottools.init_fonts()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        ax.errorbar(data["T"], data["m_avg"], yerr=data["m_std"], fmt='o', label='Hotspin')
        ax.plot(*test_squareIsing.get_m_theory(T_lim[0]-.005, T_lim[1]+.005), color='black', label='Theory')
        ax.legend()
        plt.gcf().tight_layout()
        plt.show()

    @staticmethod
    def test_N_influence_plot(data: pd.DataFrame, save=False):
        ''' If <save> is bool, the filename is automatically generated. If <save> is str, it is used as filename. '''
        T_lim = [data["T"].min(), data["T"].max()]

        hotspin.plottools.init_fonts()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        for N, local_data in data.groupby("N"):
            ax.errorbar(local_data["T"], local_data["m_avg"], yerr=local_data["m_std"], fmt='o', label=f'N={N}')
        ax.plot(*test_squareIsing.get_m_theory(T_lim[0]-.005, T_lim[1]+.005), color='black', label='Theory')
        ax.legend()
        ax.set_xlabel('Temperature $T/T_c$')
        ax.set_ylabel('Magnetization $\\langle M\\rangle /M_0$')
        ax.set_xlim([T_lim[0]-.005, T_lim[1]+.005])
        ax.set_ylim([-0.01, 1])
        plt.gcf().tight_layout()
        if save:
            if not isinstance(save, str):
                reverse = '' if data["T"].iloc[0] < data["T"].iloc[-1] else 'reverse'
                save = f"results/test_squareIsing/Tsweep{data['T'].nunique()}{reverse}_Nsweep{data['N'].nunique()}.pdf"
            hotspin.plottools.save_plot(save, ext='.pdf')
        plt.show()

    @staticmethod
    def get_m_theory(T_min, T_max):
        m_theory_range = np.linspace(T_min, T_max, 1000)
        with np.errstate(invalid='ignore'):
            m_theory = (1-np.sinh(2*(1/2*hotspin.kB*math.log(1+math.sqrt(2)))/hotspin.kB/m_theory_range)**-4)**.125 # J. M. D. Coey, Magnetism and Magnetic Materials (6.33)
        m_theory[np.isnan(m_theory)] = 0
        return m_theory_range, m_theory


if __name__ == "__main__":
    test_squareIsing().test(T_steps=21, verbose=True, save=True)
