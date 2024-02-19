""" This file tests the correspondence between theory and simulation for a
    two-dimensional square Ising model, with only exchange interactions,
    by observing the average magnetization as a function of temperature.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Literal

from context import hotspice


class test_squareIsing:
    def __init__(self, size: int = 800):
        self.J = hotspice.kB*300 # This does not matter much, just dont take it too large to prevent float errors
        self.T_lim = [0.9, 1.1] # Relative to T_c
        self.a = 1 # Lattice spacing, chosen large to get many simultaneous switches in Glauber, exchange energy is NN anyway
        self.size = size # Large to get the most statistically ok behavior

    def test(self, *args, **kwargs):
        self.data = self.test_N_influence(*args, **kwargs)

    @property
    def T_c(self):
        return 2*self.J/hotspice.kB/math.log(1+math.sqrt(2))

    def test_magnetization(self, T_steps=21, N=1000, scheme: Literal['Glauber', 'Néel', 'Wolff'] = 'Glauber', verbose=False, plot=True, save=False, reverse=False) -> hotspice.utils.Data:
        """ Performs a sweep of the temperature in <T_steps> steps. At each step, <N> Magnets.update() calls are performed.
            The final half of these <N> update calls are recorded, from which and their average/stdev of m_avg calculated.
            @param reverse [bool] (False): if True, the temperature steps are in descending order, otherwise ascending.
        """
        simparams = hotspice.SimParams(UPDATE_SCHEME=scheme)
        self.mm = hotspice.ASI.OOP_Square(self.a, self.size, energies=[hotspice.ExchangeEnergy(J=self.J)], PBC=True, pattern=('random' if reverse else 'uniform'), params=simparams)

        T_range = np.linspace(self.T_lim[0], self.T_lim[1], T_steps)
        if reverse: T_range = np.flip(T_range)
        m_avg = np.zeros_like(T_range)
        m_std = np.zeros_like(T_range)
        NN_corr = np.zeros_like(T_range)
        NN_corr_std = np.zeros_like(T_range)
        for i, T in enumerate(T_range):
            if verbose: print(f"[{i+1}/{T_steps}] N={N}, T = {T:.2f}*T_c (= {T*self.T_c:.0f} K)...")
            self.mm.T = T*self.T_c
            averages_m = []
            corrs = []
            for n in range(N):
                self.mm.update()
                if n > N/2:
                    averages_m.append(abs(self.mm.m_avg))
                    corrs.append(self.mm.correlation_NN())
            m_avg[i], m_std[i] = np.mean(averages_m), np.std(averages_m)
            NN_corr[i], NN_corr_std[i] = np.mean(corrs), np.std(corrs)

        df = pd.DataFrame({'T': T_range, 'm_avg': m_avg, 'm_std': m_std, 'NN_corr': NN_corr, 'NN_corr_std': NN_corr_std}) # TODO: add NN correlation or something similar
        metadata = {'description': r"Magnetization of 2D exchange-coupled Ising model near critical temperature."}
        constants = {'nx': self.mm.nx, 'ny': self.mm.ny, 'N': N, 'UPDATE_SCHEME': self.mm.params.UPDATE_SCHEME, 'Tsweep_reverse': reverse}
        data = hotspice.utils.Data(df, metadata=metadata, constants=constants)
        if save:
            Tsweep_direction = 'reverse' if reverse else ''
            savepath = data.save(dir="results/test_squareIsing", name=f"Tsweep{df['T'].nunique()}{Tsweep_direction}_N{N}_{self.mm.params.UPDATE_SCHEME}_{self.mm.nx}x{self.mm.ny}")
            if plot: test_squareIsing.test_magnetization_plot(df, save=savepath)
        else:
            if plot: test_squareIsing.test_magnetization_plot(df)
        return data
    
    def test_N_influence(self, *args, plot=True, save=True, **kwargs):
        """ Tests the influence of <N> (the number of iterations per value of T) on m_avg.
            Any arguments (not 'N') passed to this function are passed through to test_magnetization().
        """
        df = pd.DataFrame()

        N_range = [1000, 2000, 4000]
        for N in N_range:
            local_data = self.test_magnetization(*args, N=N, plot=False, save=False, **kwargs)
            local_df = local_data.df
            local_df['N'] = N
            df = pd.concat([df, local_df])

        metadata = {'description': r"Magnetization of 2D exchange-coupled Ising model near critical temperature, for different N (the number of update steps for each data point)."}
        constants = local_data.constants
        data = hotspice.utils.Data(df, metadata=metadata, constants=constants)
        if save:
            Tsweep_direction = 'reverse' if constants['Tsweep_reverse'] else ''
            save = data.save(dir="results/test_squareIsing", name=f"Tsweep{df['T'].nunique()}{Tsweep_direction}_Nsweep{df['N'].nunique()}_{constants['UPDATE_SCHEME']}_{constants['nx']}x{constants['ny']}")
        if plot or save: test_squareIsing.test_N_influence_plot(df, save=save, show=plot)
        return data

    @staticmethod
    def test_magnetization_plot(df: pd.DataFrame, save=False, show=True):
        """ If <save> is bool, the filename is automatically generated. If <save> is str, it is used as filename. """
        T_lim = [df['T'].min(), df['T'].max()]

        hotspice.plottools.init_style()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        ax.errorbar(df['T'], df['m_avg'], yerr=df['m_std'], fmt='o', label=r"$\langle\vec{m}\rangle$ Hotspice")
        if 'NN_corr' in df: ax.errorbar(df['T'], df['NN_corr'], yerr=df['NN_corr_std'], fmt='^', label="NN correlation Hotspice")
        theory_T = np.linspace(T_lim[0]-.005, T_lim[1]+.005)
        ax.plot(theory_T, test_squareIsing.get_m_theory(theory_T), color='black', label=r"$\langle\vec{m}\rangle$ theory")
        ax.legend()
        ax.set_xlabel("Temperature $T/T_c$")
        ax.set_ylabel(r"Magnetization $\langle M \rangle /M_0$")
        ax.set_xlim([T_lim[0]-.005, T_lim[1]+.005])
        ax.set_ylim([-0.01, 1])
        plt.gcf().tight_layout()
        if save:
            if not isinstance(save, str):
                reverse = "" if df['T'].iloc[0] < df['T'].iloc[-1] else "reverse"
                save = f"results/test_squareIsing/Tsweep{df['T'].nunique()}{reverse}.pdf"
            hotspice.plottools.save_plot(save, ext='.pdf')
        if show: plt.show()

    @staticmethod
    def test_N_influence_plot(df: pd.DataFrame, save=False, show=True):
        """ If <save> is bool, the filename is automatically generated. If <save> is str, it is used as filename. """
        T_lim = [df['T'].min(), df['T'].max()]

        hotspice.plottools.init_style()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        for i, (N, local_df) in enumerate(df.groupby('N')):
            ax.errorbar(local_df['T'], local_df['m_avg'], yerr=local_df['m_std'], fmt='o', color=f'C{i}', label=r"$\langle\vec{m}\rangle$" + f" (N={N})")
            if 'NN_corr' in local_df: ax.errorbar(local_df['T'], local_df['NN_corr'], yerr=local_df['NN_corr_std'], fmt='^', color=f'C{i}', label="NN correlation" + f" (N={N})")
        theory_T = np.linspace(T_lim[0]-.005, T_lim[1]+.005)
        ax.plot(theory_T, test_squareIsing.get_m_theory(theory_T), color='black', label=r"$\langle\vec{m}\rangle$ theory")
        ax.legend()
        ax.set_xlabel("Temperature $T/T_c$")
        ax.set_ylabel("Magnetization $\\langle M\\rangle /M_0$")
        ax.set_xlim([T_lim[0]-.005, T_lim[1]+.005])
        ax.set_ylim([-0.01, 1])
        plt.gcf().tight_layout()
        if save:
            if not isinstance(save, str):
                reverse = "" if df['T'].iloc[0] < df['T'].iloc[-1] else "reverse"
                save = f"results/test_squareIsing/Tsweep{df['T'].nunique()}{reverse}_Nsweep{df['N'].nunique()}.pdf"
            hotspice.plottools.save_plot(save, ext='.pdf')
        if show: plt.show()

    @staticmethod
    def get_m_theory(T):
        with np.errstate(invalid='ignore'):
            m_theory = (1 - np.sinh(2*(1/2*hotspice.kB*math.log(1 + math.sqrt(2)))/(hotspice.kB*T))**-4)**.125 # J. M. D. Coey, Magnetism and Magnetic Materials (6.33)
        m_theory[np.isnan(m_theory)] = 0 # Above T_C
        return m_theory


if __name__ == "__main__":
    # test_squareIsing().test(T_steps=21, verbose=True, save=True)
    # test_squareIsing().test(T_steps=21, scheme='Wolff', verbose=True, save=True)
    
    test_squareIsing(size=800).test(T_steps=21, scheme='Glauber', verbose=True, save=False)
    # test_squareIsing(size=800).test(T_steps=21, scheme='Glauber', verbose=True, save=True, reverse=True)
