r''' This file tests the correspondence between theory and simulation for a
    two-dimensional square Ising model, with exchange and dipolar interactions,
    by observing striped phases as a function of the relative strength $\delta$
    between the exchange and dipolar interaction at low temperature.
    Based on the paper
        J. H. Toloza, F. A. Tamarit, and S. A. Cannas. Aging in a two-dimensional Ising model
        with dipolar interactions. Physical Review B, 58(14):R8885, 1998.
'''
# TODO: add error bars and change line to be just data-points
import math

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context import hotspin


class test_dipolarIsing:
    def __init__(self, **kwargs):
        self.T = kwargs.get('T', 100) # Low temperature to avoid paramagnetic state
        self.a = 1e-6 # Lattice spacing
        self.size = kwargs.get('size', 400) # Large to get the most statistically ok behavior
    
    @property
    def dipolar_nearest(self):
        ''' Yields the dipolar interaction energy between nearest neighbors. '''
        return 1e-7*self.mm.get_energy('dipolar').prefactor*(self.mm._momentSq)/(self.mm.a**3)
    
    @property
    def delta(self):
        ''' Dipolar energy magnitude:  m*m* prefactor/(a**3)*1e-7*(moment**2)
            Exchange energy magnitude: m*m* J
            So delta = exchange/dipolar is given by...
        '''
        return 2*self.mm.get_energy('exchange').J/self.dipolar_nearest
    
    @delta.setter
    def delta(self, value):
        self.mm.get_energy('exchange').J = value*self.dipolar_nearest/2
    
    def test(self, *args, **kwargs):
        self.data = self.test_delta_influence(*args, **kwargs)

    def test_delta_influence(self, N=2, delta_range=np.arange(0, 3.01, .1), verbose=False, plot=True, save=True):
        self.mm = hotspin.ASI.FullASI(self.size, self.a, E_B=0, T=self.T, energies=[hotspin.DipolarEnergy(), hotspin.ExchangeEnergy()], pattern='AFM', PBC=False)
        AFMness = np.zeros_like(delta_range)
        for i, delta in enumerate(delta_range):
            if verbose: print(f"[{i+1}/{delta_range.size}] delta = {delta:.2f} ...")
            self.delta = delta # Basically sets the exchange energy in self.mm
            MCsteps0 = self.mm.MCsteps
            while self.mm.MCsteps - MCsteps0 < N:
                self.mm.update(Q=0.1)
            # if verbose: hotspin.plottools.show_m(self.mm)
            self.mm.relax()
            # if verbose: hotspin.plottools.show_m(self.mm)
            AFMness[i] = hotspin.plottools.get_AFMness(self.mm)
        
        data = pd.DataFrame({"delta": delta_range, "AFMness": AFMness})
        metadata = {"description": r"2D Ising model with exchange and dipolar interactions, sweeping $\delta$ as described in `Striped phases in two-dimensional dipolar ferromagnets` by MacIsaac et al."}
        constants = {"nx": self.mm.nx, "ny": self.mm.ny, "MCstepsize": N, "T": self.T_avg}
        if save:
            full_json = hotspin.utils.combine_json(data, metadata=metadata, constants=constants)
            savepath = hotspin.utils.save_json(full_json, path="results/test_dipolarIsing", name=f"deltasweep{data['delta'].min()}..{data['delta'].max()}({data['delta'].nunique()})_{self.mm.nx}x{self.mm.ny}")
            if plot: test_dipolarIsing.test_delta_influence_plot(data, save=savepath)
        else:
            if plot: test_dipolarIsing.test_delta_influence_plot(data)
        return data
    
    @staticmethod
    def test_delta_influence_plot(data: pd.DataFrame, save=False):
        hotspin.plottools.init_fonts()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        ax.plot(data["delta"], data["AFMness"])
        ax.set_xlabel(r"$\delta$ (relative exchange/dipolar strength)")
        ax.axvline(0.85, linestyle=':', color='black')
        ax.set_ylabel('AFM-ness')
        plt.gcf().tight_layout()
        if save:
            if not isinstance(save, str):
                save = f"results/test_dipolarIsing/deltasweep{data['delta'].min()}..{data['delta'].max()}({data['delta'].nunique()}).pdf"
            hotspin.plottools.save_plot(save, ext='.pdf')
        plt.show()


if __name__ == "__main__":
    test_dipolarIsing(size=40).test(verbose=True, save=True)
