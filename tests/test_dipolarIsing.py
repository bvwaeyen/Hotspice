r''' This file tests the correspondence between theory and simulation for a
    two-dimensional square Ising model, with exchange and dipolar interactions,
    by observing striped phases as a function of the relative strength $\delta$
    between the exchange and dipolar interaction at low temperature.
    Based on the paper
        J. H. Toloza, F. A. Tamarit, and S. A. Cannas. Aging in a two-dimensional Ising model
        with dipolar interactions. Physical Review B, 58(14):R8885, 1998.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context import hotspice


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

    def test_delta_influence(self, N=2, delta_range=np.arange(0, 4.01, .05), verbose=False, plot=True, save=True):
        self.mm = hotspice.ASI.OOP_Square(self.a, self.size, E_B=0, T=self.T, energies=[hotspice.DipolarEnergy(), hotspice.ExchangeEnergy()], pattern='AFM', PBC=False)
        AFMness = np.zeros_like(delta_range)
        AFMness_std = np.zeros_like(delta_range)
        for i, delta in enumerate(delta_range):
            if verbose: print(f"[{i+1}/{delta_range.size}] delta = {delta:.2f} ...")
            self.delta = delta # Basically sets the exchange energy in self.mm
            MCsteps0 = self.mm.MCsteps
            AFMnesses = []
            while (progress := (self.mm.MCsteps - MCsteps0)/N) < 1:
                self.mm.update(Q=0.1)
                if progress > .5: AFMnesses.append(hotspice.plottools.get_AFMness(self.mm))
            # if verbose: hotspice.plottools.show_m(self.mm)
            self.mm.relax()
            # if verbose: hotspice.plottools.show_m(self.mm)
            AFMness[i] = hotspice.plottools.get_AFMness(self.mm) # np.mean(AFMnesses)
            AFMness_std[i] = np.std(AFMnesses)
        
        df = pd.DataFrame({"delta": delta_range, "AFMness": AFMness, "AFMness_std": AFMness_std})
        metadata = {"description": r"2D Ising model with exchange and dipolar interactions, sweeping $\delta$ as described in `Striped phases in two-dimensional dipolar ferromagnets` by MacIsaac et al."}
        constants = {"nx": self.mm.nx, "ny": self.mm.ny, "MCstepsize": N, "T": self.mm.T_avg}
        data = hotspice.utils.Data(df, metadata=metadata, constants=constants)
        if save: save = data.save(dir="results/test_dipolarIsing", name=f"deltasweep{df['delta'].min()}..{df['delta'].max()}({df['delta'].nunique()})_{self.mm.nx}x{self.mm.ny}")
        if plot or save: test_dipolarIsing.test_delta_influence_plot(df, save=save, show=plot)
        return data
    
    @staticmethod
    def test_delta_influence_plot(df: pd.DataFrame, save=False, show=True):
        hotspice.plottools.init_fonts()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        ax.errorbar(df["delta"], df["AFMness"], yerr=df["AFMness_std"], fmt='o', label='Hotspice')
        ax.set_xlabel(r"$\delta$ (relative exchange/dipolar strength)")
        ax.set_ylabel('AFM-ness')
        ax.set_xlim([df["delta"].min()-.005, df["delta"].max()+.005])
        ax.set_ylim([ax.get_ylim()[0], 1])
        ax.axvline(0.85, linestyle=':', color='black')
        plt.gcf().tight_layout()
        if save:
            if not isinstance(save, str):
                save = f"results/test_dipolarIsing/deltasweep{df['delta'].min()}..{df['delta'].max()}({df['delta'].nunique()}).pdf"
            hotspice.plottools.save_plot(save, ext='.pdf')
        if show: plt.show()


if __name__ == "__main__":
    test_dipolarIsing(size=80).test(verbose=True, save=True)
