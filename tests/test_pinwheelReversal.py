''' This file tests the correspondence between experiment and simulation for a
    two-dimensional pinwheel artifical spin ice, which is initialized in a uniform
    state and reversed using an external field. Ideally, the reversal should take place
    in two steps, roughly corresponding to two 90° rotations.
'''

import math
import warnings

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from context import hotspin


class Threshold:
    def __init__(self, thresholds: list[float], start_value=0):
        ''' Use self.pass_check(value) to check if <value> is beyond any of the thresholds
            when seen from the <old_value> of the previous self.pass_check(old_value) call.
            The most recently passed threshold is disabled until another threshold is passed.
            @param thresholds: a list containing floats, each float representing a threshold.
            @param start_value (0): the <old_value> for the first time self.pass_check() is called.
        '''
        self.thresholds = np.asarray(thresholds) # List of thresholds where self.check() returns true if passed
        self.value = start_value # Last seen value
        self._last_threshold_passed = None # Used to make sure we don't activate the same threshold twice in row (is annoying)
    
    def pass_check(self, value):
        ''' Returns True if a threshold was passed when going from <value> to <self.value>. '''
        thresholds_passed = np.where(np.sign(self.thresholds - value) != np.sign(self.thresholds - self.value))[0]
        self.value = value

        if thresholds_passed.size == 0: return False # No thresholds passed
        if self._last_threshold_passed is not None:
            if thresholds_passed.size == 1:
                if self._last_threshold_passed == thresholds_passed[0]: # The only passed threshold is the last passed one
                    return False
        
        # We get here if we passed a threshold AND we either never passed a threshold before, passed multiple thresholds,
        # or the threshold we passed is not the previously passed one. In any case, store the nearest threshold and return True.
        nearest_threshold_i = (np.abs(self.thresholds - value)).argmin()
        self._last_threshold_passed = nearest_threshold_i
        return True


class test_pinwheelReversal:
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 400) # Edge length of simulation as a number of cells
        self.a = kwargs.get('a', 1e-6) # [m] Lattice spacing
        self.T = kwargs.get('T', 300) # [K] Room temperature
        self.H_max = kwargs.get('H_max', 1e-4) # [T] extremal magnitude of external field (default from flatspin paper)
        self.V = kwargs.get('V', 470e-9*170e-9*10e-9) # [m³] volume of a single magnet (default from flatspin paper)

    def test(self, *args, **kwargs):
        self.data = self.test_reversal(*args, **kwargs)
    
    def test_reversal(self, angle: float=0., N: int = 10000, verbose=False, show_intermediate=True, plot=True, save=True):
        ''' Performs a hysteresis of magnetization reversal for an external field at <angle> rad swept
            from <self.H_max> to -<self.H_max> to <self.H_max> again in a total of 2*<N> steps.
        '''
        if verbose: print(f'External field angle is {round(angle*180/math.pi):d} degrees.')
        if not (-math.pi/4 < angle < math.pi/4): warnings.warn(f"Field angle {angle} is outside the nominal -pi/4 < angle < pi/4 range. Undesired behavior might occur.", stacklevel=2)
        self.mm = hotspin.ASI.PinwheelASI(self.size, self.a, T=self.T, PBC=False)
        self.mm.initialize_m(pattern='uniform', angle=0)
        self.mm.add_energy(hotspin.ZeemanEnergy(magnitude=self.H_max, angle=angle))
        thresholds = Threshold([.7, .35, 0, -.35, -.7], start_value=self.mm.m_avg_x) # If the average magnetization crosses this, a plot is shown.
        H_range = np.linspace(self.H_max, -self.H_max, N) # One sweep from low to high H (N steps)
        H_range = np.append(H_range, np.flip(H_range)) # Sweep down and up (2*N steps)
        # H_range = np.roll(H_range, N//2) # Start at zero and ramp up first (optional, 2*N steps), use for determining appropriate H_max
        # H_range = H_range[:H_range.size//4] # Only go from zero to H_max (optional, N/2 steps)
        m_avg_H = np.zeros_like(H_range)
        for i, H in enumerate(H_range):
            if verbose and (i + 1) % 10**(math.floor(math.log10(N))-1) == 0 or i == 0:
                print(f"[{i+1}/{H_range.size}] H = {H:.2e} T, m_x={self.mm.m_avg_x:.2f}")
            self.mm.update(r=1e-7)
            self.mm.get_energy('Zeeman').set_field(magnitude=H)
            self.mm.history_entry()
            m_avg_H[i] = self.mm.m_avg_x*math.cos(angle) + self.mm.m_avg_y*math.sin(angle) # Along the directon of the ext field
            if thresholds.pass_check(self.mm.m_avg_x):
                if verbose: print(f"[{i+1}/{H_range.size}] H = {H:.2e} T, m_x={self.mm.m_avg_x:.2f} (threshold passed{', plotting magnetization...' if show_intermediate else ''})")
                if show_intermediate: hotspin.plottools.show_m(self.mm)

        data = pd.DataFrame({"H": H_range, "m_avg": m_avg_H, "H_angle": angle, "T": self.T})

        savename = f"results/test_pinwheelReversal/N={N:.0f}_H={self.H_max:.2e}_{round(angle*180/math.pi):.0f}deg_T={self.T:.0f}_{self.size}x{self.size}"
        if plot: test_pinwheelReversal.test_reversal_plot(data, save=savename)
        if save: hotspin.plottools.save_data(data, f"{savename}.csv")

        return data

    @staticmethod
    def test_reversal_plot(data: pd.DataFrame, save=False, reduce: int = 10000):
        ''' If <reduce> is nonzero, the number of plotted points is kept reasonable so the pdf does not take eons to load. '''
        N = len(data.index)
        if reduce: data = data.iloc[::math.ceil(N/reduce)]
        M_sat_parallel = data["m_avg"].abs().max() # Assuming that the saturation is achieved somewhere (normally at initialization, so this should be ok)

        hotspin.plottools.init_fonts()
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        ax.plot(data["H"]*1e3, data["m_avg"]/M_sat_parallel, linewidth=1, color='black', zorder=-1)
        ax.scatter(data["H"]*1e3, data["m_avg"]/M_sat_parallel, s=10, zorder=1)
        ax.grid(color='grey', linestyle=':')
        ax.set_xlabel('External field magnitude [mT]')
        ax.set_ylabel('Magnetization $\\langle M_{\\parallel}\\rangle /M_0$')
        ax.set_ylim([-1.1, 1.1])
        plt.gcf().tight_layout()
        if save:
            if not isinstance(save, str):
                save = f"results/test_squareIsing/N={N:.0f}_H={data['H'].max():.2e}_{round(data['H_angle'].iloc[0]*180/math.pi):.0f}deg_T={data['T'].mean():.0f}.pdf"
            hotspin.plottools.save_plot(save, ext='.pdf')
        plt.show()


if __name__ == "__main__":
    test_pinwheelReversal(T=300, size=50).test(angle=30*math.pi/180, N=2000, verbose=True, save=True, show_intermediate=False)
    # Observation: the external field here is MUCH smaller than in the flatspin paper, even though self.V is the same and self.a is reasonable