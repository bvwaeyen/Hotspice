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
    def __init__(self, thresholds, start_value=0):
        self.thresholds = np.asarray(thresholds) # List of thresholds where self.check() returns true if passed
        self.value = start_value # Last seen value
        self.last_threshold_passed = None # Used to make sure we don't activate the same threshold twice in row (is annoying)
    
    def pass_check(self, value):
        ''' Returns True if a threshold was passed when going from <value> to <self.value>. '''
        thresholds_passed = np.where(np.sign(self.thresholds - value) != np.sign(self.thresholds - self.value))[0]
        self.value = value

        if thresholds_passed.size == 0: return False # No thresholds passed
        if self.last_threshold_passed is not None:
            if thresholds_passed.size == 1:
                if self.last_threshold_passed == thresholds_passed[0]: # The only passed threshold is the last passed one
                    return False
        
        # We get here if we passed a threshold AND we either never passed a threshold before, passed multiple thresholds,
        # or the threshold we passed is not the previously passed one. In any case, store the nearest threshold and return True.
        nearest_threshold_i = (np.abs(self.thresholds - value)).argmin()
        self.last_threshold_passed = nearest_threshold_i
        return True


class test_pinwheelReversal:
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 400) # Edge length of simulation as a number of cells
        self.a = kwargs.get('a', 1e-6) # [m] Lattice spacing
        self.T = kwargs.get('T', 300) # [K] Room temperature
        self.H_mag = kwargs.get('H_mag', 1e-4) # [T] extremal magnitude of external field (default from flatspin paper)
        self.V = kwargs.get('V', 470e-9*170e-9*10e-9) # [m³] volume of a single magnet (default from flatspin paper)

    def test(self, *args, **kwargs):
        self.data = self.test_reversal(*args, **kwargs)
    
    def test_reversal(self, angle: float=0., N=10000, verbose=False, show_intermediate=True, save=False):
        ''' Performs a reversal for an external field at <angle>° swept from -<self.H_mag> to <self.H_mag>. '''
        if verbose: print(f'External field angle is {round(angle*180/math.pi):d} degrees.')
        if not (-math.pi/4 < angle < math.pi/4): warnings.warn(f"Field angle {angle} is outside the nominal -pi/4 < angle < pi/4 range. Undesired behavior might occur.", stacklevel=2)
        self.mm = hotspin.ASI.PinwheelASI(self.size, self.a, T=self.T, PBC=False)
        self.mm.initialize_m(pattern='uniform', angle=0)
        self.mm.add_energy(hotspin.ZeemanEnergy(magnitude=self.H_mag, angle=angle))
        thresholds = Threshold([.7, .35, 0, -.35, -.7], start_value=self.mm.m_avg_x) # If the average magnetization crosses this, a plot is shown.
        H_range = np.linspace(-self.H_mag, self.H_mag, N)
        H_range = np.roll(np.append(H_range, np.flip(H_range)), N//2)
        m_avg_x = np.zeros_like(H_range)
        for i, H in enumerate(H_range):
            if verbose and not (i + 1) % 10**(math.floor(math.log10(N))-1):
                print(f"[{i+1}/{H_range.size}] H = {H:.2e} T, m_x={self.mm.m_avg_x:.2f}")
            self.mm.update()
            self.mm.get_energy('Zeeman').set_field(magnitude=H)
            self.mm.history_entry()
            m_avg_x[i] = self.mm.m_avg_x
            if thresholds.pass_check(self.mm.m_avg_x):
                if verbose: print(f"[{i+1}/{H_range.size}] H = {H:.2e} T, m_x={self.mm.m_avg_x:.2f} (threshold passed{', plotting magnetization...' if show_intermediate else ''})")
                if show_intermediate: hotspin.plottools.show_m(self.mm)

        plt.scatter(H_range, m_avg_x)
        plt.show()
    # TODO: save hysteresis in pandas dataframe
    # TODO: plot nice hysteresis curve
    # TODO: use slightly different <m> measurement like is used in flatspin paper


if __name__ == "__main__":
    test_pinwheelReversal(T=300, size=400).test(angle=30*math.pi/180, N=10000, verbose=True, save=False, show_intermediate=True)
    # Observation: the external field here is MUCH smaller than in the flatspin paper, even though self.V is the same and self.a is reasonable
