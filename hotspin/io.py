import math
# import warnings

import cupy as cp
import numpy as np

from abc import ABC, abstractmethod
# from cupyx.scipy import signal

from .core import Magnets
from .plottools import show_m


class Datastream(ABC):
    def __init__(self):
        self.rng = cp.random.default_rng()

    @abstractmethod
    def get_next(self, n=1):
        """ Calling this method returns a CuPy array containing exactly <n> elements, containing the requested data.
            Depending on the specific subclass, these can be int or float.
        """

    @property
    def is_binary(self): return False

class BinaryDatastream(Datastream):
    """ Just an alias for Datastream: a normal Datastream can contain floats, while this only yields 0 or 1. """
    pass

    @property
    def is_binary(self): return True
# TODO: calculate TAmetrics differently dependent on whether the input data is binary or not.

class Inputter(ABC):
    def __init__(self, datastream: Datastream):
        self.datastream = datastream

    @abstractmethod
    def input_single(self, mm: Magnets, value=None):
        """ Performs an input corresponding to <value> (generated using <self.datastream>)
            into the <mm> simulation, and returns this <value>.
        """
        if value is None: value = self.datastream.get_next()
        return value


class OutputReader(ABC):
    def __init__(self, mm=None):
        if mm is not None: self.configure_for(mm)

    @abstractmethod
    def read_state(self, mm: Magnets) -> cp.ndarray:
        """ Reads the current state of the <mm> simulation in some way. """
        self.state = cp.arange(self.n)
        return self.state

    def configure_for(self, mm: Magnets):
        """ Subclassing this method is optional. When called, some properties of this OutputReader object
            are initialized, which depend on the Magnets object <mm>.
        """
        self.mm = mm

    @property
    @abstractmethod
    def n(self):
        ''' The number of output values when reading a given state. '''


######## Below are subclasses of the superclasses above
# TODO: class FileDatastream(Datastream) which reads bits from a file? Can use package 'bitstring' for this.
# TODO: class SemiRepeatingDatastream(Datastream) which has first <n> random bits and then <m> bits which are the same for all runs
class RandomBinaryDatastream(BinaryDatastream):
    def __init__(self, p0=.5):
        ''' Generates random bits, with <p0> probability of getting 0, and 1-<p0> probability of getting 1.
            @param p0 [float] (0.5): the probability of 0 when generating a random bit.
        '''
        self.p0 = p0
        super().__init__()

    def get_next(self, n=1):
        return cp.where(self.rng.random(size=(n,)) < self.p0, 0, 1)

class RandomUniformDatastream(Datastream):
    def __init__(self, low=0, high=1):
        ''' Generates uniform random floats between <low=0> and <high=1>. '''
        self.low, self.high = low, high
        super().__init__()

    def get_next(self, n=1):
        return (self.high - self.low)*self.rng.random(size=(n,)) + self.low


class FieldInputter(Inputter):
    def __init__(self, datastream: Datastream, magnitude=1, angle=0, n=2, sine=0, half_period=True):
        ''' Applies an external field at <angle> rad, whose magnitude is <magnitude>*<datastream.get_next()>.
            <sine> specifies the frequency [Hz] of the sinusoidal field, if it is nonzero.
            If <half_period> is True, only the first half-period of the sine is used, otherwise the full ∿ period.
        '''
        super().__init__(datastream)
        self.angle = angle
        self.magnitude = magnitude
        self.n = n # The max. number of Monte Carlo steps performed every time self.input_single(mm) is called
        self.sine = sine # Whether or not to use a sinusoidal field strength
        self.half_period = half_period

    @property
    def angle(self): # [rad]
        return self._angle
    @angle.setter
    def angle(self, value):
        self._angle = value % (2*math.pi)

    @property
    def magnitude(self): # [T]
        return self._magnitude
    @magnitude.setter
    def magnitude(self, value):
        self._magnitude = value

    def input_single(self, mm: Magnets, value=None):
        if self.sine != 0 and mm.params.UPDATE_SCHEME != 'Néel':
            raise AttributeError("Can not use temporal sine=True if UPDATE_SCHEME != 'Néel'.")
        
        if value is None: value = self.datastream.get_next()
        mm.get_energy('Zeeman').set_field(angle=(self.angle if mm.in_plane else None)) # set angle only once
        MCsteps0 = mm.MCsteps
        if not self.sine: # If frequency is zero: use constant magnitude
            mm.get_energy('Zeeman').set_field(magnitude=self.magnitude*value)
            while (progress := (mm.MCsteps - MCsteps0)/self.n) < 1:
                mm.update()
        else:
            t0 = mm.t
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.sine)) < 1:
                mm.get_energy('Zeeman').set_field(magnitude=self.magnitude*value*math.sin(progress*math.pi*(2-self.half_period)))
                mm.update(t_max=0.1/self.sine) # At least 10 steps per sine-period
        return value


class FieldInputterBinary(FieldInputter):
    def __init__(self, datastream: Datastream, magnitudes: tuple = (.8, 1), **kwargs):
        ''' Exactly the same as FieldInputter, but if <datastream> yields 0 then a field with
            magnitude <magnitudes[0]> is applied, otherwise <magnitudes[1]>.
            Using <sine=<frequency[Hz]>> yields behavior as in "Computation in artificial spin ice" by Jensen et al.
            Differing attributes compared to FieldInputter:
                <self.magnitude> is set to <magnitudes[1]>
                <self.magnitude_ratio> is used to store the ratio <magnitudes[0]/magnitudes[1]>.
        '''
        self.magnitude_ratio = magnitudes[0]/magnitudes[1]
        super().__init__(datastream, magnitude=magnitudes[1], half_period=False, **kwargs)

    def input_single(self, mm: Magnets, value=None):
        if value is None: value = self.datastream.get_next()
        value = self.magnitude_ratio if value == 0 else 1
        return super().input_single(mm, value=value)


class PerpFieldInputter(FieldInputter):
    def __init__(self, datastream: BinaryDatastream, magnitude=1, angle=0, n=2, sine=0):
        ''' Applies an external field, whose angle depends on the bit that is being applied:
            the bit '0' corresponds to an angle of <phi> radians, the bit '1' to <phi>+pi/2 radians.
            Also works for continuous numbers, resulting in intermediate angles, but this is not the intended use.
            If <sine> is True, the field starts at zero strength and follows half a sine cycle back to zero during
                the <n> steps that were specified. If False, a constant field is used for the entire duration.
        '''
        super().__init__(datastream, magnitude=magnitude, angle=angle, n=n, sine=sine)

    def input_single(self, mm: Magnets, value=None):
        if self.sine != 0 and mm.params.UPDATE_SCHEME != 'Néel':
            raise AttributeError("Can not use temporal sine=True if UPDATE_SCHEME != 'Néel'.")
        if not mm.in_plane:
            raise AttributeError("Can not use PerpFieldInputter on an out-of-plane ASI.")

        if value is None: value = self.datastream.get_next()
        angle = self.angle + value*math.pi/2
        MCsteps0 = mm.MCsteps
        if not self.sine: # If frequency is zero: use constant magnitude
            mm.get_energy('Zeeman').set_field(magnitude=self.magnitude, angle=angle)
            while (progress := (mm.MCsteps - MCsteps0)/self.n) < 1:
                mm.update()
        else:
            t0 = mm.t
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.sine)) < 1:
                mm.get_energy('Zeeman').set_field(magnitude=self.magnitude*math.sin(progress*math.pi), angle=angle)
                mm.update(t_max=0.1/self.sine) # At least 10 steps per sine-period
        return value


class RegionalOutputReader(OutputReader):
    def __init__(self, nx, ny, mm=None):
        ''' Reads the current state of the ASI with a certain level of detail.
            @param nx [int]: number of averaging bins in the x-direction.
            @param ny [int]: number of averaging bins in the y-direction.
            @param mm [hotspin.Magnets] (None): if specified, this OutputReader automatically calls self.configure_for(mm).
        '''
        self.nx, self.ny = nx, ny
        self.grid = np.zeros((self.nx, self.ny)) # We use this to ndenumerate, but CuPy does not have this, so use NumPy
        if mm is not None: self.configure_for(mm)

    @property
    def n(self):
        return self.state.size

    def configure_for(self, mm: Magnets):
        self.mm = mm
        self.region_x = cp.floor(cp.linspace(0, self.nx, mm.nx)).astype(int)
        self.region_y = cp.floor(cp.linspace(0, self.ny, mm.ny)).astype(int)
        self.region_x[-1] = self.nx - 1
        self.region_y[-1] = self.ny - 1
        self.region_x = np.tile(self.region_x, (mm.ny, 1))
        self.region_y = np.tile(self.region_y, (mm.nx, 1)).T

        n = cp.zeros_like(self.grid)
        # Determine the number of magnets in each region
        for i, _ in np.ndenumerate(self.grid): # CuPy has no ndenumerate, so use NumPy then
            here = (self.region_x == i[0]) & (self.region_y == i[1])
            n[i] = cp.sum(mm.occupation[here])
        self.normalization_factor = np.max(cp.asarray(cp.max(self.mm.moment)*n).get())

        if mm.in_plane:
            self.state = cp.zeros((self.nx, self.ny, 2))
        else:
            self.state = cp.zeros((self.nx, self.ny))

    def read_state(self, mm: Magnets = None, m=None) -> cp.ndarray:
        if mm is not None: self.configure_for(mm) # If mm not specified, we suppose configure_for() already happened
        if m is None: m = self.mm.m # If m is specified, it takes precendence over mm regardless of whether mm was specified too
        if self.mm.in_plane:
            m_x = m*self.mm.orientation[:,:,0]*self.mm.moment/self.normalization_factor
            m_y = m*self.mm.orientation[:,:,1]*self.mm.moment/self.normalization_factor

        for i, _ in np.ndenumerate(self.grid): # CuPy has no ndenumerate, so use NumPy then
            here = (self.region_x == i[0]) & (self.region_y == i[1])
            if self.mm.in_plane:
                self.state[i[0], i[1], 0] = cp.sum(m_x[here]) # Average m_x
                self.state[i[0], i[1], 1] = cp.sum(m_y[here]) # Average m_y
            else:
                self.state[i[0], i[1]] = cp.sum(m[here])

        return self.state # [Am²]

    def inflate_flat_array(self, arr: np.ndarray|cp.ndarray):
        ''' Transforms a 1D array <arr> to have the same shape as <self.state>.
            Basically the inverse transformation as done on <self.state> when calling <self.state.reshape(-1)>.
            @param arr [array]: a NumPy or CuPy array of shape (<self.n>,).
        '''
        if self.mm.in_plane:
            return arr.reshape(self.nx, self.ny, 2)
        else:
            return arr.reshape(self.nx, self.ny)
