import math
import warnings

import numpy as np

from abc import ABC, abstractmethod

from .ASI import OOP_Square
from .core import Magnets
from .plottools import show_m
from .utils import log
from . import config
if config.USE_GPU:
    import cupy as xp
else:
    import numpy as xp


class Datastream(ABC):
    def __init__(self):
        self.rng = xp.random.default_rng()

    @abstractmethod
    def get_next(self, n=1) -> xp.ndarray:
        """ Calling this method returns an array containing exactly <n> elements, containing the requested data.
            Depending on the specific subclass, these can be int or float.
        """

    @property
    def is_binary(self): return False

class BinaryDatastream(Datastream):
    """ Just an alias for Datastream: a normal Datastream can contain floats, while this only yields 0 or 1. """
    pass

    @property
    def is_binary(self): return True


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
    def read_state(self, mm: Magnets) -> xp.ndarray:
        """ Reads the current state of the <mm> simulation in some way. """
        self.state = xp.arange(self.n)
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

    def get_next(self, n=1) -> xp.ndarray:
        return xp.where(self.rng.random(size=(n,)) < self.p0, 0, 1)

class RandomUniformDatastream(Datastream):
    def __init__(self, low=0, high=1):
        ''' Generates uniform random floats between <low=0> and <high=1>. '''
        self.low, self.high = low, high
        super().__init__()

    def get_next(self, n=1) -> xp.ndarray:
        return (self.high - self.low)*self.rng.random(size=(n,)) + self.low


class ConstantDatastream(Datastream):
    def __init__(self, value=1):
        '''Always returns the same <value=1>'''
        self.value = value
        super().__init__()

    def get_next(self, n=1) -> xp.ndarray:
        return xp.full((n,), self.value)


class FieldInputter(Inputter):
    def __init__(self, datastream: Datastream, magnitude=1, angle=0, n=2, sine=False, frequency=1, half_period=True):
        ''' Applies an external field at <angle> rad, whose magnitude is <magnitude>*<datastream.get_next()>.
            <frequency> specifies the frequency [Hz] at which bits are being input to the system, if it is nonzero.
            If <sine> is True, the field magnitude follows a full sinusoidal period for each input bit.
            If <half_period> is True, only the first half-period ⏜ of the sine is used, otherwise the full ∿ period.
            At most <n> Monte Carlo steps will be performed.
        '''
        super().__init__(datastream)
        self.angle = angle
        self.magnitude = magnitude
        self.n = n # The max. number of Monte Carlo steps performed every time self.input_single(mm) is called
        self.sine = sine # Whether or not to use a sinusoidal field strength
        self.frequency = frequency
        self.half_period = half_period

    @property
    def angle(self): # [rad]
        return self._angle
    @angle.setter
    def angle(self, value):
        self._angle = value % math.tau

    @property
    def magnitude(self): # [T]
        return self._magnitude
    @magnitude.setter
    def magnitude(self, value):
        self._magnitude = value

    def input_single(self, mm: Magnets, value=None):
        if self.sine and mm.params.UPDATE_SCHEME != 'Néel':
            raise AttributeError("Can not use temporal sine=True if UPDATE_SCHEME != 'Néel'.")
        if value is None: value = self.datastream.get_next()

        mm.get_energy('Zeeman').set_field(angle=(self.angle if mm.in_plane else None)) # set angle only once
        MCsteps0, t0 = mm.MCsteps, mm.t
        if self.sine: # Change magnitude sinusoidally
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency)) < 1:
                mm.get_energy('Zeeman').set_field(magnitude=self.magnitude*value*math.sin(progress*math.pi*(2-self.half_period)))
                mm.update(t_max=min(1-progress, 0.05)/self.frequency) # At least 20 (=1/0.05) steps per sine-period
        else: # Use constant magnitude
            mm.get_energy('Zeeman').set_field(magnitude=self.magnitude*value)
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency)) < 1:
                if self.frequency:
                    mm.update(t_max=min(1-progress, 1)/self.frequency) # No sine, so no 20 steps per wavelength needed here
                else:
                    mm.update() # No time
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
    def __init__(self, datastream: BinaryDatastream, magnitude=1, angle=0, n=2, relax=True, frequency=1, **kwargs):
        ''' Applies an external field, whose angle depends on the bit that is being applied:
            the bit '0' corresponds to an angle of <phi> radians, the bit '1' to <phi>+pi/2 radians.
            Also works for continuous numbers, resulting in intermediate angles, but this is not the intended use.
            For more information about the kwargs, see FieldInputter.
        '''
        self.relax = relax
        super().__init__(datastream, magnitude=magnitude, angle=angle, n=n, frequency=frequency)


    def input_single(self, mm: Magnets, value=None):
        if (self.sine or self.frequency) and mm.params.UPDATE_SCHEME != 'Néel':
            raise AttributeError("Can not use temporal sine=True or nonzero frequency if UPDATE_SCHEME != 'Néel'.")
        if not mm.in_plane:
            raise AttributeError("Can not use PerpFieldInputter on an out-of-plane ASI.")
        if value is None: value = self.datastream.get_next()

        e = mm.get_energy('Zeeman')
        angle = self.angle + value*math.pi/2
        MCsteps0, t0 = mm.MCsteps, mm.t

        # pos and neg field
        if self.relax:
            for mag in [self.magnitude, -self.magnitude]:
                e.set_field(magnitude=mag, angle=angle)
                mm.minimize()
            # log(f'Performed {mm.MCsteps - MCsteps0} MC steps.')
        else:
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency)) < .5 - 1e-6:
                if self.frequency != 0: mm.update(t_max=0.5/self.frequency)
                else: mm.update()
                # log('updated forward')
            e.set_field(magnitude=-self.magnitude, angle=angle)
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency)) < 1 - 1e-6:
                if self.frequency != 0: mm.update(t_max=0.5/self.frequency)
                else: mm.update()
                # log('updated reverse')

        return value


    def input_single_generalized(self, mm: Magnets, value=None):
        if (self.sine or self.frequency) and mm.params.UPDATE_SCHEME != 'Néel':
            raise AttributeError("Can not use temporal sine=True or nonzero frequency if UPDATE_SCHEME != 'Néel'.")
        if not mm.in_plane:
            raise AttributeError("Can not use PerpFieldInputter on an out-of-plane ASI.")
        if value is None: value = self.datastream.get_next()

        angle = self.angle + value*math.pi/2
        MCsteps0, t0 = mm.MCsteps, mm.t
        if self.sine: # Change magnitude sinusoidally
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency)) < 1 - 1e-6: # t and MCsteps not exceeded, - 1e-6 to be sure
                mm.get_energy('Zeeman').set_field(magnitude=self.magnitude*math.sin(progress*math.pi*(2-self.half_period)), angle=angle)
                mm.update(t_max=min(1-progress, 0.125)/self.frequency) # At least 8 (=1/0.125) steps per sine-period
        else: # Use constant magnitude
            mm.get_energy('Zeeman').set_field(magnitude=self.magnitude, angle=angle)
            while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency)) < 1 - 1e-6:
                if self.frequency:
                    mm.update(t_max=min(1-progress, 1)/self.frequency) # No sine, so no 20 steps per wavelength needed here
                else:
                    mm.update() # No time

        return value


class RegionalOutputReader(OutputReader):
    def __init__(self, nx, ny, mm=None):
        ''' Reads the current state of the ASI with a certain level of detail.
            @param nx [int]: number of averaging bins in the x-direction.
            @param ny [int]: number of averaging bins in the y-direction.
            @param mm [hotspice.Magnets] (None): if specified, this OutputReader automatically calls self.configure_for(mm).
        '''
        self.nx, self.ny = nx, ny
        self.grid = np.zeros((self.nx, self.ny)) # We use this to ndenumerate, but CuPy does not have this, so use NumPy
        if mm is not None: self.configure_for(mm)

    @property
    def n(self):
        return self.state.size

    def configure_for(self, mm: Magnets):
        self.mm = mm
        self.region_x = np.linspace(0, self.nx, mm.nx, dtype=int)
        self.region_y = np.linspace(0, self.ny, mm.ny, dtype=int)
        self.region_x[-1] = self.nx - 1
        self.region_y[-1] = self.ny - 1
        self.region_x = np.tile(self.region_x, (mm.ny, 1))
        self.region_y = np.tile(self.region_y, (mm.nx, 1)).T

        n = xp.zeros_like(self.grid, dtype=int) # Number of magnets in each region
        # Determine the number of magnets in each region
        for i, _ in np.ndenumerate(self.grid): # CuPy has no ndenumerate, so use NumPy then
            here = (self.region_x == i[0]) & (self.region_y == i[1])
            n[i] = xp.sum(mm.occupation[here]) # TODO: <here> is np, <occupation> is xp, is this ok?
        self.normalization_factor = float(xp.max(self.mm.moment))*n

        if mm.in_plane:
            self.state = xp.zeros((self.nx, self.ny, 2))
        else:
            self.state = xp.zeros((self.nx, self.ny))

    def read_state(self, mm: Magnets = None, m=None) -> xp.ndarray:
        if mm is not None: self.configure_for(mm) # If mm not specified, we suppose configure_for() already happened
        if m is None: m = self.mm.m # If m is specified, it takes precendence over mm regardless of whether mm was specified too
        if self.mm.in_plane:
            m_x = m*self.mm.orientation[:,:,0]*self.mm.moment
            m_y = m*self.mm.orientation[:,:,1]*self.mm.moment

        # TODO: make self.state a 1D array, which can certainly be good for sparse simulations
        for i, _ in np.ndenumerate(self.grid): # CuPy has no ndenumerate, so use NumPy then
            here = (self.region_x == i[0]) & (self.region_y == i[1])
            if self.mm.in_plane:
                self.state[i[0], i[1], 0] = xp.sum(m_x[here]) # Average m_x
                self.state[i[0], i[1], 1] = xp.sum(m_y[here]) # Average m_y
            else:
                self.state[i[0], i[1]] = xp.sum(m[here])

        if self.mm.in_plane:
            self.state[:,:,0] /= self.normalization_factor
            self.state[:,:,1] /= self.normalization_factor
        else:
            self.state /= self.normalization_factor

        return self.state # [Am²]

    def inflate_flat_array(self, arr: np.ndarray|xp.ndarray):
        ''' Transforms a 1D array <arr> to have the same shape as <self.state>.
            Basically the inverse transformation as done on <self.state> when calling <self.state.reshape(-1)>.
            @param arr [array]: an array of shape (<self.n>,).
        '''
        if self.mm.in_plane:
            return arr.reshape(self.nx, self.ny, 2)
        else:
            return arr.reshape(self.nx, self.ny)


class OOPSquareChessFieldInputter(Inputter):
    def __init__(self, datastream: Datastream, magnitude: float = 1, n=4, frequency=1):
        ''' Applies an external stimulus in a checkerboard pattern. This is modelled as an external
            field for each magnet, with magnitude <magnitude>*<datastream.get_next()>.
            A checkerboard pattern is used to break symmetry between the two ground states of OOP_Square ASI.
            <frequency> specifies the frequency [Hz] at which bits are being input to the system, if it is nonzero.
            At most <n> Monte Carlo steps will be performed for a single input bit.
        '''
        super().__init__(datastream)
        self.magnitude = magnitude
        self.n = n # The max. number of Monte Carlo steps performed every time self.input_single(mm) is called
        self.frequency = frequency

    def input_single(self, mm: Magnets, value=None):
        if value is None: value = self.datastream.get_next()

        MCsteps0, t0 = mm.MCsteps, mm.t
        AFM_mask = (((mm.ixx + mm.iyy) % 2)*2 - 1).astype(float) # simple checkerboard pattern of 1 and -1
        mm.get_energy('Zeeman').set_field(magnitude=AFM_mask*(2*value-1)*self.magnitude)
        while (progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency)) < 1:
            if self.frequency:
                mm.update(t_max=min(1-progress, 1)/self.frequency)
            else:
                mm.update() # No time
        return value


class OOPSquareChessOutputReader(OutputReader):
    def __init__(self, nx, ny, mm=None):
        ''' This is a RegionalOutputReader optimized for OOP_Square ASI.
            Since OOP_Square has two degenerate AFM ground states, the averaging takes place
            using an AFM mask, such that they can be distinguished and domain walls can be identified.
            @param nx [int]: number of averaging bins in the x-direction.
            @param ny [int]: number of averaging bins in the y-direction.
            @param mm [hotspice.Magnets] (None): if specified, this OutputReader automatically calls self.configure_for(mm).
                Otherwise, the user will have to call configure_for() separately after initializing the class instance.
        '''
        self.nx, self.ny, self._n = nx, ny, nx*ny
        if mm is not None: self.configure_for(mm)

    def configure_for(self, mm: Magnets):
        if not isinstance(mm, (expected_type := OOP_Square)):
            if mm.in_plane:
                raise TypeError(f"{type(self).__name__} only works on out-of-plane ASI, but received {type(mm).__name__}.")
            warnings.warn(f"{type(self).__name__} expects {expected_type.__name__} ASI, but received {type(mm).__name__}.", stacklevel=2)
        self.mm = mm
        region_x = xp.tile(xp.linspace(0, self.nx*(1-1e-15), mm.nx, dtype=int), (mm.ny, 1)) # -1e-15 to have self.nx-1 as final value instead of self.nx
        region_y = xp.tile(xp.linspace(0, self.ny*(1-1e-15), mm.ny, dtype=int), (mm.nx, 1)).T # same as x but transposed and ny <-> nx
        self.region = region_x + self.nx*region_y # 2D (mm.nx x mm.ny) array representing which (1D-indexed) region each magnet belongs to

        self.normalization_factor = xp.zeros(self.n, dtype=int) # Number of magnets in each region
        # Determine the number of magnets in each region
        for regionindex in range(self.n):
            self.normalization_factor[regionindex] = xp.sum(mm.occupation[self.region == regionindex])
        self.AFM_mask = ((self.mm.ixx + self.mm.iyy) % 2)*2 - 1 # simple checkerboard pattern of 1 and -1
        self.state = xp.zeros(self.n, dtype=float)
    
    def read_state(self, mm: Magnets = None, m=None) -> xp.ndarray:
        ''' Returns a 1D array representing the AFM state of the spin ice. '''
        if mm is not None: self.configure_for(mm) # If mm is not specified, we suppose configure_for() already happened
        if m is None: m = self.mm.m # If m is specified, it takes precendence over mm regardless of whether mm was specified too

        # TODO: I suspect that this for loop is quite slow if self.n is big; should profile this and possibly vectorize.
        masked_m = m*self.AFM_mask
        for regionindex in range(self.n):
            self.state[regionindex] = xp.sum(masked_m[self.region == regionindex])
        self.state /= self.normalization_factor # To get in range [-1, 1]
        return self.state # [unitless]
    
    @property
    def n(self):
        return self._n
