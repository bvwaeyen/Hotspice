import math
import warnings

import numpy as np

from abc import ABC, abstractmethod
from typing import Literal

from .ASI import IP_ASI, OOP_Square
from .core import Magnets, ZeemanEnergy
from .plottools import show_m
from .utils import log, lower_than
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
    
    def as_bits(self, integer: int, endianness: Literal['little', 'big'] = 'little'):
        raise NotImplementedError("This datastream can not be converted to a binary stream.")

    @property
    def is_binary(self): return False

    @property
    def dtype(self): return float

class BinaryDatastream(Datastream):
    """ Just an alias for Datastream: a normal Datastream can contain floats, while this only yields 0 or 1. """
    pass

    @property
    def is_binary(self): return True

    @property
    def dtype(self): return bool


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
    def __init__(self, mm: Magnets = None):
        self.mm = mm
        if mm is not None: self.configure_for(mm)

    def configure_for(self, mm: Magnets):
        """ Subclassing this method is optional. When called, some properties of this OutputReader object
            are initialized, which depend on the Magnets object <mm>.
        """
        self.mm = mm

    @abstractmethod
    def read_state(self, mm: Magnets = None) -> xp.ndarray:
        """ Reads the current state of the <mm> simulation in some way.
            Sets <self.state>, and returns <self.state>.
        """
        if (mm is not None) and (mm is not self.mm): self.configure_for(mm)

    @property
    def n(self) -> int:
        """ The number of output values when reading a given state. """
        return self.state.size
    
    @property
    def node_coords(self) -> xp.ndarray: # Needed for Memory Capacity in TaskAgnosticExperiment, as well as squinting
        """ An Nx2 array, where each row contains representative coordinates for all output values
            in self.read_state(). The first column contains x-coordinates, the second contains y.
        """
        return self._node_coords


######## Below are subclasses of the superclasses above
# TODO: class FileDatastream(Datastream) which reads bits from a file? Can use package 'bitstring' for this.
# TODO: class SemiRepeatingDatastream(Datastream) which has first <n> random bits and then <m> bits which are the same for all runs
class RandomBinaryDatastream(BinaryDatastream):
    def __init__(self, p0=.5):
        """ Generates random bits, with <p0> probability of getting 0, and 1-<p0> probability of getting 1.
            @param p0 [float] (0.5): the probability of 0 when generating a random bit.
        """
        self.p0 = p0
        super().__init__()

    def get_next(self, n=1) -> xp.ndarray:
        return self.rng.random(size=(n,)) >= self.p0

class RandomUniformByteDatastream(Datastream):
    """ Generates random integers uniformly distributed from 0 up to and including 2**<num_bytes> - 1. """
    def __init__(self, num_bits: int = 8):
        if not isinstance(num_bits, int): raise ValueError("Number of bits per 'byte' in RandomUniformByteDatastream must be of type <int>.")
        self.num_bits = num_bits
        self._max = 2**self.num_bits
        super().__init__()

    def get_next(self, n=1) -> xp.ndarray:
        return self.rng.integers(0, self._max, size=(n,))

    def as_bits(self, integer: int, endianness: Literal['little', 'big'] = 'little') -> xp.ndarray:
        """ For our RC purposes, little-endian is preferred, because nearby integers will have similar final bits.
            This should allow for easier training for the estimators in TaskAgnosticExperiment.
        """
        bitstring = bin(integer)[2:].zfill(self.num_bits) # Slice from 2 to remove '0b', zfill to left-pad with zeros 
        if endianness == 'little': bitstring = bitstring[::-1]
        return xp.asarray([int(bit) for bit in bitstring])

class RandomUniformDatastream(Datastream):
    def __init__(self, low=0, high=1):
        """ Generates uniform random floats between <low=0> and <high=1>. """
        self.low, self.high = low, high
        super().__init__()

    def get_next(self, n=1) -> xp.ndarray:
        return (self.high - self.low)*self.rng.random(size=(n,)) + self.low


class FieldInputter(Inputter):
    def __init__(self, datastream: Datastream, magnitude=1, angle=0, n=2, sine=False, frequency=1, half_period=True):
        """ Applies an external field at <angle> rad, whose magnitude is <magnitude>*<datastream.get_next()>.
            <frequency> specifies the frequency [Hz] at which bits are being input to the system, if it is nonzero.
            If <sine> is True, the field magnitude follows a full sinusoidal period for each input bit.
            If <half_period> is True, only the first half-period ⏜ of the sine is used, otherwise the full ∿ period.
            At most <n> Monte Carlo steps will be performed.
        """
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
        Zeeman: ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None: mm.add_energy(Zeeman := ZeemanEnergy(0, 0))

        Zeeman.set_field(angle=(self.angle if mm.in_plane else None)) # set angle only once
        MCsteps0, t0 = mm.MCsteps, mm.t
        if self.sine: # Change magnitude sinusoidally
            while lower_than(progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency), 1):
                Zeeman.set_field(magnitude=self.magnitude*value*math.sin(progress*math.pi*(2-self.half_period)))
                mm.update(t_max=min(1-progress, 0.05)/self.frequency) # At least 20 (=1/0.05) steps to sample the sine decently
        else: # Use constant magnitude
            Zeeman.set_field(magnitude=self.magnitude*value)
            mm.progress(t_max=1/self.frequency, MCsteps_max=self.n) # No sine, so no 20 steps per wavelength needed here
        return value


class FieldInputterBinary(FieldInputter):
    def __init__(self, datastream: Datastream, magnitudes: tuple = (.8, 1), **kwargs):
        """ Exactly the same as FieldInputter, but if <datastream> yields 0 then a field with
            magnitude <magnitudes[0]> is applied, otherwise <magnitudes[1]>.
            Using <sine=<frequency[Hz]>> yields behavior as in "Computation in artificial spin ice" by Jensen et al.
            Differing attributes compared to FieldInputter:
                <self.magnitude> is set to <magnitudes[1]>
                <self.magnitude_ratio> is used to store the ratio <magnitudes[0]/magnitudes[1]>.
        """
        self.magnitude_ratio = magnitudes[0]/magnitudes[1]
        super().__init__(datastream, magnitude=magnitudes[1], half_period=False, **kwargs)

    def input_single(self, mm: Magnets, value=None):
        if value is None: value = self.datastream.get_next()
        value = self.magnitude_ratio if value == 0 else 1
        return super().input_single(mm, value=value)


class PerpFieldInputter(FieldInputter):
    def __init__(self, datastream: BinaryDatastream, magnitude=1, angle=0, n=2, relax=True, frequency=1, **kwargs):
        """ Applies an external field, whose angle depends on the bit that is being applied:
            the bit '0' corresponds to an angle of <phi> radians, the bit '1' to <phi>+pi/2 radians.
            Also works for continuous numbers, resulting in intermediate angles, but this is not the intended use.
            For more information about the kwargs, see FieldInputter.
        """
        self.relax = relax
        super().__init__(datastream, magnitude=magnitude, angle=angle, n=n, frequency=frequency)


    def input_single(self, mm: IP_ASI, value=None):
        if (self.sine or self.frequency) and mm.params.UPDATE_SCHEME != 'Néel':
            raise AttributeError("Can not use temporal sine=True or nonzero frequency if UPDATE_SCHEME != 'Néel'.")
        if not mm.in_plane:
            raise AttributeError("Can not use PerpFieldInputter on an out-of-plane ASI.")
        if value is None: value = self.datastream.get_next()

        Zeeman: ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None: mm.add_energy(Zeeman := ZeemanEnergy(0, 0))

        angle = self.angle + value*math.pi/2
        # pos and neg field
        if self.relax:
            for mag in [self.magnitude, -self.magnitude]:
                Zeeman.set_field(magnitude=mag, angle=angle)
                mm.minimize()
            # log(f"Performed {mm.MCsteps - MCsteps0} MC steps.")
        else:
            Zeeman.set_field(magnitude=self.magnitude, angle=angle)
            mm.progress(t_max=0.5/self.frequency, MCsteps_max=self.n)
            Zeeman.set_field(magnitude=-self.magnitude, angle=angle)
            mm.progress(t_max=0.5/self.frequency, MCsteps_max=self.n)
        return value


    def input_single_generalized(self, mm: IP_ASI, value=None):
        if (self.sine or self.frequency) and mm.params.UPDATE_SCHEME != 'Néel':
            raise AttributeError("Can not use temporal sine=True or nonzero frequency if UPDATE_SCHEME != 'Néel'.")
        if not mm.in_plane:
            raise AttributeError("Can not use PerpFieldInputter on an out-of-plane ASI.")
        if value is None: value = self.datastream.get_next()

        Zeeman: ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None:
            warnings.warn("No Zeeman energy associated with the spin ice was found. Using zero field by default.")
            mm.add_energy(Zeeman := ZeemanEnergy(0, 0))

        angle = self.angle + value*math.pi/2
        MCsteps0, t0 = mm.MCsteps, mm.t
        if self.sine: # Change magnitude sinusoidally
            while lower_than(progress := max((mm.MCsteps - MCsteps0)/self.n, (mm.t - t0)*self.frequency), 1):
                Zeeman.set_field(magnitude=self.magnitude*math.sin(progress*math.pi*(2-self.half_period)), angle=angle)
                mm.update(t_max=min(1-progress, 0.125)/self.frequency) # At least 8 (=1/0.125) steps per sine-period
        else: # Use constant magnitude
            Zeeman.set_field(magnitude=self.magnitude, angle=angle)
            mm.progress(t_max=1/self.frequency, MCsteps_max=self.n)
        return value


class FullOutputReader(OutputReader):
    # TODO: add compression to this, so it can be used without storage-worry as default when storing in a dataframe (then afterwards we can re-load this into mm and read it with another kind of outputreader)
    def __init__(self, mm: Magnets = None):
        """ Reads the full state of the ASI. Ignores empty cells.
            By reading the full state, we allow for squinting afterwards. How exactly to do this, is a # TODO point.
            @param mm [hotspice.Magnets] (None): if specified, this OutputReader automatically calls self.configure_for(mm).
        """
        super().__init__(mm)
    
    def configure_for(self, mm: Magnets):
        self.mm = mm
        self.indices = xp.nonzero(self.mm.occupation)
        self.state = self.mm.m[self.indices]
        self._node_coords = xp.asarray([mm.xx[self.indices], mm.yy[self.indices]]).T
    
    def read_state(self, mm: Magnets, m: xp.ndarray = None) -> xp.ndarray:
        super().read_state(mm)
        if m is None: m = self.mm.m
        self.state = m[self.indices]
        return self.state


class RegionalOutputReader(OutputReader):
    def __init__(self, nx: int, ny: int, mm: Magnets = None):
        """ Reads the current state of the ASI with a certain level of detail.
            @param nx [int]: number of averaging bins in the x-direction.
            @param ny [int]: number of averaging bins in the y-direction.
            @param mm [hotspice.Magnets] (None): if specified, this OutputReader automatically calls self.configure_for(mm).
        """
        self.nx, self.ny = nx, ny
        self.n_regions = self.nx*self.ny
        super().__init__(mm)

    def configure_for(self, mm: Magnets):
        self.mm = mm
        self.state = xp.zeros(self.n_regions*2) if mm.in_plane else xp.zeros(self.n_regions)

        region_x = xp.tile(xp.linspace(0, self.nx*(1-1e-15), mm.nx, dtype=int), (mm.ny, 1)) # -1e-15 to have self.nx-1 as final value instead of self.nx
        region_y = xp.tile(xp.linspace(0, self.ny*(1-1e-15), mm.ny, dtype=int), (mm.nx, 1)).T # same as x but transposed and ny <-> nx
        self.region = region_x + self.nx*region_y # 2D (mm.nx x mm.ny) array representing which (1D-indexed) region each magnet belongs to

        self.normalization_factor = xp.zeros(self.n) # Number of magnets in each region
        self._node_coords = xp.zeros((self.n, 2))
        for i in range(self.n): # Pre-calculate some quantities for each region
            regionindex = i % self.n_regions
            self.normalization_factor[i] = xp.sum(mm.occupation[self.region == regionindex]*mm.moment[self.region == regionindex])
            self._node_coords[i,0] = xp.mean(mm.xx[self.region == regionindex])
            self._node_coords[i,1] = xp.mean(mm.yy[self.region == regionindex])

    def read_state(self, mm: Magnets = None, m: xp.ndarray = None) -> xp.ndarray:
        """ Returns a 1D array representing the state of the spin ice. """
        super().read_state(mm)
        if self.mm is None: raise AttributeError("OutputReader has not yet been initialized with a Magnets object.")
        m = (self.mm.m if m is None else m)*self.mm.moment # If m is specified, it takes precendence over mm.m

        if self.mm.in_plane:
            m_x = m*self.mm.orientation[:,:,0]
            m_y = m*self.mm.orientation[:,:,1]

        for regionindex in range(self.n_regions):
            if self.mm.in_plane:
                self.state[regionindex] = xp.sum(m_x[self.region == regionindex])
                self.state[regionindex + self.n_regions] = xp.sum(m_y[self.region == regionindex])
            else:
                self.state[regionindex] = xp.sum(m[self.region == regionindex])
        self.state /= self.normalization_factor # To get in range [-1, 1]
        return self.state # [Am²]


class OOPSquareChessFieldInputter(Inputter):
    def __init__(self, datastream: Datastream, magnitude: float = 1, n=4, frequency=1):
        """ Applies an external stimulus in a checkerboard pattern. This is modelled as an external
            field for each magnet, with magnitude <magnitude>*<datastream.get_next()>.
            A checkerboard pattern is used to break symmetry between the two ground states of OOP_Square ASI.
            <frequency> specifies the frequency [Hz] at which bits are being applied to the system, if it is nonzero.
            At most <n> Monte Carlo steps will be performed for a single input bit.
        """
        super().__init__(datastream)
        self.magnitude = magnitude
        self.n = n # The max. number of Monte Carlo steps performed every time self.input_single(mm) is called
        self.frequency = frequency

    def input_single(self, mm: OOP_Square, value=None):
        if value is None: value = self.datastream.get_next(n=1)
        input_sequence = xp.asarray(value).reshape(-1) if self.datastream.is_binary else self.datastream.as_bits(int(value), endianness='little')
        # TODO: is there a way to do this int-to-bits etc. cleaner? Because this kind of method will be used more often if this succeeds... Also, the type of 'value' is getting less well defined by the day...
        Zeeman: ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None: mm.add_energy(Zeeman := ZeemanEnergy(0, 0))

        AFM_mask = (((mm.ixx + mm.iyy) % 2)*2 - 1).astype(float) # simple checkerboard pattern of 1 and -1
        for inbit in input_sequence:
            Zeeman.set_field(magnitude=AFM_mask*(2*inbit-1)*self.magnitude)
            mm.progress(t_max=1/self.frequency, MCsteps_max=self.n)
        return value


class OOPSquareChessOutputReader(OutputReader):
    def __init__(self, nx: int, ny: int, mm: OOP_Square = None):
        """ This is a RegionalOutputReader optimized for OOP_Square ASI.
            Since OOP_Square has two degenerate AFM ground states, the averaging takes place
            using an AFM mask, such that they can be distinguished and domain walls can be identified.
            @param nx [int]: number of averaging bins in the x-direction.
            @param ny [int]: number of averaging bins in the y-direction.
            @param mm [hotspice.ASI.OOP_Square] (None): if specified, this OutputReader automatically calls self.configure_for(mm).
                Otherwise, the user will have to call configure_for() separately after initializing the class instance.
        """
        self.nx, self.ny = nx, ny
        super().__init__(mm)

    def configure_for(self, mm: OOP_Square):
        if not isinstance(mm, (expected_type := OOP_Square)):
            if mm.in_plane:
                raise TypeError(f"{type(self).__name__} only works on out-of-plane ASI, but received {type(mm).__name__}.")
            warnings.warn(f"{type(self).__name__} expects {expected_type.__name__} ASI, but received {type(mm).__name__}.", stacklevel=2)
        self.mm = mm
        self.state = xp.zeros(self.nx*self.ny, dtype=float)
        self.AFM_mask = ((self.mm.ixx + self.mm.iyy) % 2)*2 - 1 # simple checkerboard pattern of 1 and -1

        region_x = xp.tile(xp.linspace(0, self.nx*(1-1e-15), mm.nx, dtype=int), (mm.ny, 1)) # -1e-15 to have self.nx-1 as final value instead of self.nx
        region_y = xp.tile(xp.linspace(0, self.ny*(1-1e-15), mm.ny, dtype=int), (mm.nx, 1)).T # same as x but transposed and ny <-> nx
        self.region = region_x + self.nx*region_y # 2D (mm.nx x mm.ny) array representing which (1D-indexed) region each magnet belongs to

        self.normalization_factor = xp.zeros(self.n, dtype=int) # Number of magnets in each region
        self._node_coords = xp.zeros((self.n, 2))
        # Determine the number of magnets in each region
        for regionindex in range(self.n):
            self.normalization_factor[regionindex] = xp.sum(mm.occupation[self.region == regionindex])
            self._node_coords[regionindex,0] = xp.mean(mm.xx[self.region == regionindex])
            self._node_coords[regionindex,1] = xp.mean(mm.yy[self.region == regionindex])
    
    def read_state(self, mm: OOP_Square = None, m: xp.ndarray = None) -> xp.ndarray:
        """ Returns a 1D array representing the AFM state of the spin ice. """
        super().read_state(mm)
        masked_m = self.AFM_mask*(self.mm.m if m is None else m)
        for regionindex in range(self.n): # TODO: I suspect that this for loop is quite slow if self.n is big; should profile this and possibly vectorize.
            self.state[regionindex] = xp.sum(masked_m[self.region == regionindex])
        self.state /= self.normalization_factor # To get in range [-1, 1]
        return self.state # [unitless]


class OOPSquareClockwiseInputter(Inputter):
    def __init__(self, datastream: Datastream, magnitude: float = 1, n=4, frequency=1):
        """ Applies an external stimulus in a clockwise manner in 4 substeps.
            In each substep, one quarter of all magnets is positively biased, while another quarter is negatively biased,
            and the other half is not stimulated at all. TODO: complete this description
            <frequency> specifies the frequency [Hz] at which bits are being input to the system, if it is nonzero.
            At most <n> Monte Carlo steps will be performed for a single input bit.
            NOTE: This inputter will probalby work best with Néel update scheme.
        """
        super().__init__(datastream)
        self.magnitude = magnitude
        self.n = n # The max. number of Monte Carlo steps performed every time self.input_single(mm) is called
        self.frequency = frequency # The min. frequency the bits are applied at, if the time is known (i.e. Néel scheme is used)

    def input_single(self, mm: OOP_Square, value=None):
        """ Performs an input corresponding to <value> (generated using <self.datastream>)
            into the <mm> simulation, and returns this <value>.
        """
        if value is None: value = self.datastream.get_next()
        Zeeman: ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None: mm.add_energy(Zeeman := ZeemanEnergy(0, 0))

        combinations = [(0, 0), (1, 0), (1, 1), (0, 1)] if value else [(0, 0), (0, 1), (1, 1), (1, 0)] # This is (the only place) where <value> comes in
        # combinations = [(0, 0), (1, 0), (0, 1), (1, 1)] if value else [(0, 0), (0, 1), (1, 0), (1, 1)] # This is (the only place) where <value> comes in
        for offset_x, offset_y in combinations: # Determines which quarter of the magnets is stimulated at each substep
            mask_positive = xp.logical_and((mm.ixx % 2) == offset_x, (mm.iyy % 2) == offset_y).astype(float) # need astype float for good multiplication with self.magnitude
            mask_negative = xp.logical_and((mm.ixx % 2) != offset_x, (mm.iyy % 2) == offset_y).astype(float) # TODO: verify this position based on SOT physics
            # magnitude_local = self.magnitude*(mask_positive - mask_negative)
            magnitude_local = self.magnitude*mask_positive
            magnitude_local = self.magnitude*(mask_positive if value else -mask_negative)
            Zeeman.set_field(magnitude=magnitude_local)
            mm.progress(t_max=1/self.frequency, MCsteps_max=self.n)
        return value


class OOPSquareChessStepsInputter(Inputter):
    def __init__(self, datastream: Datastream, magnitude: float = 1, n=4, frequency=1):
        """ Applies an external stimulus in a checkerboard pattern. This is modelled as an external
            field for each magnet, with magnitude <magnitude>*<datastream.get_next()>.
            NOTE: the difference between this and an OOPSquareChessFieldInputter is that this inputter
                  performs the input in 2 separate steps, where in each step half of the checkerboard pattern
                  is stimulated, which should prevent domain walls from propagating all the way to saturation.
            A checkerboard pattern is used to break symmetry between the two ground states of OOP_Square ASI.
            <frequency> specifies the frequency [Hz] at which bits are being applied to the system, if it is nonzero.
            At most <n> Monte Carlo steps will be performed for a single input bit.
        """
        super().__init__(datastream)
        self.magnitude = magnitude
        self.n = n # The max. number of Monte Carlo steps performed every time self.input_single(mm) is called
        self.frequency = frequency

    def input_single(self, mm: OOP_Square, value=None):
        if value is None: value = self.datastream.get_next(n=1)
        input_sequence = xp.asarray(value).reshape(-1) if self.datastream.is_binary else self.datastream.as_bits(int(value), endianness='little')
        # TODO: is there a way to do this int-to-bits etc. cleaner? Because this kind of method will be used more often if this succeeds... Also, the type of 'value' is getting less well defined by the day...
        Zeeman: ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None: mm.add_energy(Zeeman := ZeemanEnergy(0, 0))

        AFM_mask = ((mm.ixx + mm.iyy) % 2).astype(float) # Need astype float for correct multiplication
        AFM_mask_step1 = AFM_mask*self.magnitude # 0 and 1 on a checkerboard pattern
        AFM_mask_step2 = (AFM_mask - 1)*self.magnitude # -1 and 0 checkerboard pattern, with 0s on other spots w.r.t. step 1
        for inbit in input_sequence:
            sign = 2*inbit - 1
            Zeeman.set_field(magnitude=AFM_mask_step1*sign)
            mm.progress(t_max=1/self.frequency, MCsteps_max=self.n)
            Zeeman.set_field(magnitude=AFM_mask_step2*sign)
            mm.progress(t_max=1/self.frequency, MCsteps_max=self.n)
        return value
