import math
# import warnings

import cupy as cp
import numpy as np

from abc import ABC, abstractmethod
# from cupyx.scipy import signal

from .core import Magnets
from .plottools import show_m


class DataStream(ABC):
    @abstractmethod
    def get_next(self, n=1):
        """ Calling this method returns a CuPy array containing exactly <n> elements, which are either 0 or 1 (int!). """


class Inputter(ABC):
    def __init__(self, datastream: DataStream):
        self.datastream = datastream

    @abstractmethod
    def input_bit(self, mm: Magnets, bit=None):
        """ Inputs a bit (generated using <self.datastream>) into the <mm> simulation. """
        if bit is None: bit = self.datastream.get_next()


class OutputReader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read_state(self, mm: Magnets):
        """ Reads the current state of the <mm> simulation in some way (the exact manner should be
            implemented through subclassing), 
        """
    
    def configure_for(self, mm: Magnets):
        """ Subclassing this method is optional. When called, some properties of this OutputReader object
            are initialized, which depend on the Magnets object <mm>.
        """
        self.mm = mm
    
    @property
    @abstractmethod
    def n(self):
        ''' The number of output bits when reading a given state. '''


######## Below are subclasses of the superclasses above
# TODO: class FileDataStream(DataStream) which reads bits from a file? Can use package 'bitstring' for this.
# TODO: class SemiRepeatingDataStream(DataStream) which has first <n> random bits and then <m> bits which are the same for all runs
class RandomDataStream(DataStream):
    def __init__(self, p0=.5):
        ''' Generates random bits, with <p0> probability of getting 0, and 1-<p0> probability of getting 1.
            @param p0 [float] (0.5): the probability of 0 when generating a random bit.
        '''
        self.p0 = p0

    def get_next(self, n=1):
        return cp.where(cp.random.uniform(size=(n,)) < self.p0, 0, 1)


class PerpFieldInputter(Inputter):
    def __init__(self, datastream: DataStream, magnitude=1, phi=0, n=2):
        ''' Applies an external field, whose angle depends on the bit that is being applied:
            the bit '0' corresponds to an angle of <phi> radians, the bit '1' to <phi>+pi/2 radians.
        '''
        super().__init__(datastream)
        self._phi = phi
        self._magnitude = magnitude
        self.n = n # The number of steps (as a multiple of mm.n) done every time self.input_bit(mm) is called
        # TODO: decide whether to use n, or t, or something else?
    
    @property
    def phi(self):
        return self._phi
    @phi.setter
    def phi(self, value):
        self._phi = value % (2*math.pi)
    
    @property
    def magnitude(self):
        return self._magnitude
    @magnitude.setter
    def magnitude(self, value):
        self._magnitude = value
    
    def input_bit(self, mm: Magnets, bit=None):
        if bit is None: bit = self.datastream.get_next()
        angle = self.phi + bit*math.pi/2
        mm.energy_Zeeman_setField(magnitude=self.magnitude, angle=angle)
        for _ in range(int(self.n*mm.n)): # TODO: make this r-dependent (r in Magnets.select())
            mm.update()


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
        if mm.in_plane:
            self.state = cp.zeros((self.nx, self.ny, 2))
        else:
            self.state = cp.zeros((self.nx, self.ny))

    def read_state(self, m=None):
        if m is None: m = self.mm.m
        if self.mm.in_plane:
            m_x = m*self.mm.orientation[:,:,0]
            m_y = m*self.mm.orientation[:,:,1]
            for i, _ in np.ndenumerate(self.grid): # CuPy has no ndenumerate, so use NumPy then
                self.state[i[0], i[1], 0] = cp.mean(m_x[np.logical_and(self.region_x == i[0], self.region_y == i[1])]) # Average m_x
                self.state[i[0], i[1], 1] = cp.mean(m_y[np.logical_and(self.region_x == i[0], self.region_y == i[1])]) # Average m_y
        else:
            for i, _ in np.ndenumerate(self.grid):
                self.state[i[0], i[1]] = cp.mean(m[np.logical_and(self.region_x == i[0], self.region_y == i[1])])
        return self.state
