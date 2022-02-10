import math
import warnings

import cupy as cp
import numpy as np

from abc import ABC, abstractmethod

from .core import Magnets


class ASI(ABC, Magnets):
    @abstractmethod
    def _set_m(self, pattern):
        ''' Directly sets <self.m>, depending on <pattern>. Usually, <pattern> is "uniform", "AFM" or "random". '''
        pass

    @abstractmethod
    def _get_unitcell(self):
        ''' Returns a tuple containing the number of cells in a unit cell along both the x- and y-axis. '''
        pass

    @abstractmethod
    def _get_occupation(self):
        ''' Returns a 2D CuPy array which contains 1 at the cells which are occupied by a magnet, and 0 elsewhere. '''
        pass

    @abstractmethod
    def _get_appropriate_avg(self):
        ''' Returns the most appropriate averaging mask for a given type of ASI. '''
        pass

    @abstractmethod
    def _get_AFMmask(self):
        ''' Returns the (normalized) mask used to determine how anti-ferromagnetic the magnetization profile is. '''
        pass

    @abstractmethod
    def _get_nearest_neighbors(self):
        ''' Returns a small mask with the magnet at the center, and 1 at the positions of its nearest neighbors (elsewhere 0). '''
        pass
    
    @abstractmethod
    def _get_groundstate(self):
        ''' Returns one of either strings: 'uniform', 'AFM', 'random'.
            Use 'random' if the ground state is more complex than uniform or AFM.
        '''
        pass

class FullASI(ASI):
    def __init__(self, n, a, ny=None, **kwargs):
        ''' Out-of-plane ASI in a square arrangement. '''
        self.a = a
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=False, **kwargs)
    
    def _set_m(self, pattern):
        if pattern == 'uniform':
            self.m = cp.ones_like(self.xx)
        elif pattern == 'AFM':
            self.m = ((self.ixx - self.iyy) % 2)*2 - 1
        else:
            self.m = cp.random.randint(0, 2, size=cp.shape(self.xx))*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)
    
    def _get_unitcell(self):
        return (1, 1)

    def _get_occupation(self):
        return cp.ones_like(self.xx)

    def _get_appropriate_avg(self):
        return 'point'
    
    def _get_plotting_params(self):
        return {
            'quiverscale': 1,
            'max_mean_magnitude': 1
        }

    def _get_AFMmask(self):
        return cp.array([[1, -1], [-1, 1]], dtype='float')/4 # TODO: this might need a change?
    
    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    def _get_groundstate(self):
        return 'AFM'


class IsingASI(ASI):
    def __init__(self, n, a, ny=None, **kwargs):
        ''' In-plane ASI with all spins on a square grid, all pointing in the same direction. '''
        self.a = a
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)
    
    def _set_m(self, pattern):
        if pattern == 'uniform':
            self.m = cp.ones_like(self.xx)
        elif pattern == 'AFM':
            self.m = (self.iyy % 2)*2 - 1
        else:
            self.m = cp.random.randint(0, 2, size=cp.shape(self.xx))*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)
        
    def _set_orientation(self, angle=0.):
        self.orientation = np.zeros(np.shape(self.xx) + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        self.orientation[:,:,0] = math.cos(angle)
        self.orientation[:,:,1] = math.sin(angle)
        self.orientation = cp.asarray(self.orientation)
    
    def _get_unitcell(self):
        return (1, 1)

    def _get_occupation(self):
        return cp.ones_like(self.xx)

    def _get_appropriate_avg(self):
        return 'point'
    
    def _get_plotting_params(self):
        return {
            'quiverscale': 1.1,
            'max_mean_magnitude': 1
        }

    def _get_AFMmask(self):
        return cp.array([[1, 1], [-1, -1]])/4
    
    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    def _get_groundstate(self):
        return 'uniform'


class SquareASI(ASI):
    def __init__(self, n, a, ny=None, **kwargs):
        ''' In-plane ASI with the spins placed on, and oriented along, the edges of squares. '''
        self.a = a
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a/2
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)
    
    def _set_m(self, pattern):
        if pattern == 'uniform':
            self.m = cp.ones_like(self.xx)
        elif pattern == 'AFM':
            self.m = ((self.ixx - self.iyy)//2 % 2)*2 - 1
        else:
            self.m = cp.random.randint(0, 2, size=cp.shape(self.xx))*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)
    
    def _set_orientation(self, angle=0):
        self.orientation = np.zeros(np.shape(self.xx) + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        occupation = self.occupation.get()
        iyy = self.iyy.get()
        self.orientation[iyy % 2 == 0,0] = math.cos(angle)
        self.orientation[iyy % 2 == 0,1] = math.sin(angle)
        self.orientation[iyy % 2 == 1,0] = math.cos(angle + math.pi/2)
        self.orientation[iyy % 2 == 1,1] = math.sin(angle + math.pi/2)
        self.orientation[occupation == 0,0] = 0
        self.orientation[occupation == 0,1] = 0
        self.orientation = cp.asarray(self.orientation)

    def _get_unitcell(self):
        return (2, 2)

    def _get_occupation(self):
        return (self.ixx + self.iyy) % 2 == 1

    def _get_appropriate_avg(self):
        return 'cross'
    
    def _get_plotting_params(self):
        # examples of this include quiverscale, magnitude scale to normalize value in hsv, ...
        return {
            'quiverscale': 0.7,
            'max_mean_magnitude': 1/math.sqrt(2)
        }

    def _get_AFMmask(self):
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4
    
    def _get_nearest_neighbors(self):
        return cp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    
    def _get_groundstate(self):
        return 'AFM'


class PinwheelASI(SquareASI):
    def __init__(self, n, a, ny=None, **kwargs):
        ''' In-plane ASI similar to SquareASI, but all spins rotated by 45°, hence forming a pinwheel geometry. '''
        super().__init__(n, a, ny=ny, **kwargs)
        
    def _set_orientation(self, angle=0.):
        super()._set_orientation(angle + math.pi/4)
    
    def _get_groundstate(self):
        return 'uniform'


class KagomeASI(ASI):
    def __init__(self, n, a, ny=None, **kwargs):
        ''' In-plane ASI with all spins placed on, and oriented along, the edges of hexagons. '''
        self.a = a
        self.nx = n
        if ny is None:
            self.ny = int(self.nx/math.sqrt(3))//4*4
            if 'PBC' in kwargs:
                if not kwargs['PBC']:
                    self.ny -= 1
        else:
            self.ny = ny
        self.dx = a/4
        self.dy = math.sqrt(3)*self.dx
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)
    
    def _set_m(self, pattern):
        if pattern == 'uniform':
            self.m = cp.ones_like(self.xx)
            self.m[(self.ixx - self.iyy) % 4 == 1] = -1
        elif pattern == 'AFM':
            self.m = cp.ones_like(self.xx)
            self.m[(self.ixx + self.iyy) % 4 == 3] = -1
        else:
            self.m = cp.random.randint(0, 2, size=cp.shape(self.xx))*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)
        
    def _set_orientation(self, angle=0.):
        self.orientation = np.zeros(np.shape(self.xx) + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        occupation = self.occupation.get()
        self.orientation[:,:,0] = math.cos(angle + math.pi/2)
        self.orientation[:,:,1] = math.sin(angle + math.pi/2)
        self.orientation[cp.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1).get(),0] = math.cos(angle - math.pi/6)
        self.orientation[cp.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1).get(),1] = math.sin(angle - math.pi/6)
        self.orientation[cp.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1).get(),0] = math.cos(angle + math.pi/6)
        self.orientation[cp.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1).get(),1] = math.sin(angle + math.pi/6)
        self.orientation[occupation == 0,0] = 0
        self.orientation[occupation == 0,1] = 0
        self.orientation = cp.asarray(self.orientation)
    
    def _get_unitcell(self):
        return (4, 4)

    def _get_occupation(self):
        occupation = cp.zeros_like(self.xx)
        occupation[(self.ixx + self.iyy) % 4 == 1] = 1 # One bunch of diagonals \
        occupation[(self.ixx - self.iyy) % 4 == 3] = 1 # Other bunch of diagonals /
        return occupation

    def _get_appropriate_avg(self):
        return 'hexagon'
    
    def _get_plotting_params(self):
        return {
            'quiverscale': 0.7,
            'max_mean_magnitude': 2/3
        }

    def _get_AFMmask(self):
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])
    
    def _get_groundstate(self):
        return 'uniform'


class TriangleASI(KagomeASI):
    def __init__(self, n, a, ny=None, **kwargs):
        ''' In-plane ASI with all spins placed on, and oriented along, the edges of triangles. '''
        super().__init__(n, a, ny=ny, **kwargs)

    def _set_orientation(self, angle=0.):
        super()._set_orientation(angle + math.pi/2)
    
    def _get_appropriate_avg(self):
        return 'triangle'
    
    def _get_plotting_params(self):
        return {
            'quiverscale': 0.5,
            'max_mean_magnitude': 2/3
        }
    
    def _get_groundstate(self):
        return 'AFM'
