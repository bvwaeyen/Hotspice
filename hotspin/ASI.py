import math
import warnings

import cupy as cp

from abc import ABC, abstractmethod

from .core import Magnets


class ASI(ABC, Magnets):
    @abstractmethod
    def _set_m(self, pattern: str):
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
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' Out-of-plane ASI in a square arrangement. '''
        self.a = a # [m] The distance between two nearest neighboring spins
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=False, **kwargs)

    def _set_m(self, pattern: str, angle: float = 0):
        if pattern == 'uniform': # PYTHONUPDATE_3.10: use structural pattern matching
            self.m = cp.ones_like(self.xx)*(2*(math.cos(angle) >= 0) - 1)
        elif pattern == 'AFM':
            self.m = ((self.ixx - self.iyy) % 2)*2 - 1
        else:
            self.m = cp.random.randint(0, 2, size=self.xx.shape)*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)

    def _get_unitcell(self):
        return (1, 1)

    def _get_occupation(self):
        return cp.ones_like(self.xx)

    def _get_appropriate_avg(self):
        return 'point'

    def _get_AFMmask(self):
        return cp.array([[0, -1], [-1, 2]], dtype='float')/4 # Possible situations: ▚/▞ -> 1, ▀▀/▄▄/█ / █ -> 0.5, ██ -> 0 

    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def _get_groundstate(self):
        return 'AFM'


class IsingASI(ASI):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' In-plane ASI with all spins on a square grid, all pointing in the same direction. '''
        self.a = a # [m] The distance between two nearest neighboring spins
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str, angle: float = 0):
        # PYTHONUPDATE_3.10: use structural pattern matching
        if pattern == 'uniform': # Angle 0°
            self.m = 2*((self.orientation[:,:,0]*math.cos(angle) + self.orientation[:,:,1]*math.sin(angle)) >= 0) - 1 # Setting empty cells to zero is responsibility of Magnets() class
        elif pattern == 'AFM':
            self.m = (self.iyy % 2)*2 - 1
        else:
            self.m = cp.random.randint(0, 2, size=self.xx.shape)*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)

    def _set_orientation(self, angle: float = 0.):
        self.orientation = cp.zeros(self.xx.shape + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        self.orientation[True,True,:] = math.cos(angle), math.sin(angle)

    def _get_unitcell(self):
        return (1, 1)

    def _get_occupation(self):
        return cp.ones_like(self.xx)

    def _get_appropriate_avg(self):
        return 'point'

    def _get_AFMmask(self):
        return cp.array([[1, 1], [-1, -1]])/4

    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def _get_groundstate(self):
        return 'uniform'


class SquareASI(ASI):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' In-plane ASI with the spins placed on, and oriented along, the edges of squares. '''
        self.a = a # [m] The side length of the squares (i.e. side length of a unit cell)
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a/2
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str, angle: float = 0):
        angle += 1e-6 # To avoid possible ambiguous rounding in uniform() if angle is perpendicular to self.orientation
        uniform = lambda angle: 2*((self.orientation[:,:,0]*math.cos(angle) + self.orientation[:,:,1]*math.sin(angle)) >= 0) - 1
        # PYTHONUPDATE_3.10: use structural pattern matching
        if pattern == 'uniform': # Angle 45°
            self.m = uniform(angle)
        elif pattern == 'AFM':
            self.m = ((self.ixx - self.iyy)//2 % 2)*2 - 1
        elif pattern == 'vortex':
            # angle near 0 or math.pi: clockwise/anticlockwise vortex, respectively
            # angle near math.pi/2 or -math.pi/2: radial out/in, respectively
            self.m = cp.ones_like(self.xx)
            distSq = ((self.ixx - (self.nx-1)/2)**2 + (self.iyy - (self.ny-1)/2)**2) # Try to put the vortex close to the center of the simulation
            distSq[cp.where(self.occupation == 1)] = cp.nan # We don't want to place the vortex center at an occupied cell
            middle_y, middle_x = divmod(cp.argmax(distSq == cp.min(distSq[~cp.isnan(distSq)])), self.nx) # The non-occupied cell closest to the center
            # Build bottom, left, top and right areas and set their magnetizations
            N = cp.where((self.ixx - middle_x < self.iyy - middle_y) & (self.ixx + self.iyy >= middle_x + middle_y))
            E = cp.where((self.ixx - middle_x >= self.iyy - middle_y) & (self.ixx + self.iyy > middle_x + middle_y))
            S = cp.where((self.ixx - middle_x > self.iyy - middle_y) & (self.ixx + self.iyy <= middle_x + middle_y))
            W = cp.where((self.ixx - middle_x <= self.iyy - middle_y) & (self.ixx + self.iyy < middle_x + middle_y))
            self.m[N] = uniform(angle            )[N]
            self.m[E] = uniform(angle - math.pi/2)[E]
            self.m[S] = uniform(angle + math.pi  )[S]
            self.m[W] = uniform(angle + math.pi/2)[W]
        else:
            self.m = cp.random.randint(0, 2, size=self.xx.shape)*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)

    def _set_orientation(self, angle: float = 0.):
        self.orientation = cp.zeros(self.xx.shape + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        self.orientation[self.iyy % 2 == 0,:] = math.cos(angle), math.sin(angle)
        self.orientation[self.iyy % 2 == 1,:] = math.cos(angle + math.pi/2), math.sin(angle + math.pi/2)
        self.orientation[self.occupation == 0,:] = 0

    def _get_unitcell(self):
        return (2, 2)

    def _get_occupation(self):
        return (self.ixx + self.iyy) % 2 == 1

    def _get_appropriate_avg(self):
        return 'cross'

    def _get_AFMmask(self):
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return cp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])

    def _get_groundstate(self):
        return 'AFM'


class PinwheelASI(SquareASI):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' In-plane ASI similar to SquareASI, but all spins rotated by 45°, hence forming a pinwheel geometry. '''
        super().__init__(n, a, ny=ny, **kwargs)

    def _set_orientation(self, angle: float = 0.):
        super()._set_orientation(angle - math.pi/4)

    def _get_groundstate(self):
        return 'uniform' if self.PBC else 'vortex'


class KagomeASI(ASI):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' In-plane ASI with all spins placed on, and oriented along, the edges of hexagons. '''
        self.a = a # [m] The distance between opposing sides of a hexagon
        self.nx = n
        if ny is None:
            self.ny = int(self.nx/math.sqrt(3))//4*4
            if not kwargs.get('PBC', False):
                self.ny -= 1
        else:
            self.ny = ny
        self.dx = a/4
        self.dy = math.sqrt(3)*self.dx
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str, angle: float = 0):
        # PYTHONUPDATE_3.10: use structural pattern matching
        self.m = 2*((self.orientation[:,:,0]*math.cos(angle) + self.orientation[:,:,1]*math.sin(angle)) >= 0) - 1
        if pattern == 'uniform': # Angle 90°
            self.m[(self.ixx - self.iyy) % 4 == 1] *= -1
        elif pattern == 'AFM':
            self.m[(self.ixx + self.iyy) % 4 == 3] *= -1
        else:
            self.m *= cp.random.randint(0, 2, size=self.xx.shape)*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)

    def _set_orientation(self, angle: float = 0.):
        self.orientation = cp.zeros(self.xx.shape + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        self.orientation[True,True,:] = math.cos(angle + math.pi/2), math.sin(angle + math.pi/2)
        self.orientation[((self.ixx - self.iyy) % 4 == 1) & (self.ixx % 2 == 1),:] = math.cos(angle - math.pi/6), math.sin(angle - math.pi/6)
        self.orientation[((self.ixx + self.iyy) % 4 == 3) & (self.ixx % 2 == 1),:] = math.cos(angle + math.pi/6), math.sin(angle + math.pi/6)
        self.orientation[self.occupation == 0,:] = 0

    def _get_unitcell(self):
        return (4, 4)

    def _get_occupation(self):
        occupation = cp.zeros_like(self.xx)
        occupation[(self.ixx + self.iyy) % 4 == 1] = 1 # One bunch of diagonals \
        occupation[(self.ixx - self.iyy) % 4 == 3] = 1 # Other bunch of diagonals /
        return occupation

    def _get_appropriate_avg(self):
        return 'hexagon'

    def _get_AFMmask(self):
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])

    def _get_groundstate(self):
        return 'uniform'


class TriangleASI(KagomeASI):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' In-plane ASI with all spins placed on, and oriented along, the edges of triangles. '''
        super().__init__(n, a, ny=ny, **kwargs)

    def _set_orientation(self, angle: float = 0.):
        super()._set_orientation(angle - math.pi/2)

    def _get_appropriate_avg(self):
        return 'triangle'

    def _get_groundstate(self):
        return 'AFM'
