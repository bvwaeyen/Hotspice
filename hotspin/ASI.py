import math
import warnings

import cupy as cp

from abc import ABC, abstractmethod

from .core import Magnets


class FullASI(Magnets):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' Out-of-plane ASI in a square arrangement. '''
        self.a = a # [m] The distance between two nearest neighboring spins
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=False, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'uniform':
                self.m = self._get_m_uniform()
            case 'afm':
                self.m = ((self.ixx - self.iyy) % 2)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

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


class IsingASI(Magnets):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' In-plane ASI with all spins on a square grid, all pointing in the same direction. '''
        self.a = a # [m] The distance between two nearest neighboring spins
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'uniform':
                self.m = self._get_m_uniform()
            case 'afm':
                self.m = (self.iyy % 2)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _set_orientation(self):
        self.orientation = cp.zeros(self.xx.shape + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        self.orientation[:,:,0] = 1
        return 0

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


class SquareASI(Magnets):
    def __init__(self, n: int, a: float, *, ny: int = None, **kwargs):
        ''' In-plane ASI with the spins placed on, and oriented along, the edges of squares. '''
        self.a = a # [m] The side length of the squares (i.e. side length of a unit cell)
        self.nx = n
        self.ny = n if ny is None else ny
        self.dx = self.dy = a/2
        super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'uniform':
                self.m = self._get_m_uniform()
            case 'afm':
                self.m = ((self.ixx - self.iyy)//2 % 2)*2 - 1
            case 'vortex':
                # When using 'angle' property of Magnets.initialize_m:
                # <angle> near 0 or math.pi: clockwise/anticlockwise vortex, respectively
                # <angle> near math.pi/2 or -math.pi/2: bowtie configuration (top region: up/down, respectively)
                self.m = cp.ones_like(self.xx)
                distSq = ((self.ixx - (self.nx-1)/2)**2 + (self.iyy - (self.ny-1)/2)**2) # Try to put the vortex close to the center of the simulation
                distSq[cp.where(self.occupation == 1)] = cp.nan # We don't want to place the vortex center at an occupied cell
                middle_y, middle_x = divmod(cp.argmax(distSq == cp.min(distSq[~cp.isnan(distSq)])), self.nx) # The non-occupied cell closest to the center
                # Build bottom, left, top and right areas and set their magnetizations
                N = cp.where((self.ixx - middle_x < self.iyy - middle_y) & (self.ixx + self.iyy >= middle_x + middle_y))
                E = cp.where((self.ixx - middle_x >= self.iyy - middle_y) & (self.ixx + self.iyy > middle_x + middle_y))
                S = cp.where((self.ixx - middle_x > self.iyy - middle_y) & (self.ixx + self.iyy <= middle_x + middle_y))
                W = cp.where((self.ixx - middle_x <= self.iyy - middle_y) & (self.ixx + self.iyy < middle_x + middle_y))
                self.m[N] = self._get_m_uniform(0         )[N]
                self.m[E] = self._get_m_uniform(-math.pi/2)[E]
                self.m[S] = self._get_m_uniform(math.pi   )[S]
                self.m[W] = self._get_m_uniform(math.pi/2 )[W]
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _set_orientation(self):
        self.orientation = cp.zeros(self.xx.shape + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        self.orientation[self.iyy % 2 == 0,0] = 1
        self.orientation[self.iyy % 2 == 1,1] = 1
        return 0

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

    def _set_orientation(self):
        super()._set_orientation()
        return -math.pi/4

    def _get_groundstate(self):
        return 'uniform' if self.PBC else 'vortex'


class KagomeASI(Magnets):
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

    def _set_m(self, pattern: str, angle=None):
        match str(pattern).strip().lower():
            case 'uniform':
                self.m = self._get_m_uniform()
            case 'afm':
                self.m = cp.ones_like(self.ixx)
                self.m[(self.ixx + self.iyy) % 4 == 3] *= -1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _set_orientation(self):
        self.orientation = cp.zeros(self.xx.shape + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        self.orientation[:,:,1] = 1
        self.orientation[((self.ixx - self.iyy) % 4 == 1) & (self.ixx % 2 == 1),:] = math.cos(-math.pi/6), math.sin(-math.pi/6)
        self.orientation[((self.ixx + self.iyy) % 4 == 3) & (self.ixx % 2 == 1),:] = math.cos(math.pi/6), math.sin(math.pi/6)
        return 0

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

    def _set_orientation(self):
        super()._set_orientation()
        return -math.pi/2

    def _get_appropriate_avg(self):
        return 'triangle'

    def _get_groundstate(self):
        return 'AFM'
