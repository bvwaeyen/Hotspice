import math

from .core import Magnets
from . import config
if config.USE_GPU:
    import cupy as xp
    from cupyx.scipy import signal
    from cupyx.scipy import ndimage
else:
    import numpy as xp
    from scipy import signal
    from scipy import ndimage


class OOP_ASI(Magnets):
    """ Generic abstract class for out-of-plane ASI. """
    # Example of __init__ method for out-of-plane ASI:
    # def __init__(self, n: int, a: float, *, nx: int = None, ny: int = None, **kwargs):
    #     self.a = a # [m] Some sort of lattice constant representative for the ASI
    #     self.nx, self.ny = nx or n, ny or n # Use nx/ny if they are passed as an argument, otherwise use n
    #     self.dx, self.dy = kwargs.pop('dx', a), kwargs.pop('dy', a)
    #     super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=False, **kwargs)

    def _get_angles(self):
        # This abstract method is irrelevant for OOP ASI, so just return NaNs.
        return xp.nan*xp.zeros_like(self.ixx)

    def correlation_NN(self, NN=None): # Basically <S_i S_{i+1}>
        """ `NN` can be used to set a different NN mask (default: `self._get_nearest_neighbors()`),
            for example in subclasses for further-NN implementations.
        """
        if NN is None: NN = self._get_nearest_neighbors()
        total_NN = signal.convolve2d(self.occupation, NN, mode='same', boundary='wrap' if self.PBC else 'fill')*self.occupation
        neighbors_sign_sum = signal.convolve2d(self.m, NN, mode='same', boundary='wrap' if self.PBC else 'fill')
        valid_positions = xp.where(total_NN != 0)
        return float(xp.mean(self.m[valid_positions]*neighbors_sign_sum[valid_positions]/total_NN[valid_positions]))

class IP_ASI(Magnets):
    """ Generic abstract class for in-plane ASI. """
    # Example of __init__ method for in-plane ASI:
    # def __init__(self, n: int, a: float, *, nx: int = None, ny: int = None, **kwargs):
    #     self.a = a # [m] Some sort of lattice constant representative for the ASI
    #     self.nx, self.ny = nx or n, ny or n # Use nx/ny if they are passed as an argument, otherwise use n
    #     self.dx, self.dy = kwargs.pop('dx', a), kwargs.pop('dy', a)
    #     super().__init__(self.nx, self.ny, self.dx, self.dy, in_plane=True, **kwargs)


class OOP_Square(OOP_ASI):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ Out-of-plane ASI in a square arrangement. """
        self.a = a # [m] The distance between two nearest neighboring spins
        if nx is None: nx = n
        if ny is None: ny = n
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dx, dy = kwargs.pop('dx', a), kwargs.pop('dy', a)
        super().__init__(nx, ny, dx, dy, in_plane=False, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = ((self.ixx - self.iyy) % 2)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_occupation(self):
        return xp.ones_like(self.xx)

    def _get_appropriate_avg(self):
        return ['point', 'square', 'crossfour'] # 'triangle' is not horrendous either, but does not make much sense

    def _get_AFMmask(self):
        return xp.array([[0, -1], [-1, 2]], dtype='float')/4 # Possible situations: ▚/▞ -> 1, ▀▀/▄▄/█ / █ -> 0.5, ██ -> 0 

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def _get_groundstate(self):
        return 'afm'
    
    def get_domains(self): # TODO: perhaps introduce this in other ASIs?
        return (((self.ixx + self.iyy) % 2)*2 - 1) == self.m # TODO: get the number of distinct domain possibilities

    def domain_sizes(self):
        domains = self.get_domains()
        sizes = []
        for domain in xp.unique(domains):
            array, total_domains = ndimage.label(domains == domain)
            sizes.append(xp.bincount(array.reshape(-1))[1:]) # Discard 1st element because it counts the zeros, which is the other domains
        return xp.concatenate(sizes)
    def domain_size(self):
        return xp.mean(self.domain_sizes())
    
    def correlation_NN(self, N=1): # Basically <S_i S_{i+1}>
        """ Calculates the correlation with the `N`-th nearest neighbor (average over the system). """
        if not isinstance(N, int) and not (1 <= N <= 3): raise ValueError("Can only calculate NN correlation up to third-nearest neighbor")
        match N:
            case 1: NN = None
            case 2: NN = xp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
            case 3: NN = xp.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
        return super().correlation_NN(NN=NN)


class OOP_Triangle(OOP_ASI):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ Out-of-plane ASI on a triangular (hexagonal) lattice. """
        self.a = a # [m] The distance between two nearest neighboring spins
        if nx is None: nx = n
        if ny is None:
            ny = int(nx/math.sqrt(3))//2*2 # Try to make the domain reasonably square-shaped
            if not kwargs.get('PBC', False): ny -= 1 # Remove dangling spins if no PBC
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dx = kwargs.pop('dx', a/2)
        dy = kwargs.pop('dy', math.sqrt(3)*dx)
        super().__init__(nx, ny, dx, dy, in_plane=False, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = ((self.ixx - self.iyy)//2 % 2)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_occupation(self):
        return (self.ixx + self.iyy) % 2 == 1

    def _get_appropriate_avg(self): # TODO: not so relevant
        return ['point', 'triangle']

    def _get_AFMmask(self): # TODO: not so relevant
        return xp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])

    def _get_groundstate(self): # TODO: not so relevant
        return 'afm'


class OOP_Honeycomb(OOP_ASI):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ Out-of-plane ASI in a honeycomb arrangement.
            This happens e.g. when the magnets themselves are triangular and close-packed.
            Unitcell 6x2.
        """
        self.a = a # [m] The distance between two nearest neighboring spins
        if nx is None: nx = n
        if ny is None:
            ny = int(nx/math.sqrt(3))//2*2 # Try to make the domain reasonably square-shaped
            if not kwargs.get('PBC', False): ny -= 1 # Remove dangling spins if no PBC
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dy = kwargs.pop('dx', a/2) # TODO: add nonuniform grid here as well because Honeycomb can benefit from it
        dx = kwargs.pop('dy', math.sqrt(3)*dy)
        super().__init__(nx, ny, dx, dy, in_plane=False, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = (self.iyy % 3 == 1)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_occupation(self):
        occupation = xp.ones_like(self.xx)
        occupation[(self.ixx + self.iyy) % 2 == 0] = 0 # A bunch of diagonals 
        occupation[self.iyy % 3 == 2] = 0 # Some vertical lines that are completely empty
        return occupation

    def _get_appropriate_avg(self): # TODO: not so relevant
        return ['point', 'cross']

    def _get_AFMmask(self): # TODO: not so relevant
        return xp.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])

    def _get_groundstate(self): # TODO: not so relevant
        return 'afm'

    def get_domains(self):
        sublattice = xp.where(self.iyy % 3 == 0, -1, 1)
        return (sublattice == self.m).astype(int)


class OOP_Cairo(OOP_ASI):
    BETA = math.pi/4 + math.acos(math.sqrt(2)/4) # In an equilateral pentagon with two non-adjacent right angles, `beta` is the size of the two other equal angles. 
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, beta: float = None, offset_factor: float = 1., **kwargs):
        """ In-plane ASI with all spins placed on the edges of the equilateral Cairo tiling.
            NOTE: `dx` and `dy` can not be specified spearately, all is controlled by `a` and `beta`.
            `offset_factor` can be used to offset the magnets to make the 4-vertices and 3-vertices look less unbalanced.
                The nicest shape is achieved for `offset_factor=1.2`.
        """
        self.a = a # [m] The side length of the pentagons
        if nx is None: nx = n
        if ny is None: ny = n
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        
        self.beta = OOP_Cairo.BETA if beta is None else beta
        dx = dy = [-a*math.cos(self.beta),-a*math.cos(self.beta),a/2,a/2]
        super().__init__(nx, ny, dx, dy, in_plane=False, **kwargs)

    def _set_m(self, pattern: str, angle=None):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = xp.ones_like(self.ixx)
                self.m[(self.ixx + self.iyy) % 5 == 0] *= -1
                self.m[(self.ixx + self.iyy) % 5 == 3] *= -1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_occupation(self):
        occupation = xp.zeros_like(self.xx)
        for x, y in [(1,1), (1,5), (2,3), (3,6), (3,0), (4,3), (5,1), (5,5), (6,7), (7,2), (7,4), (0,7)]:
            occupation[y::8,x::8] = 1
        return occupation

    def _get_appropriate_avg(self):
        return ['point']

    def _get_AFMmask(self):
        return xp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

    def _get_groundstate(self):
        return 'afm'


class IP_Ising(IP_ASI):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ In-plane ASI with all spins on a square grid, all pointing in the same direction. """
        self.a = a # [m] The distance between two nearest neighboring spins
        if nx is None: nx = n
        if ny is None: ny = n
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dx, dy = kwargs.pop('dx', a), kwargs.pop('dy', a)
        super().__init__(nx, ny, dx, dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = (self.iyy % 2)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_angles(self):
        return xp.zeros_like(self.xx)

    def _get_occupation(self):
        return xp.ones_like(self.xx)

    def _get_appropriate_avg(self):
        return ['point', 'cross', 'squarefour']

    def _get_AFMmask(self):
        return xp.array([[1, 1], [-1, -1]])/4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def _get_groundstate(self):
        return 'uniform'


class IP_Square_Closed(IP_ASI):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ In-plane ASI with the spins placed on, and oriented along, the edges of squares. """
        self.a = a # [m] The side length of the squares (i.e. side length of a unit cell)
        if nx is None: nx = n
        if ny is None: ny = n
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dx, dy = kwargs.pop('dx', a/2), kwargs.pop('dy', a/2)
        super().__init__(nx, ny, dx, dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = ((self.ixx - self.iyy)//2 % 2)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_angles(self):
        angles = xp.zeros_like(self.xx)
        angles[self.iyy % 2 == 1] = math.pi/2
        return angles

    def _get_occupation(self):
        return (self.ixx + self.iyy) % 2 == 1

    def _get_appropriate_avg(self):
        return ['cross', 'square', 'squarefour']

    def _get_AFMmask(self):
        return xp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return xp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])

    def _get_groundstate(self):
        return 'afm'


class IP_Pinwheel_Diamond(IP_Square_Closed):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ In-plane ASI similar to IP_Square, but all spins rotated by 45°, hence forming a pinwheel geometry. """
        kwargs['angle'] = kwargs.get('angle', 0) - math.pi/4
        super().__init__(a, n=n, nx=nx, ny=ny, **kwargs)

    def _get_appropriate_avg(self):
        return ['squarefour', 'cross']

    def _get_groundstate(self):
        return 'uniform' if self.PBC else 'vortex'


class IP_Square(IP_Square_Closed): pass # Just an alias for IP_Square, to follow naming scheme from "Apparent ferromagnetism in the pinwheel artificial spin ice"
class IP_Square_Open(IP_ASI):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ In-plane ASI with the spins placed on, and oriented along, the edges of squares.
            The entire domain does not, however, form a square, but rather a 'diamond'.
        """
        self.a = a # [m] The side length of the squares
        if nx is None: nx = n
        if ny is None: ny = n
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dx, dy = kwargs.pop('dx', a/math.sqrt(2)), kwargs.pop('dy', a/math.sqrt(2))
        super().__init__(nx, ny, dx, dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = (self.iyy % 2)*2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_angles(self):
        angles = -xp.ones_like(self.xx)*math.pi/4
        angles[(self.iyy + self.ixx) % 2 == 1] = math.pi/4
        return angles

    def _get_occupation(self):
        return xp.ones_like(self.xx)

    def _get_appropriate_avg(self):
        return ['squarefour']

    def _get_AFMmask(self):
        return xp.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def _get_groundstate(self):
        return 'afm'

class IP_Pinwheel(IP_Pinwheel_Diamond): pass # Just an alias for IP_Pinwheel, to follow naming scheme from "Apparent ferromagnetism in the pinwheel artificial spin ice"
class IP_Pinwheel_LuckyKnot(IP_Square_Open):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ In-plane ASI similar to IP_Square_Open, but all spins rotated by 45°, hence forming a pinwheel geometry. """
        kwargs['angle'] = kwargs.get('angle', 0) - math.pi/4
        super().__init__(a, n=n, nx=nx, ny=ny, **kwargs)

    def _get_groundstate(self):
        return 'uniform' if self.PBC else 'vortex'
    
    def _get_appropriate_avg(self):
        return ['crossfour', 'square']


class IP_Kagome(IP_ASI): # TODO: change the meaning of <a> such that it is the side length of a hexagon (this would be more in line with other ASI definitions).
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ In-plane ASI with all spins placed on, and oriented along, the edges of hexagons. """
        self.a = a # [m] The distance between opposing sides of a hexagon
        if nx is None: nx = n
        if ny is None:
            ny = int(nx/math.sqrt(3))//4*4 # Try to make the domain reasonably square-shaped
            if not kwargs.get('PBC', False): ny -= 1 # Remove dangling spins if no PBC
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dx = kwargs.pop('dx', a/4)
        dy = kwargs.pop('dy', math.sqrt(3)*dx)
        super().__init__(nx, ny, dx, dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str, angle=None):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = xp.ones_like(self.ixx)
                self.m[(self.ixx + self.iyy) % 4 == 3] *= -1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_angles(self):
        angles = xp.ones_like(self.xx)*math.pi/2
        angles[((self.ixx - self.iyy) % 4 == 1) & (self.ixx % 2 == 1)] = -math.pi/6
        angles[((self.ixx + self.iyy) % 4 == 3) & (self.ixx % 2 == 1)] = math.pi/6
        return angles

    def _get_occupation(self):
        occupation = xp.zeros_like(self.xx)
        occupation[(self.ixx + self.iyy) % 4 == 1] = 1 # One bunch of diagonals \
        occupation[(self.ixx - self.iyy) % 4 == 3] = 1 # Other bunch of diagonals /
        return occupation

    def _get_appropriate_avg(self):
        return ['cross', 'point', 'triangle', 'hexagon']

    def _get_AFMmask(self):
        return xp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])

    def _get_groundstate(self):
        return 'uniform'


class IP_Triangle(IP_Kagome):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        """ In-plane ASI with all spins placed on, and oriented along, the edges of triangles. """
        kwargs['angle'] = kwargs.get('angle', 0) - math.pi/2
        super().__init__(a, n=n, nx=nx, ny=ny, **kwargs)

    def _get_appropriate_avg(self):
        return ['point', 'cross', 'triangle']

    def _get_groundstate(self):
        return 'afm'


class IP_Cairo(IP_ASI):
    BETA = math.pi/4 + math.acos(math.sqrt(2)/4) # In an equilateral pentagon with two non-adjacent right angles, `beta` is the size of the two other equal angles. 
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, beta: float = None, offset_factor: float = 1., **kwargs):
        """ In-plane ASI with all spins placed on the edges of the equilateral Cairo tiling.
            NOTE: `dx` and `dy` can not be specified spearately, all is controlled by `a` and `beta`.
            `offset_factor` can be used to offset the magnets to make the 4-vertices and 3-vertices look less unbalanced.
                The nicest shape is achieved for `offset_factor=1.2`.
        """
        self.a = a # [m] The side length of the pentagons
        if nx is None: nx = n
        if ny is None: ny = n
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        
        self.beta = IP_Cairo.BETA if beta is None else beta
        dxPQ = dyRS = a/2*math.sin(self.beta)
        dxQR = dxPQ - a/2*math.cos(self.beta)
        dxRS = dyPQ = math.sin(math.pi/2 - self.beta)*a + math.cos(math.pi/2 - self.beta)*a - a/2*math.cos(self.beta)
        dyQR = (dyPQ - dxPQ)*offset_factor
        dx = dy = [-a*math.cos(self.beta)*offset_factor,dyQR,dxPQ,dxPQ,dyQR]
        super().__init__(nx, ny, dx, dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str, angle=None):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = xp.ones_like(self.ixx)
                self.m[(self.ixx + self.iyy) % 5 == 1] *= -1
                self.m[(self.ixx + self.iyy) % 5 == 3] *= -1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_angles(self):
        angles = xp.zeros_like(self.xx)
        s = lambda x, y: (slice(y,self.ny,10), slice(x,self.nx,10)) # Unit cell is 10x10
        angles[s(3,3)] = angles[s(8,8)] = 0 # -
        angles[s(3,8)] = angles[s(8,3)] = math.pi/2 # |
        angles[s(1,2)] = angles[s(5,4)] = angles[s(6,7)] = angles[s(0,9)] = math.pi - self.beta # steep /
        angles[s(1,4)] = angles[s(5,2)] = angles[s(6,9)] = angles[s(0,7)] = self.beta # steep \
        angles[s(2,6)] = angles[s(4,0)] = angles[s(7,1)] = angles[s(9,5)] = self.beta - math.pi/2 # shallow /
        angles[s(2,0)] = angles[s(4,6)] = angles[s(7,5)] = angles[s(9,1)] = math.pi/2 - self.beta # shallow \
        return angles

    def _get_occupation(self):
        occupation = xp.zeros_like(self.xx)
        occupation[xp.where(self._get_angles() != 0)] = 1 # Non-horizontal magnets
        occupation[3::10,3::10] = occupation[8::10,8::10] = 1 # Horizontal magnets
        return occupation

    def _get_appropriate_avg(self):
        return ['point']

    def _get_AFMmask(self):
        return xp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')/4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

    def _get_groundstate(self):
        return 'afm'


class IP_Shakti(IP_ASI):
    def __init__(self, a: float, n: int = None, *, nx: int = None, ny: int = None, **kwargs):
        # TODO: Include a default value for the length of each magnet (for monopoles or finite dipole) in the ASI class definition. Otherwise the user has to define the non-equal d themselves, which is suboptimal.
        """ In-plane 'Shakti' ASI, which contains magnets of unequal size.
            NOTE: While this lattice is equivalent to `IP_Cairo(beta=pi/2)`, this class exists
                  because it can use an 8x8 unit cell, rather than the 10x10 of `IP_Cairo`.
            NOTE: The unequal magnet size is not included in this class.
        """
        self.a = a/4
        if nx is None: nx = n
        if ny is None: ny = n
        if nx is None or ny is None: raise AttributeError("Must specify <n> if either <nx> or <ny> are not specified.")
        dx, dy = kwargs.pop('dx', a / 4), kwargs.pop('dy', a / 4)
        super().__init__(nx, ny, dx, dy, in_plane=True, **kwargs)

    def _set_m(self, pattern: str, angle=None):
        match str(pattern).strip().lower():
            case 'afm':
                self.m = ((self.ixx - self.iyy) % 2) * 2 - 1
            case str(unknown_pattern):
                super()._set_m(pattern=unknown_pattern)

    def _get_angles(self):
        angles = xp.zeros_like(self.xx)
        angles[self.ixx % 2 == 0] = xp.pi/2
        angles[2::8,2::8] = angles[6::8,6::8] = 0 # Horizontal magnets
        return angles

    def _get_occupation(self):
        occupation = xp.zeros_like(self.xx)
        occupation[(self.ixx + self.iyy) % 2 == 1] = 1
        occupation[2::4, 2::4] = 1
        occupation[1::4, 2::4] = occupation[3::4, 2::4] = occupation[2::4, 1::4] = occupation[2::4, 3::4] = 0
        return occupation

    def _get_appropriate_avg(self):
        return ['point']

    def _get_AFMmask(self):
        return xp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float') / 4

    def _get_nearest_neighbors(self):
        return xp.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

    def _get_groundstate(self):
        return 'afm'
