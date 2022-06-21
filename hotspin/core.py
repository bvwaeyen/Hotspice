import math
import warnings

import cupy as cp
import cupyx as cpx
import numpy as np

from abc import ABC, abstractmethod
from cupyx.scipy import signal
from dataclasses import dataclass, field
from functools import cache
from scipy.spatial import distance
from textwrap import dedent

from .poisson import PoissonGrid
from .utils import as_cupy_array, check_repetition, clean_indices, Field, full_obj_name, mirror4


kB = 1.380649e-23

# The random number generator used throughout hotspin.
rng = cp.random.default_rng() # There is no significant speed difference between XORWOW or MRG32k3a or Philox4x3210


@dataclass(slots=True)
class SimParams:
    SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF: int = 20 # If there are strictly more than <this> switches in a single iteration, a convolution is used, otherwise the energies are just summed.
    # TODO: THOROUGH ANALYSIS OF REDUCED_KERNEL_SIZE AND TAKE APPROPRIATE MEASURES (e.g. full recalculation every <n> steps)
    REDUCED_KERNEL_SIZE: int = 20 # If nonzero, the dipolar kernel is cropped to an array of shape (2*<this>-1, 2*<this>-1).
    UPDATE_SCHEME: str = 'Glauber' # Can be any of 'Néel', 'Glauber', 'Wolff'
    MULTISAMPLING_SCHEME: str = 'grid' # Can be any of 'single', 'grid', 'Poisson', 'cluster'. Only used if UPDATE_SCHEME is 'Glauber'.

    def __post_init__(self):
        self.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF = int(self.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF)
        self.REDUCED_KERNEL_SIZE = int(self.REDUCED_KERNEL_SIZE)
        if self.UPDATE_SCHEME not in (allowed := ['Glauber', 'Néel', 'Wolff']):
            raise ValueError(f"UPDATE_SCHEME='{self.UPDATE_SCHEME}' is invalid: allowed values are {allowed}.")
        if self.MULTISAMPLING_SCHEME not in (allowed := ['single', 'grid', 'Poisson', 'cluster']):
            raise ValueError(f"MULTISAMPLING_SCHEME='{self.MULTISAMPLING_SCHEME}' is invalid: allowed values are {allowed}.")
        # If a multisampling scheme is incompatible with an update scheme, an error should be raised at runtime, not here.


class Magnets(ABC):
    def __init__(
        self, nx: int, ny: int, dx: float, dy: float, *,
        T: Field = 300, E_B: Field = 0., moment: Field = None, Msat: Field = 800e3, V: Field = 2e-22,
        pattern: str = None, energies: tuple = None, PBC: bool = False, angle: float = 0., params: SimParams = None, in_plane: bool = False):
        '''
            !!! THIS CLASS SHOULD NOT BE INSTANTIATED DIRECTLY, USE AN ASI WRAPPER INSTEAD !!!
            The position of magnets is specified using <nx>, <ny>, <dx> and <dy>. Only rectilinear grids are currently allowed.
            The initial configuration of a Magnets geometry consists of 3 parts:
                1) in_plane: Magnets can be in-plane or out-of-plane: True or False, respectively. Determined by subclassing.
                2) ASI type: Defined through subclasses (pinwheel, kagome, Ising...). This concerns the layout of spins.
                3) pattern:  The initial magnetization direction (e.g. up/down) can be 'uniform', 'AFM', 'vortex' or 'random'.
            One can also specify which energies should be considered by passing a tuple of Energy objects to <energies>.
                If <energies> is not passed, only the dipolar energy is considered. If a tuple is passed, however, the
                dipolar energy is not automatically added: if still desired, the user should include this themselves.
            The arguments <T>, <E_B>, <moment>, <Msat> and <V> can all be constant (by passing them as a float) or spatially
                varying (by passing them as a shape (nx, ny) array).
            The magnetic moment can either be specified using the 2 arguments <Msat> (default 800e3 A/m) and <V> (default 2e-22 m³),
                or directly by passing a value to <moment>. The argument <moment> takes precedence over <Msat> and <V>.
            To enable periodic boundary conditions (default: disabled), pass PBC=True.
            To add an extra rotation to all spins in addition to self._get_angles(), pass <angle> as a float (in radians).
            The parameter <in_plane> should only be used in a super().__init__() call when subclassing this class.
        '''
        self.params = SimParams() if params is None else params # This can just be edited and accessed normally since it is just a dataclass
        energies: tuple[Energy] = (DipolarEnergy(),) if energies is None else tuple(energies) # [J] use dipolar energy by default

        # Initialize properties that are necessary to access by subsequent method calls
        self.energies = list[Energy]() # Don't manually add anything to this, instead call self.add_energy()
        self.in_plane = in_plane
        self.nx, self.ny = int(nx), int(ny)
        self.dx, self.dy = float(dx), float(dy)
        self.xx, self.yy = cp.meshgrid(cp.linspace(0, self.dx*(self.nx-1), self.nx), cp.linspace(0, self.dy*(self.ny-1), self.ny)) # [m]
        self.index = range(self.xx.size)
        self.ixx, self.iyy = cp.meshgrid(cp.arange(0, self.nx), cp.arange(0, self.ny))
        self.x_min, self.y_min, self.x_max, self.y_max = float(self.xx[0,0]), float(self.yy[0,0]), float(self.xx[-1,-1]), float(self.yy[-1,-1])

        # Main initialization steps to create the geometry
        self.occupation = self._get_occupation().astype(bool).astype(int) # Make sure that it is either 0 or 1
        self.n = int(cp.sum(self.occupation)) # Number of magnets in the simulation
        if self.n == 0: raise ValueError(f"Can not initialize {full_obj_name(self)} of size {self.nx}x{self.ny}, as this does not contain any spins.")
        if self.in_plane: self._initialize_ip(angle=angle) # Initialize orientation of each magnet
        self.unitcell = self._get_unitcell() # This needs to be after occupation and initialize_ip, and before any defects are introduced
        self.PBC = PBC
        self.initialize_m(self._get_groundstate() if pattern is None else pattern, update_energy=False)

        # Initialize field-like properties (!!! these need the geometry to exist already, since they have the same shape)
        self.T = T # [K]
        self.E_B = E_B # [J]
        self.moment = Msat*V if moment is None else moment # [Am²] moment is saturation magnetization multiplied by volume

        # History
        self.t = 0. # [s]
        self.history = History()
        self.switches, self.attempted_switches = 0, 0

        # Finally initialize the energies (at the end, since they might require self.orientation etc.)
        for energy in energies: self.add_energy(energy)

    def _get_closest_dist(self):
        ''' Returns the closest distance between two magnets in the simulation. '''
        slice = cp.where(self.occupation[:self.unitcell.y*2,:self.unitcell.x*2]) # We only need at most two unit cells
        pos_x, pos_y = self.xx[slice], self.yy[slice]
        return distance.pdist(cp.asarray([pos_x, pos_y]).get().T).min()

    def _get_m_uniform(self, angle=0):
        ''' Returns the self.m state with all magnets aligned along <angle> as much as possible. '''
        angle += 1e-6 # To avoid possible ambiguous rounding for popular angles, if <angle> ⊥ <self.orientation>
        if self.in_plane:
            return self.occupation*(2*((self.orientation[:,:,0]*cp.cos(angle) + self.orientation[:,:,1]*cp.sin(angle)) >= 0) - 1)
        else:
            return cp.ones_like(self.xx)*((cp.floor(angle/math.pi - .5) % 2)*2 - 1)
    
    def _get_unitcell(self, max_cell=100):
        ''' Returns a Unitcell containing the number of single grid cells in a unit cell along the x- and y-axis.
            @param max_cell [int] (100): Only unitcells with ux+uy < max_cell are considered for performance reasons.
        '''
        for n in range(1, min(self.nx + self.ny + 1, max_cell + 1)): # Test possible unit cells in a triangle-like manner: (1,1), (2,1), (1,2), (3,1), (2,2), (1,3), ...
            for i in range(max(0, n - self.nx) + 1, min(n, self.ny) + 1): # Don't test unit cells larger than the domain
                ux = n - i + 1
                uy = i
                if self.in_plane:
                    if check_repetition(self.orientation, ux, uy):
                        return Unitcell(ux, uy)
                else:
                    if check_repetition(self.occupation, ux, uy):
                        return Unitcell(ux, uy)
        warnings.warn(dedent(f"""
            Could not detect a reasonably sized unit cell. Defaulting to entire domain {self.nx}x{self.ny}.
            For large simulations, this can cause high memory consumption and very poor performance."""), stacklevel=2)
        return Unitcell(self.nx, self.ny)

    def initialize_m(self, pattern='random', *, angle=0, update_energy=True):
        ''' Initializes the self.m (array of -1, 0 or 1) and occupation.
            @param pattern [str]: can be any of "random", "uniform", "AFM".
        '''
        self._set_m(pattern)
        self.m = self.m.astype(float)
        self.m = cp.multiply(self.m, self.occupation)
        self.m[cp.where(self._get_m_uniform() != self._get_m_uniform(angle))] *= -1 # Allow 'rotation' for any kind of initialized state
        if update_energy: self.update_energy() # Have to recalculate all the energies since m changed completely

    def _initialize_ip(self, angle: float = 0.):
        ''' Initialize the angles of all the magnets (only applicable in the in-plane case).
            This function should only be called by the Magnets() class itself, not by the user.
            @param angle [float] (0): the additional angle (in radians) by which every spin in
                the system will be rotated, i.e. in addition to self._get_angles().
        '''
        assert self.in_plane, "Can not _initialize_ip() if magnets are not in-plane (in_plane=False)."
        self.angles = (self._get_angles() + angle)*self.occupation
        self.orientation = cp.zeros(self.xx.shape + (2,))
        self.orientation[:,:,0] = cp.cos(self.angles)*self.occupation
        self.orientation[:,:,1] = cp.sin(self.angles)*self.occupation

    def add_energy(self, energy: 'Energy', verbose=True):
        ''' Adds an Energy object to self.energies. This object is stored under its reduced name,
            e.g. ZeemanEnergy is stored under 'zeeman'.
            @param energy [Energy]: the energy to be added.
        '''
        energy.initialize(self)
        for i, e in enumerate(self.energies):
            if type(e) is type(energy):
                if verbose: warnings.warn(f'An instance of {type(energy).__name__} was already included in the simulation, and has now been overwritten.', stacklevel=2)
                self.energies[i] = energy
                return
        self.energies.append(energy)

    def remove_energy(self, name: str, verbose=True):
        ''' Removes the specified energy from self.energies.
            @param name [str]: the name of the energy to be removed. Case-insensitive and may or may not include
                the 'energy' part of the class name. Valid options include e.g. 'dipolar', 'DipolarEnergy'...
        '''
        name = name.lower().replace('energy', '')
        for i, e in enumerate(self.energies):
            if name == e.shortname:
                self.energies.pop(i)
                return
        if verbose: warnings.warn(f"There is no '{name}' energy associated with this Magnets object. Valid energies are: {[e.shortname for e in self.energies]}", stacklevel=2)

    def get_energy(self, name: str, verbose=True):
        ''' Returns the specified energy from self.energies.
            @param name [str]: the name of the energy to be returned. Case-insensitive and may or may not include
                the 'energy' part of the class name. Valid options include e.g. 'dipolar', 'DipolarEnergy'...
            @returns [Energy]: the requested energy object.
        '''
        name = name.lower().replace('energy', '')
        for e in self.energies:
            if name == e.shortname: return e
        if verbose: warnings.warn(f"There is no '{name}' energy associated with this Magnets object. Valid energies are: {[e.shortname for e in self.energies]}", stacklevel=2)
        return None

    def update_energy(self, index: np.ndarray|cp.ndarray = None):
        ''' Updates all the energies which are currently present in the simulation.
            @param index [array] (None): if specified, only the magnets at these indices are considered in the calculation.
                We need a NumPy or CuPy array (to easily determine its size: if =2, then only a single switch is considered.)
        '''
        if index is not None: index = clean_indices(index)
        for energy in self.energies:
            if index is None: # No index specified, so update fully
                energy.update()
            elif index[0].size == 1: # Only a single index is present, so just use update_single
                energy.update_single(index)
            elif index[0].size > 1: # Index is specified and contains multiple ints, so we must use update_multiple
                energy.update_multiple(index)
    
    def switch_energy(self, indices2D):
        ''' @return [cp.array]: the change in energy for each magnet in indices2D, in the same order, if they were to switch. '''
        indices2D = clean_indices(indices2D)
        return cp.sum(cp.asarray([energy.energy_switch(indices2D) for energy in self.energies]), axis=0)

    @property
    def MCsteps(self): # Number of Monte Carlo steps
        return self.attempted_switches/self.n

    @property
    def E(self) -> cp.ndarray: # This could be relatively expensive to calculate, so maybe not ideal
        return cp.sum(cp.asarray([energy.E for energy in self.energies]), axis=0)

    @property
    def T(self) -> cp.ndarray:
        return self._T
    
    @T.setter
    def T(self, value: Field):
        self._T = as_cupy_array(value, self.xx.shape)

    @property
    def E_B(self) -> cp.ndarray:
        return self._E_B
    
    @E_B.setter
    def E_B(self, value: Field):
        self._E_B = as_cupy_array(value, self.xx.shape)

    @property
    def moment(self) -> cp.ndarray:
        return self._moment
    
    @moment.setter
    def moment(self, value: Field):
        self._moment = as_cupy_array(value, self.xx.shape)
        self._momentSq = self._moment*self._moment

    @property
    def PBC(self):
        return self._PBC
    
    @PBC.setter
    def PBC(self, value: bool):
        self._PBC = bool(value)
        if hasattr(self, 'energies'):
            for energy in self.energies: energy.initialize(self)
        if hasattr(self, 'unitcell'):
            if self._PBC and (self.nx % self.unitcell.x != 0 or self.ny % self.unitcell.y != 0): # Unit cells don't match along the edges
                warnings.warn(dedent(f"""
                    Be careful with PBC, as there are not an integer number of unit cells in the simulation!
                    Hence, the boundaries might not nicely fit together. Adjust nx or ny to alleviate this
                    (unit cell has size {self.unitcell.x}x{self.unitcell.y})."""), stacklevel=2)

    @property
    def kBT(self) -> cp.ndarray:
        return kB*self._T

    @property
    def m_avg_x(self) -> float:
        return float(cp.mean(cp.multiply(self.m, self.orientation[:,:,0])))*self.nx*self.ny/self.n if self.in_plane else 0
    @property
    def m_avg_y(self) -> float:
        return float(cp.mean(cp.multiply(self.m, self.orientation[:,:,1])))*self.nx*self.ny/self.n if self.in_plane else 0
    @property
    def m_avg(self) -> float:
        return (self.m_avg_x**2 + self.m_avg_y**2)**(1/2) if self.in_plane else float(cp.mean(self.m))
    @property
    def m_avg_angle(self) -> float: # If m_avg=0, this is 0. For OOP ASI, this is 0 if m_avg > 0, and Pi if m_avg < 0
        return math.atan2(self.m_avg_y, self.m_avg_x) if self.in_plane else 0. + (self.m_avg < 0)*math.pi

    @property
    def E_tot(self) -> float:
        return float(sum([e.E_tot for e in self.energies]))

    @property
    def T_avg(self) -> float:
        return float(cp.mean(self.T))
    
    @property
    def E_B_avg(self) -> float:
        return float(cp.mean(self.E_B))
    
    @property
    def moment_avg(self) -> float:
        return float(cp.mean(self.moment))
    
    def set_T(self, f, /, *, center=False, crystalunits=False):
        ''' Sets the temperature field according to a spatial function. To simply set the temperature array, use self.T = assignment instead.
            @param f [function(x,y)->T]: x and y in meters yield T in Kelvin (should accept CuPy arrays as x and y).
            @param center [bool] (False): if True, the origin is put in the middle of the simulation,
                otherwise it is in the usual lower left corner.
            @param crystalunits [bool] (False): if True, x and y are assumed to be a number of cells instead of a distance in meters (but still float).
        '''
        if center:
            xx = self.xx + self.dx*self.nx/2
            yy = self.yy + self.dy*self.ny/2
        else:
            xx, yy = self.xx, self.yy
        if crystalunits: xx, yy = xx/self.dx, xx/self.dy
        self._T = f(xx, yy)


    def select(self, **kwargs):
        ''' Performs the appropriate sampling method as specified by self.params.MULTISAMPLING_SCHEME.
            - 'single': strictly speaking, this is the only physically correct sampling scheme. However,
                to improve efficiency while making as small an approximation as possible, the grid and
                Poisson schemes exist for use in the Glauber update scheme.
            - 'grid': built for efficient parallel generation of samples, but this introduces strong grid
                bias in the sample distribution. To remedy this to some extent, using poisson=True randomizes
                the positioning of the supercells, though these are still placed on a supergrid and therefore
                the bias is not completely eliminated. No spatially varying r is possible with this method.
            - 'Poisson': uses a true Poisson disk sampling algorithm to sample some magnets with a minimal
                distance requirement, but is much slower than 'grid' for simulations larger than ~(6r, 6r).
                Also, this does not take into account self.occupation, resulting in a possible loss of
                samples for some spin ices. Not implemented, but possible: spatially dependent r.
            - 'cluster': only to be used in the Wolff update scheme. This is supposed to reduce "critical
                slowing down" in the 2D square-lattice exchange-coupled Ising model.
            @param r [float] (self.calc_r(0.01)): minimal distance between magnets, in meters
            @return [cp.array]: a 2xN array, where the 2 rows represent y- and x-coordinates (!), respectively.
                (this is because the indexing of 2D arrays is e.g. self.m[y,x])
        '''
        match self.params.MULTISAMPLING_SCHEME:
            case 'single': # Strictly speaking, this is the only physically correct sampling scheme
                pos = self._select_single(**kwargs)
            case 'grid':
                pos = self._select_grid(**kwargs)
            case 'Poisson':
                pos = self._select_Poisson(**kwargs)
            case 'cluster':
                pos = self._select_cluster(**kwargs)
        if pos.size > 0:
            valid = np.where(self.occupation[pos[0], pos[1]])[0]
        else: valid = np.zeros(0)
        return pos[:,valid] if valid.size > 0 else self.select(**kwargs) # If at first you don't succeed try, try and try again
    
    def _select_single(self, **kwargs):
        ''' Selects just a single magnet from the simulation domain.
            Strictly speaking, this is the only physically correct sampling scheme.
        '''
        nonzero_y, nonzero_x = cp.nonzero(self.occupation)
        nonzero_idx = cp.random.choice(self.n, 1) # CUPYUPDATE: use hotspin.rng once cp.random.Generator supports choice() method
        return cp.asarray([nonzero_y[nonzero_idx], nonzero_x[nonzero_idx]]).reshape(2, -1)

    def _select_grid(self, r=None, poisson=None, **kwargs):
        ''' Uses a supergrid with supercells of size <r> to select multiple sufficiently-spaced magnets at once.
            ! <r> is a distance in meters, not a number of cells !
            @param r [float] (self.calc_r(0.01)): the minimal distance between two samples.
            @param poisson [bool] (True|False): whether or not to choose the supercells using a poisson-like scheme.
                Using poisson=True is slower (already 2x slower for more than ~15x15 supercells) and places less
                samples at once, but provides a much better sample distribution where the orthogonal grid bias is
                not entirely eliminated but nonetheless significantly reduced as compared to poisson=False.
        '''
        if r is None: r = self.calc_r(0.01)
        if poisson is None: poisson = (15*r)**2 < self.nx*self.ny # use poisson supercells only if there are not too many supercells
        Rx, Ry = math.ceil(r/self.dx) - 1, math.ceil(r/self.dy) - 1 # - 1 because effective minimal distance in grid-method is supercell_size + 1
        Rx, Ry = self.unitcell.x*math.ceil(Rx/self.unitcell.x), self.unitcell.y*math.ceil(Ry/self.unitcell.y) # To have integer number of unit cells in a supercell (necessary for occupation_supercell)
        Rx, Ry = min(Rx, self.nx), min(Ry, self.ny) # No need to make supercell larger than simulation itself
        Rx, Ry = max(Rx, 1 if poisson else 2), max(Ry, 1 if poisson else 2) # Also don't take too small, too much randomness could be removed, resulting in nonphysical behavior

        supercells_nx = self.nx//Rx + 1
        supercells_ny = self.ny//Ry + 1
        if poisson:
            # TODO: with parallel Poisson sampling, generate a lot more samples than necessary at once so we have enough for the next several iterations as well
            supercells_x, supercells_y = cp.asarray(PoissonGrid(supercells_nx, supercells_ny))
        else:
            supercells_x = self.ixx[:supercells_ny:2, :supercells_nx:2].ravel()
            supercells_y = self.iyy[:supercells_ny:2, :supercells_nx:2].ravel()
        n = supercells_x.size # Number of selected supercells

        offset_x, offset_y = np.random.randint(self.nx), np.random.randint(self.ny) # To hide any weird edge effects
        dx, dy = offset_x % Rx, offset_y % Ry # offset in a single supercell
        occupation_supercell = self.occupation[dy:dy+Ry, dx:dx+Rx]
        occupation_nonzero = occupation_supercell.nonzero() # tuple: (array(x_indices), array(y_indices))
        random_nonzero_indices = cp.random.choice(occupation_nonzero[0].size, n) # CUPYUPDATE: use hotspin.rng once cp.random.Generator supports choice() method
        idx_x = supercells_x*Rx + occupation_nonzero[1][random_nonzero_indices]
        idx_y = supercells_y*Ry + occupation_nonzero[0][random_nonzero_indices]

        # Remove the max_x and max_y borders to ensure PBC
        if self.PBC: # Cutoff index to ensure PBC if applicable (max() to ensure that (0,0) supercell is intact)
            ok = (idx_x < max(Rx, self.nx - Rx)) & (idx_y < max(Ry, self.ny - Ry))
            idx_x, idx_y = idx_x[ok], idx_y[ok]

        # Roll the supergrid somewhere randomly (necessary for uniform sampling)
        if Rx < self.nx: idx_x = (idx_x + offset_x) % self.nx
        if Ry < self.ny: idx_y = (idx_y + offset_y) % self.ny
        if idx_x.size != 0:
            return cp.asarray([idx_y.reshape(-1), idx_x.reshape(-1)])
        else:
            return self._select_grid(r) # If no samples survived, try again

    def _select_Poisson(self, r=None, **kwargs): # WARN: does not take occupation into account, so preferably only use on FullASI/IsingASI!
        if r is None: r = self.calc_r(0.01)
        from .poisson import SequentialPoissonDiskSampling, poisson_disc_samples

        # p = SequentialPoissonDiskSampling(self.ny*self.dy, self.nx*self.dx, r, tries=1).fill()
        p = poisson_disc_samples(self.ny*self.dy, self.nx*self.dx, r, k=4)
        points = (np.array(p)/self.dx).astype(int)
        return cp.asarray(points.T)
    
    def _select_cluster(self, **kwargs):
        ''' Selects a cluster based on the Wolff algorithm for the two-dimensional square Ising model.
            This is supposed to alleviate the 'critical slowing down' near the critical temperature.
            Therefore, this should not be used for low or high T, only near the T_C is this useful.
        '''
        if self.get_energy('dipolar', verbose=False) is not None:
            return self._select_cluster_longrange()
        else:
            return self._select_cluster_exchange()

    def _select_cluster_longrange(self):
        # TODO: see if the algorithm can be done efficiently for OOP ASI with long-range interactions
        # because it seems that it is certainly possible (if we define an interaction energy between magnets)
        # but that the limiting factor is the sheer number of 'neighbors' that a long-range interaction creates.
        raise NotImplementedError("Can not (yet?) select Wolff cluster with long-range interactions.")

    def _select_cluster_exchange(self):
        exchange: ExchangeEnergy = self.get_energy('exchange')
        if exchange is None: raise NotImplementedError("Can not select Wolff cluster if there is no exchange energy in the system.")
        neighbors = exchange.local_interaction # self._get_nearest_neighbors()
        seed = tuple(self._select_single().flat)
        cluster = cp.zeros_like(self.m)
        cluster[seed] = 1
        checked = cp.copy(cluster)
        checked[cp.where(self.m != self.m[seed])] = 1
        while cp.any(current_frontier := (1-checked)*signal.convolve2d(cluster, neighbors, mode='same', boundary='wrap' if self.PBC else 'fill')):
            checked[current_frontier != 0] = 1
            bond_prob = 1 - cp.exp(-2*exchange.J/self.kBT*current_frontier) # Swendsen-Wang bond probability
            current_frontier[rng.random(size=cluster.shape) > bond_prob] = 0 # Rejected probabilistically
            cluster[current_frontier != 0] = 1
        return cp.asarray(cp.where(cluster == 1))

    def update(self, *args, **kwargs):
        if cp.any(self.T == 0):
            warnings.warn('Temperature is zero somewhere, so no switch will be simulated to prevent DIV/0 errors.', stacklevel=2)
            return # We just warned that no switch will be simulated, so let's keep our word
        match self.params.UPDATE_SCHEME:
            case 'Néel':
                idx = self._update_Néel(*args, **kwargs)
            case 'Glauber':
                idx = self._update_Glauber(*args, **kwargs)
            case 'Wolff':
                idx = self._update_Wolff(*args, **kwargs)
            case str(unrecognizedscheme):
                raise ValueError(f"The Magnets.params object has an invalid value for UPDATE_SCHEME='{unrecognizedscheme}'.")
        return clean_indices(idx)

    def _update_Glauber(self, idx=None, Q=0.05, r=0):
        # 1) Choose a bunch of magnets at random
        if idx is None: idx = self.select(r=self.calc_r(Q) if r == 0 else r)
        self.attempted_switches += idx.shape[1]
        # 2) Compute the change in energy if they were to flip, and the corresponding Boltzmann factor.
        # OPTION 1: TRUE GLAUBER, NO ENERGY BARRIER (or at least not explicitly)
        # exponential = cp.clip(cp.exp(-self.switch_energy(idx)/self.kBT[idx[0], idx[1]]), 1e-10, 1e10) # clip to avoid inf
        # OPTION 2: AD-HOC ADJUSTED GLAUBER, where we assume that the 'other state' is at the top of the energy barrier
        # TODO: more accurate energy barrier using 4-state approach? (might not be worth the computational cost)
        barrier = cp.maximum((delta_E := self.switch_energy(idx)), self.E_B[idx[0], idx[1]] + delta_E/2) # To correctly account for the situation where energy barrier disappears
        exponential = cp.clip(cp.exp(-barrier/self.kBT[idx[0], idx[1]]), 1e-10, 1e10) # clip to avoid inf
        # 3) Flip the spins with a certain exponential probability. There are two commonly used and similar approaches:
        idx_switch = idx[:,cp.where(rng.random(size=exponential.shape) < exponential)[0]] # Acceptance condition from detailed balance
        # idx = idx[:,cp.where(rng.random(size=exponential.shape) < (exponential/(1+exponential)))[0]] # From https://en.wikipedia.org/wiki/Glauber_dynamics, unsure if this satisfied detailed balance
        if idx_switch.shape[1] > 0:
            self.m[idx_switch[0], idx_switch[1]] *= -1
            self.switches += idx_switch.shape[1]
            self.update_energy(index=idx_switch)
        return idx_switch # TODO: can we get some sort of elapsed time? Yes for one switch, but what about many at once?
    
    def _update_Néel(self, t_max=1, attempt_freq=1e10):
        ''' Performs a single magnetization switch, if the switch would occur sooner than <t_max> seconds. '''
        # TODO: we might be able to multi-switch here as well, by taking into account the double-switch time
        barrier = (self.E_B - self.E)/self.occupation # Divide by occupation to make non-occupied grid cells have infinite barrier
        minBarrier = cp.min(barrier)
        barrier -= minBarrier # Energy is relative, so set min(barrier) to zero (this prevents issues at low T)
        with np.errstate(over='ignore'): # Ignore overflow warnings in the exponential: such high barriers wouldn't switch anyway
            taus = rng.random(size=barrier.shape)*cp.exp(barrier/self.kBT) # Draw random relative times from an exponential distribution
            indexmin2D = divmod(cp.argmin(taus), self.m.shape[1]) # The min(tau) index in 2D form for easy indexing
            time = taus[indexmin2D]*cp.exp(minBarrier/self.kBT[indexmin2D])/attempt_freq # This can become cp.inf quite quickly if T is small
            self.attempted_switches += 1
            if time > t_max:
                self.t += t_max
                return None # Just exit if switching would take ages and ages and ages
            self.m[indexmin2D] *= -1
            self.switches += 1
            self.t += time
        self.update_energy(index=indexmin2D)
        return indexmin2D

    def _update_Wolff(self):
        ''' Only works for Exchange-coupled 2D Ising system. '''
        cluster = self._select_cluster()
        self.m[cluster[0], cluster[1]] *= -1
        self.switches += cluster.shape[1]
        self.attempted_switches += cluster.shape[1]
        self.update_energy(index=cluster)
        return cluster


    def calc_r(self, Q, as_scalar=True) -> float|cp.ndarray:
        ''' Calculates the minimal value of r (IN METERS). Considering two nearby sampled magnets, the switching probability
            of the first magnet will depend on the state of the second. For magnets further than <calc_r(Q)> apart, the switching
            probability of the first will not change by more than <Q> if the second magnet switches.
            @param Q [float]: (0<Q<1) the maximum allowed change in switching probability of a sample if any other sample switches.
            @param as_scalar [bool] (True): if True, a safe value for the whole grid is returned.
                If False, a CuPy array is returned which can for example be used in adaptive Poisson disc sampling.
            @return [float|cp.ndarray]: the minimal distance (in meters) between samples, if their mutual influence on their
                switching probabilities is to be less than <Q>.
        '''
        r = (8e-7*self._momentSq/(Q*self.kBT))**(1/3)
        return float(cp.max(r)) if as_scalar else r # Use max to be safe if requested to be scalar value


    def minimize_single(self, N=1):
        ''' Switches the magnet which has the highest switching probability. Repeat this <N> times. '''
        all_indices = cp.asarray([self.iyy.reshape(-1), self.ixx.reshape(-1)])
        for _ in range(N):
            indexmin2D = divmod(cp.argmin(self.switch_energy(all_indices)), self.nx)
            self.m[indexmin2D] = -self.m[indexmin2D]
            self.switches += 1
            self.attempted_switches += 1
            self.update_energy(index=indexmin2D)
    
    def minimize_all(self, simultaneous=False):
        ''' Switches all magnets that can gain energy by switching.
            By default, several subsets of the domain are updated in a random order, such that in total
            every cell got exactly one chance at switching. This randomness may increase the runtime, but
            reduces systematic bias and possible loops which would otherwise cause self.relax() to get stuck.
            @param simultaneous [bool] (False): if True, all magnets are evaluated simultaneously (nonphysical, error-prone).
                If False, several subgrids are evaluated after each other in a random order to prevent systematic bias.
        '''
        if simultaneous:
            step_x = step_y = 1
        else:
            step_x, step_y = self.unitcell.x*2, self.unitcell.y*2
        order = cp.random.permutation(step_x*step_y) # CUPYUPDATE: use hotspin.rng once cp.random.Generator supports permutation() method
        for i in order:
            y, x = divmod(i, step_x)
            if self.occupation[y, x] == 0: continue
            subset_indices = cp.asarray([self.iyy[y::step_y,x::step_x].reshape(-1), self.ixx[y::step_y,x::step_x].reshape(-1)])
            indices = subset_indices[:, cp.where(self.switch_energy(subset_indices) < 0)[0]]
            if indices.size > 0:
                self.m[indices[0], indices[1]] = -self.m[indices[0], indices[1]]
                self.switches += indices.shape[1]
                self.attempted_switches += indices.shape[1]
                self.update_energy(index=indices)
    
    def relax(self, verbose=False):
        ''' Relaxes self.m to a (meta)stable state using multiple self.minimize_all() calls.
            NOTE: this can be an expensive operation for large simulations.
        '''
        if verbose:
            import matplotlib.pyplot as plt
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            im = ax.imshow(self.m.get())

        # TODO: use a better stopping criterion than just "oh no we exceeded n_max steps", and remove verbose code when all is fixed
        n, n_max = 0, int(math.sqrt(self.nx**2 + self.ny**2)) # Maximum steps is to get signal across the whole grid
        previous_states = [self.m.copy(), cp.zeros_like(self.m), cp.zeros_like(self.m)] # Current state, previous, and the one before
        while not cp.allclose(previous_states[2], previous_states[0]) and n < n_max: # Loop exits once we detect a 2-cycle, i.e. the only thing happening is some magnets keep switching back and forth
            self.minimize_all()
            if verbose:
                im.set_array(self.m.get())
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            previous_states[2] = previous_states[1].copy()
            previous_states[1] = previous_states[0].copy()
            previous_states[0] = self.m.copy()
            n += 1
        if verbose: print(f"Used {n} (out of {n_max}) steps in relax().")


    def history_save(self, *, E_tot=None, t=None, T=None, m_avg=None):
        ''' Records E_tot, t, T and m_avg as they were last calculated. This default behavior can be overruled: if
            passed as keyword parameters, their arguments will be saved instead of the self.<E_tot|t|T|m_avg> value(s).
        '''
        self.history_entry()
        if E_tot is not None: self.history.E[-1] = float(E_tot)
        if t is not None: self.history.t[-1] = float(t)
        if T is not None: self.history.T[-1] = float(cp.mean(T) if isinstance(T, cp.ndarray) else np.mean(T))
        if m_avg is not None: self.history.m[-1] = float(m_avg)
    
    def history_entry(self):
        self.history.entry(self)
    
    def history_clear(self):
        self.history.clear()
    

    def autocorrelation(self, *, normalize=True):
        ''' In case of a 2D signal, this basically calculates the autocorrelation of the
            self.m matrix, while taking into account self.orientation for the in-plane case.
            Note that strictly speaking this assumes dx and dy to be the same for all cells.
        '''
        # Now, convolve self.m with its point-mirrored/180°-rotated counterpart
        if self.in_plane:
            mx, my = self.m*self.orientation[:,:,0], self.m*self.orientation[:,:,1]
            corr_xx = signal.correlate2d(mx, mx, boundary='wrap' if self.PBC else 'fill')
            corr_yy = signal.correlate2d(my, my, boundary='wrap' if self.PBC else 'fill')
            corr = corr_xx + corr_yy # Basically trace of autocorrelation matrix
        else:
            corr = signal.correlate2d(self.m, self.m, boundary='wrap' if self.PBC else 'fill')

        if normalize:
            corr_norm = signal.correlate2d(cp.ones_like(self.m), cp.ones_like(self.m), boundary='wrap' if self.PBC else 'fill') # Is this necessary? (this is number of occurences of each cell in correlation sum)
            corr = corr*self.m.size/self.n/corr_norm # Put between 0 and 1
        
        return corr[(self.ny-1):(2*self.ny-1),(self.nx-1):(2*self.nx-1)]**2
    
    def correlation_length(self, correlation=None):
        if correlation is None: correlation = self.autocorrelation()
        # First calculate the distance between all spins in the simulation.
        rr = (self.xx**2 + self.yy**2)**(1/2) # This only works if dx, dy is the same for every cell!
        return float(cp.sum(cp.abs(correlation) * rr * rr)/cp.max(cp.abs(correlation))) # Do *rr twice, once for weighted avg, once for 'binning' by distance

    ######## Now, some useful functions to overwrite when subclassing this class
    @abstractmethod # Not needed for out-of-plane ASI, but essential for in-plane ASI, therefore abstractmethod anyway
    def _get_angles(self):
        ''' Returns a 2D array containing the angle (measured counterclockwise from the x-axis)
            of each spin. The angle of unoccupied cells (self._get_occupation() == 0) is ignored.
        '''
        return cp.zeros_like(self.ixx) # Example

    @abstractmethod # TODO: should this really be an abstractmethod? Uniform and random can be defaults and subclasses can define more of course
    def _set_m(self, pattern: str):
        ''' Directly sets <self.m>, depending on <pattern>. Usually, <pattern> is "uniform", "vortex", "AFM" or "random".
            <self.m> is a shape (ny, nx) array containing only -1, 0 or 1 indicating the magnetization direction.
        '''
        match str(pattern).strip().lower():
            case 'uniform':
                self.m = self._get_m_uniform()
            case 'vortex':
                # When using 'angle' property of Magnets.initialize_m:
                #   <angle> near 0 or math.pi: clockwise/anticlockwise vortex, respectively
                #   <angle> near math.pi/2 or -math.pi/2: bowtie configuration (top region: up/down, respectively)
                self.m = cp.ones_like(self.xx)
                distSq = ((self.ixx - (self.nx-1)/2)**2 + (self.iyy - (self.ny-1)/2)**2) # Try to put the vortex close to the center of the simulation
                distSq[cp.where(self.occupation == 1)] = cp.nan # We don't want to place the vortex center at an occupied cell
                middle_y, middle_x = divmod(cp.argmax(distSq == cp.min(distSq[~cp.isnan(distSq)])), self.nx) # The non-occupied cell closest to the center
                N = cp.where((self.ixx - middle_x < self.iyy - middle_y) & (self.ixx + self.iyy >= middle_x + middle_y))
                E = cp.where((self.ixx - middle_x >= self.iyy - middle_y) & (self.ixx + self.iyy > middle_x + middle_y))
                S = cp.where((self.ixx - middle_x > self.iyy - middle_y) & (self.ixx + self.iyy <= middle_x + middle_y))
                W = cp.where((self.ixx - middle_x <= self.iyy - middle_y) & (self.ixx + self.iyy < middle_x + middle_y))
                self.m[N] = self._get_m_uniform(0         )[N]
                self.m[E] = self._get_m_uniform(-math.pi/2)[E]
                self.m[S] = self._get_m_uniform(math.pi   )[S]
                self.m[W] = self._get_m_uniform(math.pi/2 )[W]
            case str(unknown_pattern):
                self.m = rng.integers(0, 2, size=self.xx.shape)*2 - 1
                if unknown_pattern != 'random': warnings.warn(f'Pattern "{unknown_pattern}" not recognized, defaulting to "random".', stacklevel=2)
    
    @abstractmethod
    def _get_occupation(self):
        ''' Returns a 2D CuPy array which contains 1 at the cells which are occupied by a magnet, and 0 elsewhere. '''
        return cp.ones_like(self.ixx) # Example
    
    @abstractmethod
    def _get_groundstate(self):
        ''' Returns <pattern> in self.initialize_m(<pattern>) which corresponds to a global ground state of the system.
            Use 'random' if no ground state is implemented in self._set_m().
        '''
        return 'random'

    @abstractmethod
    def _get_nearest_neighbors(self):
        ''' Returns a small mask with the magnet at the center, and 1 at the positions of its nearest neighbors (elsewhere 0). '''
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # Example

    @abstractmethod # TODO: this is a pretty archaic method at this point
    def _get_appropriate_avg(self):
        ''' Returns the most appropriate averaging mask for a given type of ASI. '''
        return 'cross' # Example

    @abstractmethod # TODO: this is a pretty archaic method at this point as well. Sould this really be abstract?
    def _get_AFMmask(self):
        ''' Returns the (normalized) mask used to determine how anti-ferromagnetic the magnetization profile is. '''
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float') # Example


class Energy(ABC):
    def __init__(self):
        ''' The __init__ method contains all initialization of variables which do not depend
            on a specific given Magnets() object. It is not required to override this method.
        '''
        pass
    
    @abstractmethod
    def initialize(self, mm: Magnets):
        ''' Do all the things which can be done when given a certain Magnets() object. '''
        self.mm = mm # Like this
        self.update() # And this, to initialize the energies etc.
    
    @abstractmethod
    def update(self):
        ''' Calculates the entire self.E array, for the situation in self.mm.m. '''
        self.E = cp.zeros_like(self.mm.xx) # [J]

    @abstractmethod
    def energy_switch(self, indices2D):
        ''' Returns the change in energy experienced by the magnets at <indices2D>, if they were to switch.
            @param indices2D [tuple(2)]: A tuple containing two equal-size 1D arrays representing the y- and x-
                indices of the sampled magnets, such that this tuple can be used directly to index self.E.
            @return [list(N)]: A list containing the local changes in energy for each magnet of <indices2D>, in the same order.
        '''

    @abstractmethod
    def update_single(self, index2D):
        ''' Updates self.E by only taking into account that a single magnet (at index2D) switched.
            @param index2D [tuple(2)]: A tuple containing two size-1 arrays representing y- and x-index of the switched magnet.
        '''
    
    @abstractmethod
    def update_multiple(self, indices2D):
        ''' Updates self.E by only taking into account that some magnets (at indices2D) switched.
            This seems like it is just multiple times self.update_single(), but sometimes an optimization is possible,
            hence this required alternative function for updating multiple magnets at once.
            @param indices2D [tuple(2)]: A tuple containing two equal-size 1D arrays representing the y- and x-
                indices of the sampled magnets, such that this tuple can be used directly to index self.E.
        '''

    @property
    @abstractmethod
    def E_tot(self):
        ''' Returns the total energy for this energy contribution. This function is necessary since this is not equal
            for all energies: e.g. sum(E) in the DipolarEnergy would count each interaction twice, while sum(E) is
            correct for ZeemanEnergy.
        '''

    @property
    @cache
    def shortname(self):
        return type(self).__name__.lower().replace('energy', '')


class ZeemanEnergy(Energy):
    def __init__(self, magnitude=0, angle=0):
        ''' This ZeemanEnergy class implements the Zeeman energy for a spatially uniform external field, whose magnitude
            (and angle, if the magnetization is in-plane) can be set using the set_field method.
            @param magnitude [float] (0): The magnitude of the external field.
            @param angle [float] (0): The angle (in radians) of the external field.
        '''
        self.magnitude, self.angle = magnitude, angle # [T], [rad]
    
    def initialize(self, mm: Magnets):
        self.mm = mm
        self.set_field(self.magnitude, self.angle)

    def set_field(self, magnitude=None, angle=None):
        self.magnitude = self.magnitude if magnitude is None else magnitude
        self.angle = self.angle if angle is None else angle
        if self.mm.in_plane:
            self.B_ext = self.magnitude*cp.array([math.cos(self.angle), math.sin(self.angle)]) # [T]
        else:
            self.B_ext = self.magnitude # [T]
            if self.angle != 0:
                self.angle = 0
                warnings.warn(f'You tried to set the angle of an out-of-plane field in ZeemanEnergy.set_field(), but this is not supported.', stacklevel=2)
        self.update()

    def update(self):
        if self.mm.in_plane:
            self.E = -self.mm.moment*cp.multiply(self.mm.m, self.B_ext[0]*self.mm.orientation[:,:,0] + self.B_ext[1]*self.mm.orientation[:,:,1])
        else:
            self.E = -self.mm.moment*self.mm.m*self.B_ext

    def energy_switch(self, indices2D):
        return -2*self.E[indices2D]

    def update_single(self, index2D):
        self.E[index2D] *= -1
    
    def update_multiple(self, indices2D):
        self.E[indices2D] *= -1
    
    @property
    def E_tot(self):
        return cp.sum(self.E)


class DipolarEnergy(Energy):
    def __init__(self, prefactor=1):
        ''' This DipolarEnergy class implements the interaction between the magnets of the simulation themselves.
            It should therefore always be included in the simulations.
            @param prefactor [float] (1): The relative strength of the dipolar interaction.
        '''
        self.prefactor = prefactor

    def initialize(self, mm: Magnets):
        self.mm = mm
        self.unitcell = self.mm.unitcell
        self.E = cp.zeros_like(self.mm.xx)
        
        # Let us first make the four-mirrored distance matrix rinv3
        # WARN: this four-mirrored technique only works if (dx, dy) is the same for every cell everywhere!
        # This could be generalized by calculating a separate rrx and rry for each magnet in a unit cell similar to toolargematrix_o{x,y}
        rrx = self.mm.xx - self.mm.xx[0,0]
        rry = self.mm.yy - self.mm.yy[0,0]
        rr_sq = rrx**2 + rry**2
        rr_sq[0,0] = cp.inf
        rr_inv = cpx.rsqrt(rr_sq) # Due to the previous line, this is now never infinite
        rr_inv3 = rr_inv**3
        rinv3 = mirror4(rr_inv3)
        # Now we determine the normalized rx and ry
        ux = mirror4(rrx*rr_inv, negativex=True)
        uy = mirror4(rry*rr_inv, negativey=True)
        # Now we initialize the full ox
        if self.mm.in_plane:
            unitcell_ox = self.mm.orientation[:self.unitcell.y,:self.unitcell.x,0]
            unitcell_oy = self.mm.orientation[:self.unitcell.y,:self.unitcell.x,1]
        else:
            unitcell_ox = unitcell_oy = cp.zeros((self.unitcell.y, self.unitcell.x))
        num_unitcells_x = 2*math.ceil(self.mm.nx/self.unitcell.x) + 1
        num_unitcells_y = 2*math.ceil(self.mm.ny/self.unitcell.y) + 1
        toolargematrix_ox = cp.tile(unitcell_ox, (num_unitcells_y, num_unitcells_x)) # This is the maximum that we can ever need (this maximum
        toolargematrix_oy = cp.tile(unitcell_oy, (num_unitcells_y, num_unitcells_x)) # occurs when the simulation does not cut off any unit cells)
        # Now comes the part where we start splitting the different cells in the unit cells
        self.kernel_unitcell = [[None for _ in range(self.unitcell.x)] for _ in range(self.unitcell.y)]
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                if self.mm.in_plane:
                    ox1, oy1 = unitcell_ox[y,x], unitcell_oy[y,x] # Scalars
                    if ox1 == oy1 == 0:
                        continue # Empty cell in the unit cell, so keep self.kernel_unitcell[y][x] equal to None
                    # Get the useful part of toolargematrix_o{x,y} for this (x,y) in the unit cell
                    slice_startx = (self.unitcell.x - ((self.mm.nx-1) % self.unitcell.x) + x) % self.unitcell.x # Final % not strictly necessary because
                    slice_starty = (self.unitcell.y - ((self.mm.ny-1) % self.unitcell.y) + y) % self.unitcell.y # toolargematrix_o{x,y} large enough anyway
                    ox2 = toolargematrix_ox[slice_starty:slice_starty+2*self.mm.ny-1,slice_startx:slice_startx+2*self.mm.nx-1]
                    oy2 = toolargematrix_oy[slice_starty:slice_starty+2*self.mm.ny-1,slice_startx:slice_startx+2*self.mm.nx-1]
                    kernel1 = ox1*ox2*(3*ux**2 - 1)
                    kernel2 = oy1*oy2*(3*uy**2 - 1)
                    kernel3 = 3*(ux*uy)*(ox1*oy2 + oy1*ox2)
                    kernel = -(kernel1 + kernel2 + kernel3)*rinv3
                else:
                    kernel = rinv3 # 'kernel' for out-of-plane is very simple

                if self.mm.PBC: # Just copy the kernel 8 times, for the 8 'nearest simulations'
                    kernelcopy = kernel.copy()
                    kernel[:,self.mm.nx:] += kernelcopy[:,:self.mm.nx-1]
                    kernel[self.mm.ny:,self.mm.nx:] += kernelcopy[:self.mm.ny-1,:self.mm.nx-1]
                    kernel[self.mm.ny:,:] += kernelcopy[:self.mm.ny-1,:]
                    kernel[self.mm.ny:,:self.mm.nx-1] += kernelcopy[:self.mm.ny-1,self.mm.nx:]
                    kernel[:,:self.mm.nx-1] += kernelcopy[:,self.mm.nx:]
                    kernel[:self.mm.ny-1,:self.mm.nx-1] += kernelcopy[self.mm.ny:,self.mm.nx:]
                    kernel[:self.mm.ny-1,:] += kernelcopy[self.mm.ny:,:]
                    kernel[:self.mm.ny-1,self.mm.nx:] += kernelcopy[self.mm.ny:,:self.mm.nx-1]

                kernel *= 1e-7 # [J/Am²], 1e-7 is mu_0/4Pi
                self.kernel_unitcell[y][x] = kernel
        self.update()
    
    def update(self):
        total_energy = cp.zeros_like(self.mm.m)
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                kernel = self.kernel_unitcell[y][x]
                if kernel is None:
                    continue
                else:
                    partial_m = cp.zeros_like(self.mm.m)
                    partial_m[y::self.unitcell.y, x::self.unitcell.x] = self.mm.m[y::self.unitcell.y, x::self.unitcell.x]

                    total_energy = total_energy + partial_m*signal.convolve2d(kernel, self.mm.m, mode='valid')*self.mm._momentSq
        self.E = self.prefactor*total_energy
    
    def energy_switch(self, indices2D):
        return -2*self.E[indices2D]
    
    def update_single(self, index2D):
        # First we get the x and y coordinates of magnet <i> in its unit cell
        y, x = index2D
        x_unitcell = int(x) % self.unitcell.x
        y_unitcell = int(y) % self.unitcell.y
        # The kernel to use is then
        kernel = self.kernel_unitcell[y_unitcell][x_unitcell]
        if kernel is not None:
            # Multiply with the magnetization
            usefulkernel = kernel[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
            interaction = self.prefactor*self.mm.m[index2D]*cp.multiply(self.mm.m, usefulkernel)*self.mm._momentSq
        else:
            interaction = cp.zeros_like(self.mm.m)

        self.E += 2*interaction
        self.E[index2D] *= -1 # This magnet switched, so all its interactions are inverted

    def update_multiple(self, indices2D):
        self.E[indices2D] *= -1
        indices2D_unitcell_raveled = (indices2D[1] % self.unitcell.x) + (indices2D[0] % self.unitcell.y)*self.unitcell.x
        binned_unitcell_raveled = cp.bincount(indices2D_unitcell_raveled)
        for i in binned_unitcell_raveled.nonzero()[0]: # Iterate over the unitcell indices present in indices2D
            y_unitcell, x_unitcell = divmod(int(i), self.unitcell.x)
            kernel = self.kernel_unitcell[y_unitcell][x_unitcell]
            if kernel is None: continue # This should never happen, but check anyway in case indices2D includes empty cells
            indices_here = cp.where(indices2D_unitcell_raveled == i)[0]
            indices2D_here = (indices2D[0][indices_here], indices2D[1][indices_here])
            if indices_here.size > self.mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF:
                ### EITHER WE DO THIS (CONVOLUTION) (starts to be better at approx. 40 simultaneous switches for 41x41 kernel, taking into account the need for complete recalculation every <something> steps, so especially for large T this is good)
                switched_field = cp.zeros_like(self.mm.m)
                switched_field[indices2D_here] = self.mm.m[indices2D_here]
                n = self.mm.params.REDUCED_KERNEL_SIZE
                usefulkernel = kernel[self.mm.ny-1-n:self.mm.ny+n, self.mm.nx-1-n:self.mm.nx+n] if n else kernel
                convolvedkernel = signal.convolve2d(switched_field, usefulkernel, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
            else:
                ### OR WE DO THIS (BASICALLY self.update_single BUT SLIGHTLY PARALLEL AND SLIGHTLY NONPARALLEL) 
                convolvedkernel = cp.zeros_like(self.mm.m) # Still is convolved, just not in parallel
                for j in range(indices_here.size): # Here goes the manual convolution
                    y, x = indices2D_here[0][j], indices2D_here[1][j]
                    convolvedkernel += self.mm.m[y,x]*kernel[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
            interaction = self.prefactor*cp.multiply(self.mm.m*self.mm._momentSq, convolvedkernel)
            self.E += 2*interaction
    
    @property
    def E_tot(self):
        return cp.sum(self.E)/2


class ExchangeEnergy(Energy):
    def __init__(self, J=1):
        self.J = J # [J]

    def initialize(self, mm: Magnets):
        self.mm = mm
        self.local_interaction = self.mm._get_nearest_neighbors()
        self.update()
    
    def update(self):
        if self.mm.in_plane: # Use the XY hamiltonian (but spin has fixed axis so model is still Ising-like)
            mx = self.mm.orientation[:,:,0]*self.mm.m
            my = self.mm.orientation[:,:,1]*self.mm.m
            sum_mx = signal.convolve2d(mx, self.local_interaction, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
            sum_my = signal.convolve2d(my, self.local_interaction, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
            self.E = -self.J*(cp.multiply(sum_mx, mx) + cp.multiply(sum_my, my))
        else: # Use Ising hamiltonian
            self.E = -self.J*cp.multiply(signal.convolve2d(self.mm.m, self.local_interaction, mode='same', boundary='wrap' if self.mm.PBC else 'fill'), self.mm.m)

    def energy_switch(self, indices2D):
        return -2*self.E[indices2D]
    
    def update_single(self, index2D):
        self.update() # There are much faster ways of doing this, but this becomes difficult with PBC and in/out-of-plane
    
    def update_multiple(self, indices2D):
        self.update()
    
    @property
    def E_tot(self):
        return cp.sum(self.E)/2


@dataclass
class History:
    ''' Stores the history of the energy, temperature, time, and average magnetization. '''
    E: list = field(default_factory=list) # Total energy
    T: list = field(default_factory=list) # Average temperature
    t: list = field(default_factory=list) # Elapsed time/switches
    m: list = field(default_factory=list) # Average magnetization magnitude

    def entry(self, mm: Magnets):
        self.E.append(mm.E_tot)
        self.T.append(mm.T_avg)
        self.t.append(mm.t if mm.t != 0 else mm.switches)
        self.m.append(mm.m_avg)

    def clear(self):
        self.E.clear()
        self.T.clear()
        self.t.clear()
        self.m.clear()

@dataclass(slots=True)
class Unitcell:
    ''' Stores x and y components, so we don't need to index [0] or [1] in a tuple, which would be unclear. '''
    x: float
    y: float
