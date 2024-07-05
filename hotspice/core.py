__all__ = ['xp', 'kB', 'SimParams', 'Magnets']

import math
import warnings

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from scipy.spatial import distance
from textwrap import dedent
from typing import Literal

from .poisson import PoissonGrid, SequentialPoissonDiskSampling, poisson_disc_samples
from .utils import as_2D_array, asnumpy, check_repetition, Field, full_obj_name, lower_than
from .energies import Energy, DipolarEnergy, ExchangeEnergy
from . import config
if config.USE_GPU:
    import cupy as xp
    from cupyx.scipy import signal
else:
    import numpy as xp
    from scipy import signal

kB = 1.380649e-23


@dataclass(slots=True)
class SimParams:
    SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF: int = 20 # If there are strictly more than <this> switches in a single iteration, a convolution is used, otherwise the energies are just summed.
    REDUCED_KERNEL_SIZE: int = 20 # If nonzero, the dipolar kernel is cropped to an array of shape (2*<this>-1, 2*<this>-1).
    UPDATE_SCHEME: Literal['Néel', 'Metropolis', 'Wolff'] = 'Néel' # Wolff is only available for exchange-coupled ASI
    MULTISAMPLING_SCHEME: Literal['single', 'grid', 'Poisson', 'cluster'] = 'grid' # Only used if UPDATE_SCHEME is 'Metropolis'.
    ENERGY_BARRIER_METHOD: Literal['simple', 'parabolic'] = 'simple' # Determines how intricately the energy barrier is calculated

    def __post_init__(self):
        self.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF = int(self.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF)
        self.REDUCED_KERNEL_SIZE = int(self.REDUCED_KERNEL_SIZE)
        if self.UPDATE_SCHEME not in (allowed := ['Metropolis', 'Néel', 'Wolff']):
            if self.UPDATE_SCHEME == 'Glauber': self.UPDATE_SCHEME = 'Metropolis'
            raise ValueError(f"UPDATE_SCHEME='{self.UPDATE_SCHEME}' is invalid: allowed values are {allowed}.")
        if self.MULTISAMPLING_SCHEME not in (allowed := ['single', 'grid', 'Poisson', 'cluster']):
            raise ValueError(f"MULTISAMPLING_SCHEME='{self.MULTISAMPLING_SCHEME}' is invalid: allowed values are {allowed}.")
        # If a multisampling scheme is incompatible with an update scheme, an error should be raised at runtime, not here.


class Magnets(ABC): # TODO: make it possible to offset the ASI by some amount of cells (e.g. to create open-edge IP_Square)
    def __init__(
        self, nx: int, ny: int, dx: float|Field, dy: float|Field, *,
        T: Field = 300, E_B: Field = 0., moment: Field = None, Msat: Field = 800e3, V: Field = 2e-22,
        pattern: str = None, energies: tuple['Energy'] = None,
        PBC: bool = False, m_perp_factor: float = None,
        major_axis: float = 0, minor_axis: float = None, # Ellipse shape is constant throughout the array because it is baked into the dipolar kernel.
        angle: float = 0., params: SimParams = None, in_plane: bool = False):
        """ The position of magnets is specified using `nx`, `ny`, `dx` and `dy`. Only rectilinear grids are currently allowed.
            The initial configuration of a `Magnets` geometry consists of 3 parts:
                1) `in_plane`: Magnets can be in-plane or out-of-plane: True or False, respectively. Determined by subclassing.
                2) ASI type: Defined through subclasses (pinwheel, kagome, Ising...). This concerns the layout of spins.
                3) `pattern`:  The initial magnetization direction (e.g. up/down) can be 'uniform', 'AFM', 'vortex' or 'random'.
            One can also specify which energies should be considered by passing a tuple of `Energy` objects to `energies`.
                If `energies` is not passed, only the dipolar energy is considered. If a tuple is passed, however, the
                dipolar energy is not automatically added: if still desired, the user should include this themselves.
            The arguments `T`, `E_B`, `moment`, `Msat` and `V` can all be constant (by passing them as a float) or spatially
                varying (by passing them as a shape (`nx`, `ny`) array).
            The magnetic moment can either be specified using the 2 arguments `Msat` (default 800e3 A/m) and `V` (default 2e-22 m³),
                or directly by passing a value to `moment`. The argument `moment` takes precedence over `Msat` and `V`.
            Corrections to the Ising approximation can be applied using `major_axis` and `minor_axis`: when nonzero,
                the dipole interaction will consider the magnets to be ellipses with these major/minor axes.
                If only `major_axis` is specified, the magnet is assumed to be round with radius `major_axis/2`.
            To enable periodic boundary conditions (default: disabled), pass `PBC=True`.
            To add an extra rotation to all spins in addition to `self._get_angles()`, pass `angle` as a float (in radians).
            The parameter `in_plane` should only be used in a `super().__init__()` call when subclassing this class.
        """
        self.rng = xp.random.default_rng() # There is no significant speed difference between XORWOW or MRG32k3a or Philox4x3210
        self.params = SimParams() if params is None else params # This can just be edited and accessed normally since it is just a dataclass # TODO: find a better way to do this, this is not very user-friendly now

        # Initialize properties that are necessary to access by subsequent method calls
        self._energies = list[Energy]() # Don't manually add anything to this, instead call self.add_energy()
        self.in_plane = in_plane
        self.nx, self.ny = int(nx), int(ny)
        self.ixx, self.iyy = xp.meshgrid(xp.arange(0, self.nx), xp.arange(0, self.ny))
        self.dx, self.dy = dx, dy

        # Main initialization steps to create the geometry
        self.occupation = self._get_occupation().astype(bool).astype(int) # Make sure that it is either 0 or 1
        with np.errstate(invalid='ignore', divide='ignore'):
            self._occupation_inf = 1/self.occupation # inf on empty cells, 1 on filled cells. Used for efficient multiplication rather than expensive division.
        self.nonzero = xp.nonzero(self.occupation) # Indices of nonzero elements, precalculated for efficiency
        self._nonzero_array = xp.asarray(self.nonzero).reshape(2, -1) # Nonzero indices in array form, can be more efficient in some cases
        self.n = int(xp.sum(self.occupation)) # Number of magnets in the simulation
        if self.n == 0: raise ValueError(f"Can not initialize {full_obj_name(self)} of size {self.nx}x{self.ny}, as this does not contain any spins.")
        if self.in_plane: self._initialize_ip(angle=angle) # Initialize orientation of each magnet
        self.unitcell = self._get_unitcell() # This needs to be after occupation and initialize_ip, and before any defects are introduced
        self.PBC = PBC
        self.initialize_m(self._get_groundstate() if pattern is None else pattern, update_energy=False)

        # Initialize field-like properties (!!! these need the geometry to exist already, since they have the same shape)
        self.T = T # [K]
        self.E_B = E_B # [J]
        self.moment = Msat*V if moment is None else moment # [Am²] moment is saturation magnetization multiplied by volume
        self.m_perp_factor = in_plane if m_perp_factor is None else m_perp_factor
        self.major_axis = major_axis
        self.minor_axis = major_axis if minor_axis is None else minor_axis # Shorthand: `minor_axis or major_axis`

        # History
        self.t = 0. # [s]
        self.history = History()
        self.switches, self.attempted_switches = 0, 0
        
        # Technical utilities
        self._zeros = xp.zeros_like(self.m)

        # Finally initialize the energies (at the end, since they might require self.orientation etc.)
        if energies is None: # User did not bother to specify energies, so assume just dipolar
            energies = (DipolarEnergy(),) # Use dipolar energy by default
        elif isinstance(energies, Energy): # Only one energy was specified
            energies = (energies,) # But henceforth we assume energies to be an iterable of Energies, so make it so
        else: # Just input sanitization. This is allowed to throw an error because if that happens then the input was weird anyway.
            energies = tuple(energies)
        for energy in energies: self.add_energy(energy)

    def _get_closest_dist(self):
        """ Returns the closest distance between two magnets in the simulation. """
        slice = xp.where(self.occupation[:self.unitcell.y*2,:self.unitcell.x*2]) # We only need at most two unit cells
        pos_x, pos_y = self.xx[slice], self.yy[slice]
        return distance.pdist(asnumpy(xp.asarray([pos_x, pos_y]).T)).min()

    def _get_m_uniform(self, angle=0):
        """ Returns the `self.m` state with all magnets aligned along `angle` as much as possible. """
        angle += 1e-6 # To avoid possible ambiguous rounding for popular angles, if <angle> ⊥ <self.orientation>
        if self.in_plane:
            return self.occupation*(2*((self.orientation[:,:,0]*xp.cos(angle) + self.orientation[:,:,1]*xp.sin(angle)) >= 0) - 1)
        else:
            return xp.ones_like(self.xx)*((xp.floor(angle/math.pi - .5) % 2)*2 - 1)

    def _get_unitcell(self, max_cell=100):
        """ Returns a `Unitcell` containing the number of single grid cells in a unit cell along the x- and y-axis.
            Only works for square-lattice unit cells. # TODO: allow manual assignment of more complex unit cells
            @param max_cell [int] (100): Only unitcells with ux+uy < `max_cell` are considered for performance reasons.
        """
        dx, dy = self.dx.reshape(1, -1), self.dy.reshape(-1, 1)
        for n in range(1, min(self.nx + self.ny + 1, max_cell + 1)): # Test possible unit cells in a triangle-like manner: (1,1), (2,1), (1,2), (3,1), (2,2), (1,3), ...
            for uy in range(max(0, n - self.nx) + 1, min(n, self.ny) + 1): # Don't test unit cells larger than the domain
                ux = n - uy + 1
                if check_repetition(self.orientation if self.in_plane else self.occupation, ux, uy):
                    if check_repetition(dx, ux, 1) and check_repetition(dy, 1, uy):
                        return Unitcell(ux, uy)
        warnings.warn(dedent(f"""
            Could not detect a reasonably sized unit cell. Defaulting to entire domain {self.nx}x{self.ny}.
            For large simulations, this can cause high memory consumption and very poor performance."""), stacklevel=2)
        return Unitcell(self.nx, self.ny)

    def initialize_m(self, pattern: str|Field = 'random', *, angle: float = 0, update_energy: bool = True):
        """ Initializes `self.m` (array of -1, 0 or 1) based on `pattern` and occupation.
            @param pattern [str|utils.Field]: If type str, this can be any of 'random', 'uniform', 'AFM' by default.
            @param angle [float]: Can be used to change the direction of the magnets for a given pattern.
        """
        if isinstance(pattern, str):
            self._set_m(pattern)
        else: # Then pattern is a Field-like object
            try:
                self.m = as_2D_array(pattern, self.shape)
            except (ValueError, TypeError):
                raise ValueError(f"Argument <pattern> could not be parsed as a valid array for <self.m>.")

        self.m = self.m.astype(float) # Need float to multiply correctly with other float arrays
        self.m = xp.sign(self.m) # Put all to -1., 0. or 1.
        self.m = xp.multiply(self.m, self.occupation) # TODO: make occupation editable to allow dynamic removal ("decimation") of magnets (so have self._occupation and a property with setter that takes care of all that needs to be done)
        self.m[xp.where(self._get_m_uniform() != self._get_m_uniform(angle))] *= -1 # Allow 'rotation' for any kind of initialized state
        if update_energy: self.update_energy() # Have to recalculate all the energies since m changed completely

    def _initialize_ip(self, angle: float = 0.):
        """ Initialize the angles of all the magnets (only applicable in the in-plane case).
            This function should only be called by the `Magnets()` class itself, not by the user.
            @param angle [float] (0): the additional angle (in radians) by which every spin in
                the system will be rotated, i.e. in addition to `self._get_angles()`.
        """
        assert self.in_plane, "Can not _initialize_ip() if magnets are not in-plane (in_plane=False)."
        self.angles = (self._get_angles() + angle)*self.occupation
        self.orientation = xp.zeros(self.ixx.shape + (2,))
        self.orientation[:,:,0] = xp.cos(self.angles)*self.occupation
        self.orientation[:,:,1] = xp.sin(self.angles)*self.occupation

    def add_energy(self, energy: 'Energy', exist_ok=False, verbose=True):
        """ Adds an `Energy` object to `self._energies`. This object is stored under its reduced name,
            e.g. `ZeemanEnergy` is stored under 'zeeman'.
            @param energy [Energy]: the energy to be added.
            @param exist_ok [bool] (False): if True, the energy is not added if an energy of its type is already present.
        """
        energy.initialize(self)
        for i, e in enumerate(self._energies):
            if type(e) is type(energy):
                if exist_ok: return e
                if verbose: warnings.warn(f"An instance of {type(energy).__name__} was already included in the simulation, and has now been overwritten.", stacklevel=2)
                self._energies[i] = energy
                return
        self._energies.append(energy)
        return energy

    def remove_energy(self, name: str, verbose=True): # This is faster when using self._energies as dict
        """ Removes the specified energy from `self._energies`.
            @param name [str]: the name of the energy to be removed. Case-insensitive and may or may not include
                the 'energy' part of the class name. Valid options include e.g. 'dipolar', 'DipolarEnergy'...
        """
        name = name.lower().replace('energy', '')
        for i, e in enumerate(self._energies):
            if name == e.shortname:
                self._energies.pop(i)
                return
        if verbose: warnings.warn(f"There is no '{name}' energy associated with this Magnets object. Valid energies are: {[e.shortname for e in self._energies]}", stacklevel=2)

    def has_energy(self, name: str): # Is basically "self.get_energy(name, verbose=False) is not None"
        name = name.lower().replace('energy', '')
        for e in self._energies:
            if name == e.shortname: return True
        return False

    def get_energy(self, name: str, verbose=True): # This is faster when using self._energies as dict
        """ Returns the specified energy from `self._energies`.
            @param name [str]: the name of the energy to be returned. Case-insensitive and may or may not include
                the 'energy' part of the class name. Valid options include e.g. 'dipolar', 'DipolarEnergy'...
            @returns [Energy]: the requested energy object.
        """
        name = name.lower().replace('energy', '')
        for e in self._energies:
            if name == e.shortname: return e
        if verbose: warnings.warn(f"There is no '{name}' energy associated with this Magnets object. Valid energies are: {[e.shortname for e in self._energies]}", stacklevel=2)
        return None

    def update_energy(self, index: xp.ndarray = None): # This is faster when using self._energies as list
        """ Updates all the energies which are currently present in the simulation.
            @param index [xp.array(2, -)] (None): a 2D array with the first row representing the
                y-coordinates and the second representing the x-coordinates of the indices. if specified, only the magnets at these indices are considered in the calculation.
                We need a NumPy or CuPy array (to easily determine its size: if =2, then only a single switch is considered.)
        """
        for energy in self._energies:
            if index is None: # No index specified, so update fully
                energy.update()
            elif index[0].size == 1: # Only a single index is present, so just use update_single
                energy.update_single(index)
            elif index[0].size > 1: # Index is specified and contains multiple ints, so we must use update_multiple
                energy.update_multiple(index)

    def switch_energy(self, indices2D=None):
        """ @param indices2D [xp.array(2, -)] (None): a 2D array with the first row representing the
                y-coordinates and the second representing the x-coordinates of the indices.
            @return [xp.array]: the change in energy for each magnet in indices2D, in the same order, if they were to switch.
        """
        return sum(energy.energy_switch(indices2D) for energy in self._energies)
    
    def perp_energy(self, indices2D=None):
        """ @param indices2D [xp.array(2, -)] (None): a 2D array with the first row representing the
                y-coordinates and the second representing the x-coordinates of the indices.
            @return [xp.array]: the energy of the ↺ metastable state for each magnet in indices2D, in the same order.
        """
        if not self.USE_PERP_ENERGY:
            return self._zeros if indices2D is None else np.zeros(indices2D.shape[1])
        return self.m_perp_factor*sum([
            energy.E_perp if indices2D is None else energy.E_perp[indices2D[0], indices2D[1]]
            for energy in self._energies])
    
    @property
    def USE_PERP_ENERGY(self):
        return bool(self.m_perp_factor) and self.in_plane # Always return false if OOP
    @property
    def m_perp_factor(self):
        return self._m_perp_factor
    @m_perp_factor.setter
    def m_perp_factor(self, value): # We assume this is not called very often
        if not hasattr(self, '_m_perp_factor'): self._m_perp_factor = 0
        if not isinstance(value, (int, float, bool)): raise ValueError("m_perp_factor must be scalar. It is not currently supported as a 2D array.")
        if value and not self.in_plane: warnings.warn("You set a nonzero `m_perp_energy`, but this has no effect for an OOP ASI.")
        USED_PERP_ENERGY = self.USE_PERP_ENERGY
        self._m_perp_factor = float(value)
        if not USED_PERP_ENERGY and self.USE_PERP_ENERGY:
            self.update_energy() # So suddenly we will be using E_perp. But for performance reasons it was not updated while m_perp_factor=0, so recalculate the whole array.

    @property
    def t(self): # Elapsed time
        return float(self._t) # Need this as @property to guarantee that self.t is always a float
    @t.setter
    def t(self, value):
        self._t = float(value)

    @property
    def MCsteps(self): # Number of Monte Carlo steps
        return self.attempted_switches/self.n

    def reset_stats(self):
        self.t = 0
        self.switches = 0
        self.attempted_switches = 0 # This sets MCsteps automatically

    @property
    def E(self) -> xp.ndarray: # This could be relatively expensive to calculate, so maybe not ideal
        return sum([energy.E for energy in self._energies])

    @property
    def E_perp(self) -> xp.ndarray: # NOTE: E_perp should be defined as "energy if the spin were rotated 90°↺"
        """ If only part of the array is needed, use `self.perp_energy(idx)` instead. """
        return sum([energy.E_perp for energy in self._energies])*self.m_perp_factor
    
    @property
    def H_eff(self) -> xp.ndarray: # NOTE: E_perp should be activated
        """ Returns the effective field (H_x, H_y) at the position of each magnet. """
        if not self.USE_PERP_ENERGY: return np.zeros_like(self.m), np.zeros_like(self.m)
        E_1, E_perp = self.E, sum([energy.E_perp for energy in self._energies]) #! Don't use self.E_perp, that is already scaled by m_perp_factor!
        moment = np.where(self.moment, self.moment, np.inf)
        with np.errstate(invalid='ignore', divide='ignore'):
            a = np.where(E_1 != 0, E_1*xp.sqrt(1 + (E_perp/E_1)**2), np.abs(E_perp))/moment # [T] Field strength
            b = xp.arctan(E_perp/E_1) # Field angle w.r.t. magnetization direction (if E_1 = 0: yields sign(E_perp)*np.pi/2)
        angle = b + self.angles + (self.m + 1)*np.pi/2
        H_x, H_y = a*xp.cos(angle), a*xp.sin(angle)
        return H_x, H_y

    @property
    def T(self) -> xp.ndarray:
        return self._T
    @T.setter
    def T(self, value: Field):
        self._T = as_2D_array(value, self.shape)
        self.kBT = kB*self._T # Array representing the thermal energy k_B*T throughout the system.
        self.beta = 1/self.kBT # Array representing the reciprocal thermal energy 1/kBT throughout the system.

    @property
    def E_B(self) -> xp.ndarray:
        return self._E_B
    @E_B.setter
    def E_B(self, value: Field):
        self._E_B = as_2D_array(value, self.shape)
        with np.errstate(divide='ignore'): # For zero-E_B case (nonphysical but better to warn that in a more obvious way than here)
            self._E_B_inv_over_four = .25/self._E_B # Precalculated for slightly faster E_barrier calculation

    @property
    def moment(self) -> xp.ndarray:
        return self._moment
    @moment.setter
    def moment(self, value: Field):
        self._moment = as_2D_array(value, self.shape)
        self._momentSq = self._moment*self._moment

    @property
    def shape(self):
        return self.m.shape

    @property
    def PBC(self):
        return self._PBC
    @PBC.setter
    def PBC(self, value: bool):
        if hasattr(self, 'unitcell'): # First check if PBC are even possible with the current nx, ny and unitcell
            if value and (self.nx % self.unitcell.x != 0 or self.ny % self.unitcell.y != 0): # Unit cells don't match along the edges
                # We want an integer number of unit cells (if PBC should be True) otherwise all sorts of boundary issues will occur all throughout the code.
                ideal_nx = self.unitcell.x*math.ceil(self.nx/self.unitcell.x)
                ideal_ny = self.unitcell.y*math.ceil(self.ny/self.unitcell.y)
                # Due to the way PBC are initialized etc., it is not easily possible to adjust nx and ny automatically. :(
                warnings.warn(dedent(f"""
                    WARNING: PBC can not be enabled, because there is a non-integer number of unit cells along the x- and/or y-axis,
                             so the boundaries do not match (unit cell has size {self.unitcell.x}x{self.unitcell.y}). Create a new instance of this class with 
                             nx and ny an appropriate multiple of this to allow PBC (e.g. {self.nx}x{self.ny} -> {ideal_nx}x{ideal_ny})."""), stacklevel=2)
                self._PBC = False
                return
        self._PBC = bool(value)
        for energy in self._energies: energy.initialize(self)

    @property
    def dx(self):
        """ Spacing between cells along the x-axis, in meters. """
        return self._dx
    @dx.setter
    def dx(self, value: float):
        """ When specifying `dx` as an array (instead of a single scalar), then it should specify
            the elements [x1-x0, x2-x1, ..., x(ux)-x(ux-1)], if a unitcell has size (ux, uy).
            Note that the last element is the spacing over the boundaries of a unit cell!
        """
        value = xp.asarray(value).reshape(-1)
        self._dx = xp.tile(value, math.ceil(self.nx/value.size))[:self.nx]
        self.x = xp.zeros(self.nx)
        self.x[1:] = xp.cumsum(self._dx)[:-1]
        self.xx, _ = xp.meshgrid(self.x, xp.empty(self.ny))
        self.x_min, self.x_max = float(self.x[0]), float(self.x[-1])
        for energy in self._energies: energy.initialize(self)
    
    @property
    def dy(self):
        """ Spacing between cells along the y-axis, in meters. """
        return self._dy
    @dy.setter
    def dy(self, value: float):
        """ When specifying `dy` as an array (instead of a single scalar), then it should specify
            the elements [y1-y0, y2-y1, ..., y(uy)-y(uy-1)], if a unitcell has size (ux, uy).
            Note that the last element is the spacing over the boundaries of a unit cell!
        """
        value = xp.asarray(value).reshape(-1)
        self._dy = xp.tile(value, math.ceil(self.ny/value.size))[:self.ny]
        self.y = xp.zeros(self.ny)
        self.y[1:] = xp.cumsum(self._dy)[:-1]
        _, self.yy = xp.meshgrid(xp.empty(self.nx), self.y)
        self.y_min, self.y_max = float(self.y[0]), float(self.y[-1])
        for energy in self._energies: energy.initialize(self)

    @property
    def m_avg_x(self) -> float:
        return float(xp.mean(xp.multiply(self.m, self.orientation[:,:,0])))*self.nx*self.ny/self.n if self.in_plane else 0
    @property
    def m_avg_y(self) -> float:
        return float(xp.mean(xp.multiply(self.m, self.orientation[:,:,1])))*self.nx*self.ny/self.n if self.in_plane else 0
    @property
    def m_avg(self) -> float:
        return (self.m_avg_x**2 + self.m_avg_y**2)**(1/2) if self.in_plane else float(xp.mean(self.m))
    @property
    def m_avg_angle(self) -> float: # If m_avg=0, this is 0. For OOP ASI, this is 0 if m_avg > 0, and Pi if m_avg < 0
        return math.atan2(self.m_avg_y, self.m_avg_x) if self.in_plane else 0. + (self.m_avg < 0)*math.pi

    @property
    def E_tot(self) -> float:
        return float(sum([e.E_tot for e in self._energies]))

    @property
    def T_avg(self) -> float:
        return float(xp.mean(self.T))

    @property
    def E_B_avg(self) -> float:
        return float(xp.mean(self.E_B))

    @property
    def moment_avg(self) -> float:
        return float(xp.mean(self.moment))

    def set_T(self, f, /, *, center=False, crystalunits=False):
        """ Sets the temperature field according to a spatial function. To simply set the temperature array, use `self.T = ` instead.
            @param f [function(x,y)->T]: x and y in meters yield T in Kelvin (should accept CuPy/NumPy arrays as x and y).
            @param center [bool] (False): if True, the origin is put in the middle of the simulation,
                otherwise it is in the usual lower left corner.
            @param crystalunits [bool] (False): if True, x and y are assumed to be a number of cells instead of a distance in meters (but still float).
        """
        xx = self.ixx if crystalunits else self.xx
        yy = self.iyy if crystalunits else self.yy
        len_x = self.nx - 1 if crystalunits else self.x_max - self.x_min
        len_y = self.ny - 1 if crystalunits else self.y_max - self.y_min
        self._T = f(xx - center*len_x/2, yy - center*len_y/2)


    def select(self, **kwargs):
        """ Performs the appropriate sampling method as specified by `self.params.MULTISAMPLING_SCHEME`.
            - 'single': strictly speaking, this is the only physically correct sampling scheme. However,
                to improve efficiency while making as small an approximation as possible, the grid and
                Poisson schemes exist for use in the Metropolis update scheme.
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
            @return [xp.array]: a 2xN array, where the 2 rows represent y- and x-coordinates (!), respectively.
                (this is because the indexing of 2D arrays is e.g. `self.m[y,x]`)
        """
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
        """ Selects just a single magnet from the simulation domain.
            Strictly speaking, this is the only physically correct sampling scheme.
        """
        idx = np.random.randint(self.n) # MUCH faster than cp.random
        return self._nonzero_array[:,idx].reshape(2,-1)

    def _select_grid(self, r=None, poisson=None, **kwargs):
        """ Uses a supergrid with supercells of size `r` to select multiple sufficiently-spaced magnets at once.
            ! <r> is a distance in meters, not a number of cells !
            @param r [float] (self.calc_r(0.01)): the minimal distance between two samples.
            @param poisson [bool] (True|False): whether or not to choose the supercells using a poisson-like scheme.
                Using `poisson=True` is slower (already 2x slower for more than ~15x15 supercells) and places less
                samples at once, but provides a much better sample distribution where the orthogonal grid bias is
                not entirely eliminated but nonetheless significantly reduced as compared to `poisson=False`.
        """
        ## Appropriately determine r, Rx, Ry
        if r is None: r = self.calc_r(0.01)
        # if poisson is None: poisson = (15*r)**2 < self.nx*self.ny # use poisson supercells only if there are not too many supercells
        if poisson is None: poisson = False # TODO: when parallel Poisson is implemented, uncomment the line above etc.
        Rx, Ry = math.ceil(r/xp.min(self.dx)) - 1, math.ceil(r/xp.min(self.dy)) - 1 # - 1 because effective minimal distance in grid-method is supercell_size + 1
        Rx, Ry = self.unitcell.x*math.ceil(Rx/self.unitcell.x), self.unitcell.y*math.ceil(Ry/self.unitcell.y) # To have integer number of unit cells in a supercell (necessary for occupation_supercell)
        Rx, Ry = min(Rx, self.nx), min(Ry, self.ny) # No need to make supercell larger than simulation itself
        Rx, Ry = max(Rx, 1 if poisson else 2), max(Ry, 1 if poisson else 2) # Also don't take too small, too much randomness could be removed, resulting in nonphysical behavior

        ## Determine size and indices of relevant supercells
        supercells_nx = self.nx//Rx + 1
        supercells_ny = self.ny//Ry + 1
        if poisson:
            # TODO: with parallel Poisson sampling, generate a lot more samples than necessary at once so we have enough for the next several iterations as well
            supercells_x, supercells_y = xp.asarray(PoissonGrid(supercells_nx, supercells_ny))
        else:
            supercells_x = self.ixx[:supercells_ny:2, :supercells_nx:2].ravel()
            supercells_y = self.iyy[:supercells_ny:2, :supercells_nx:2].ravel()
        n = supercells_x.size # Total number of selected supercells

        ## Move the supergrid to improve uniformity of random selection
        if self.PBC:
            offset_x, offset_y = np.random.randint(self.nx), np.random.randint(self.ny) # To hide all edge effects
        else:
            offset_x, offset_y = np.random.randint(-Rx, Rx), np.random.randint(-Ry, Ry) # Enough to fill gaps between supercells
        if Rx == self.nx: offset_x = 0
        if Ry == self.ny: offset_y = 0

        ## Select an occupied cell in each supercell
        dx, dy = offset_x % self.unitcell.x, offset_y % self.unitcell.y # offset in a single supercell
        occupation_supercell = self.occupation[dy:dy+Ry, dx:dx+Rx] # occupation of a single supercell to choose only relevant cells
        occupation_nonzero = occupation_supercell.nonzero() # tuple: (array(x_indices), array(y_indices))
        # TODO: is it faster to use np.random.choice here?
        random_nonzero_indices = xp.random.choice(occupation_nonzero[0].size, n) # CUPYUPDATE: use hotspice.rng once cp.random.Generator supports choice() method
        idx_x = supercells_x*Rx + occupation_nonzero[1][random_nonzero_indices]
        idx_y = supercells_y*Ry + occupation_nonzero[0][random_nonzero_indices]

        ## Add the offset to the selected indices and wrap/cut appropriately
        if self.PBC:
            # We remove a region of width R along x and y to ensure sufficient distance between all samples when taking PBC into account
            ok = (idx_x < max(Rx, self.nx - Rx)) & (idx_y < max(Ry, self.ny - Ry)) #! max() to ensure that (0,0) supercell remains intact
            idx_x, idx_y = idx_x[ok], idx_y[ok]
            idx_x = (idx_x + offset_x) % self.nx #! It is important to offset AFTER slicing the [ok] indices
            idx_y = (idx_y + offset_y) % self.ny
        else:
            idx_x = idx_x + offset_x
            idx_y = idx_y + offset_y
            ok = (idx_x < self.nx) & (idx_y < self.ny) & (idx_x >= 0) & (idx_y >= 0)
            idx_x, idx_y = idx_x[ok], idx_y[ok] #! It is important to slice AFTER the offset (contrary to PBC=True)

        if idx_x.size != 0:
            return xp.asarray([idx_y.reshape(-1), idx_x.reshape(-1)])
        else:
            return self._select_single() # If no samples survived, just select a single one at random

    def _select_Poisson(self, r=None, **kwargs): # WARN: does not take occupation into account, so preferably only use on OOP_Square/IP_Ising!
        if r is None: r = self.calc_r(0.01)

        # p = SequentialPoissonDiskSampling(self.y_max, self.x_max, r, tries=1).fill()
        p = np.asarray(poisson_disc_samples(self.y_max, self.x_max, r, k=4))
        
        def nearest_grid_indices(grid_1D, coords_1D): # TODO: this method is flawed because if some dx are very small they will almost never be visited
            return np.abs(grid_1D[:, np.newaxis] - coords_1D).argmin(axis=0)
        points = np.column_stack((nearest_grid_indices(self.x, p[:,0] + self.x_min),
                                  nearest_grid_indices(self.y, p[:,1] + self.y_min)))
        return xp.asarray(points.T)

    def _select_cluster(self, **kwargs):
        """ Selects a cluster based on the Wolff algorithm for the two-dimensional square Ising model.
            This is supposed to alleviate the 'critical slowing down' near the critical temperature.
            Therefore, this should not be used for low or high T, only near the T_C is this useful.
        """
        if self.get_energy('dipolar', verbose=False) is not None:
            return self._select_cluster_longrange()
        else:
            return self._select_cluster_exchange()

    def _select_cluster_longrange(self):
        # TODO: see if the algorithm can be done efficiently for OOP ASI with long-range interactions
        # because it seems that it is certainly possible (if we define an interaction energy between magnets)
        # but that the limiting factor is the sheer number of 'neighbors' that a long-range interaction creates.
        raise NotImplementedError("Can not (yet?) select Wolff cluster if long-range interactions are present.")

    def _select_cluster_exchange(self):
        exchange: ExchangeEnergy = self.get_energy('exchange')
        if exchange is None: raise NotImplementedError("Can not select Wolff cluster if there is no exchange energy in the system.")
        neighbors = exchange.local_interaction
        seed = tuple(self._select_single().flat)
        cluster = xp.zeros_like(self.m)
        cluster[seed] = 1
        checked = xp.copy(cluster)
        checked[xp.where(self.m != self.m[seed])] = 1
        while xp.any(current_frontier := (1-checked)*signal.convolve2d(cluster, neighbors, mode='same', boundary='wrap' if self.PBC else 'fill')):
            checked[current_frontier != 0] = 1
            bond_prob = 1 - xp.exp(-2*exchange.J/self.kBT*current_frontier) # Swendsen-Wang bond probability
            current_frontier[self.rng.random(size=cluster.shape) > bond_prob] = 0 # Rejected probabilistically
            cluster[current_frontier != 0] = 1
        return xp.asarray(xp.where(cluster == 1))

    def update(self, *args, **kwargs):
        """ Runs a single update step using the update scheme and magnet selection method in `self.params`. """
        can_handle_zero_temperature = ['Metropolis']
        if xp.any(self.T == 0):
            if self.params.UPDATE_SCHEME not in can_handle_zero_temperature:
                warnings.warn("T=0 somewhere in the domain, so no switch will be simulated to prevent DIV/0 errors.", stacklevel=2)
            else:
                warnings.warn("T=0 somewhere in the domain, so ergodicity can not be guaranteed. Proceed with caution.", stacklevel=2)
            return # We just warned that no switch will be simulated, so let's keep our word
        match self.params.UPDATE_SCHEME:
            case 'Néel':
                idx = self._update_Néel(*args, **kwargs)
            case 'Metropolis':
                idx = self._update_Metropolis(*args, **kwargs)
            case 'Wolff':
                idx = self._update_Wolff(*args, **kwargs)
            case str(unrecognizedscheme):
                raise ValueError(f"The Magnets.params object has an invalid value for UPDATE_SCHEME='{unrecognizedscheme}'.")
        return idx


    def E_barrier(self, idx=None, min_only=False): # E_B is intrinsic anisotropy, E_barrier is effective barrier (dependent on neighboring magnets, external field, ...)
        """ @param `idx`: (2,N)-array with rows representing y- and x-coordinates, respectively.
            @param `min_only` [bool] (False): If True, only the smallest of the two energy barriers is returned.
                Otherwise a tuple with the two energy barriers is returned (for clockwise/counterclockwise rotation, respectively).
            @return: 1D array (if `min_only` is True) or tuple with two 1D arrays (if `min_only` is False),
                with the same length as `idx.shape[1]`. If `idx` is None, then it will be 2D arrays of the same shape as `self.m`.
            Example: if `idx` is not None and `min_only` is False, a return value could be (array(1.4, -0.3, 0.6 ...), array(2.5, -0.12, 0.2 ...)).
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            delta_E = self.switch_energy(idx)*(self._occupation_inf if idx is None else self._occupation_inf[idx[0], idx[1]])
            E = -delta_E/2 # Don't use self.E, because that would need additional indexing
            E_B = self.E_B if idx is None else self.E_B[idx[0], idx[1]]
            E_B_inv_over_four = self._E_B_inv_over_four if idx is None else self._E_B_inv_over_four[idx[0], idx[1]]

        if not self.USE_PERP_ENERGY: # Use simplified calculation, taken from old Néel/Metropolis
            # THERE ARE TWO CHOICES OF FORMULA, DEPENDING ON THE INTERPRETATION OF THE CASE WHERE THE ENERGY BARRIER DISAPPEARS:
            match self.params.ENERGY_BARRIER_METHOD:
                case 'simple':
                    barrier = xp.maximum(delta_E, E_B - E) # THIS FORMULA IS FULLY BACKWARDS COMPATIBLE (pre-2024), and in a sense uses the 'curvature' of the energy landscape for cases where it is <0
                case 'parabolic': # This is the best that can be done without E_perp. (is also exact for OOP ASI because 
                    barrier = xp.where(delta_E > 4*E_B, delta_E, xp.minimum(E_B*(delta_E*E_B_inv_over_four + 1)**2, delta_E + 4*E_B)) # PARABOLIC FORMULA, in the region without real barrier it rises/drops equally fast as delta_E, this is correct if one assumes the effective field is along the easy axis (which is never the case if E_perp != 0).
            # TODO: Determine what is the best approach for the 'barrier < 0' case. barrier=delta_E? barrier=curvature? barrier=0?
            # barrier = xp.where(E_B > E_highest_state, E_B - E, delta_E) # This formula has a discontinuity as soon as barrier < 0, because then the opposite state is used (which is always suddenly much lower), thus ensuring that barrierless magnets will always switch before magnets with even the tiniest barrier.
            return barrier if min_only else (barrier, barrier)
        
        E_perp = self.perp_energy(idx) # Better than self.E_perp because this only calculates for relevant idx
        E_total_perp1, E_total_perp2 = E_B + E_perp, E_B - E_perp
        E_barrier_1 = xp.maximum(delta_E, E_total_perp1 - E)
        E_barrier_2 = xp.maximum(delta_E, E_total_perp2 - E)
        # E_barrier_1 = xp.where(E_total_perp1 > E_highest_state, E_total_perp1 - E, delta_E) # If energy 'barrier' is actually a pit, then take the switch_energy as the 'barrier'. This is allowed because,
        # E_barrier_2 = xp.where(E_total_perp2 > E_highest_state, E_total_perp2 - E, delta_E) # statistically speaking, the magnet will be canted to the lowest energy state most of the time.
        return xp.minimum(E_barrier_1, E_barrier_2) if min_only else (E_barrier_1, E_barrier_2)


    def _update_Metropolis(self, idx=None, Q=0.05, r=0, attempt_freq=1e10): # TODO: flag to choose which kind of exponential to use: clip(0, exp, 1) or exp/(1+exp)
         # TODO: implement maximum elapsed time. (t_max)
        # 1) Choose a bunch of magnets at random
        if idx is None: idx = self.select(r=self.calc_r(Q) if r == 0 else r)
        self.attempted_switches += idx.shape[1]
        beta = self.beta[idx[0], idx[1]]
        # 2) Compute the change in energy if they were to flip, and the corresponding Boltzmann factor.
        with np.errstate(divide='ignore', over='ignore'): # To allow low T or even T=0 without warnings or errors
            exponential = xp.clip(xp.exp(-self.switch_energy(idx)*beta), 1e-10, 1e10) # clip to avoid inf
        # 3) Flip the spins with a certain exponential probability. There are two commonly used and similar approaches:
        idx_switch = idx[:,xp.where(self.rng.random(size=exponential.shape) < exponential)[0]] # METROPOLIS-HASTINGS acceptance probability, derived from detailed balance: min(1, e^-E/kT)
        # idx_switch = idx[:,xp.where(self.rng.random(size=exponential.shape) < (exponential/(1+exponential)))[0]] # GLAUBER acceptance probability, from https://en.wikipedia.org/wiki/Glauber_dynamics: e^-E/kT/(1+e^-E/kT) (Should give same statistics as Metropolis, but is just slower, might look "more natural" according to some)
        if idx_switch.shape[1] > 0:
            self.m[idx_switch[0], idx_switch[1]] *= -1
            self.switches += idx_switch.shape[1]
            self.update_energy(index=idx_switch)
            self.t += xp.sum(-xp.log(self.rng.random())/attempt_freq/self.n*xp.exp(-self.E_barrier(idx, min_only=True)*beta))
        return idx_switch

    def _update_Néel(self, t_max=1, attempt_freq=1e10):
        """ Performs a single magnetization switch, if the switch would occur sooner than `t_max` seconds.
            Returns the index of the magnet that switched, or None if none switched within `t_max`.
        """
        # TODO: is there a way to multi-switch here as well, without a danger of creating infinite loops?
        # TODO: is there a way to move faster in time if there is a magnet that keeps switching back and forth, which thus inhibits further evolution of the system?
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'): # Ignore the divide-by-zero warning that will follow, as it is intended. Also ignore overflow warnings in the exponential because such high barriers wouldn't switch anyway.
            if self.USE_PERP_ENERGY:
                barrier1, barrier2 = self.E_barrier()
                minBarrier = min(xp.nanmin(barrier1), xp.nanmin(barrier2)) # Energy is relative, so set min(barrier) to zero (this prevents issues at low T)
                barrier1 -= minBarrier
                barrier2 -= minBarrier
                frequency = xp.exp(-barrier1*self.beta) + xp.exp(-barrier2*self.beta) # Sadly we have to divide in the next step, and that is a costly function.
                taus = self.rng.exponential(scale=1/frequency) # Draw random relative times from an exponential distribution.
                attempt_freq /= 2 # Because we have 2 parallel switching channels, and we want equivalence between USE_PERP_ENERGY True and False in case of equal barriers.
            else:
                barrier = self.E_barrier(min_only=True)
                minBarrier = xp.nanmin(barrier)
                barrier -= minBarrier # Energy is relative, so set min(barrier) to zero (this prevents issues at low T)
                taus = self.rng.exponential(scale=xp.exp(barrier*self.beta)) # Draw random relative times from an exponential distribution
            indexmin2D = divmod(xp.nanargmin(taus), self.m.shape[1]) # The min(tau) index in 2D form for easy indexing
            time = taus[indexmin2D]*xp.exp(minBarrier*self.beta[indexmin2D])/attempt_freq # This can become infinite quite quickly if T is small
            self.attempted_switches += 1
            if time > t_max or np.isnan(time):
                self.t += t_max
                return None # Just exit if switching would take ages and ages and ages
            self.m[indexmin2D] *= -1
            self.switches += 1
            self.t += float(time)
        self.update_energy(index=xp.asarray(indexmin2D).reshape(2,-1))
        return indexmin2D

    def _update_Wolff(self):
        """ Only works for Exchange-coupled 2D Ising system. """
        cluster = self._select_cluster()
        self.m[cluster[0], cluster[1]] *= -1
        self.switches += cluster.shape[1]
        self.attempted_switches += cluster.shape[1]
        self.update_energy(index=cluster)
        return cluster

    @lru_cache
    def calc_r(self, Q, as_scalar=True) -> float|xp.ndarray:
        """ Calculates the minimal value of r (IN METERS): considering two nearby sampled magnets, the switching probability
            of the first magnet will depend on the state of the second. For magnets further than `calc_r(Q)` apart, the switching
            probability of the first will not change by more than `Q` if the second magnet switches.
            @param Q [float]: (0<Q<1) the maximum allowed change in switching probability of a sample if any other sample switches.
            @param as_scalar [bool] (True): if True, a safe value for the whole grid is returned.
                If False, an array is returned which can for example be used in adaptive Poisson disc sampling.
            @return [float|xp.ndarray]: the minimal distance (in meters) between samples, if their mutual influence on their
                switching probabilities is to be less than `Q`.
        """
        r = (8e-7*self._momentSq/(Q*self.kBT))**(1/3)
        r = xp.clip(r, 0, (self.x_max - self.x_min) + (self.y_max - self.y_min)) # If T=0, we clip the infinite r back to a high value nx*dx+ny*dy which is slightly larger than the simulation
        return float(xp.max(r)) if as_scalar else r # Use max to be safe if requested to be scalar value


    def progress(self, t_max: float = 1, MCsteps_max: float = 4., stepwise: bool = False, **kwargs):
        #! For backwards compatibility, <stepwise> default must be False!
        """ Runs as many self.update steps as is required to progress a certain amount of
            time or Monte Carlo steps (whichever is reached first) into the future.
            **kwargs get passed to `self.update()`.
            @param t_max [float] (None): The maximum amount of time difference between start and end of this function.
            @param MCsteps_max [float] (4.): The maximum amount of Monte Carlo steps performed during this function.
            @return [tuple(2)|generator]:
                If `stepwise` is False: tuple (elapsed time, number of MC steps performed) during the execution of this function.
                If `stepwise` is True: a generator that yields after every `self.update()` call. Can be used to inspect transients etc.
        """
        generator = self._progress_stepwise(t_max=t_max, MCsteps_max=MCsteps_max, **kwargs)
        if stepwise: return generator
        while True: # If we get here, then stepwise is False so we don't want the generator behavior, so we do next() until the end and return the final value
            try: next(generator) # Do next() until the generator stops
            except StopIteration as e: return e.value # And then this is how you can return the 'return' value of a generator as if it were a normal function

    def _progress_stepwise(self, t_max: float = 1, MCsteps_max: float = 4., **kwargs):
        """ Generator that forms the foundation of `self.progress()`. Yields after every `self.update()`. """
        t_0, MCsteps_0 = self.t, self.MCsteps
        while lower_than(dt := self.t - t_0, t_max) and lower_than(dMCsteps := self.MCsteps - MCsteps_0, MCsteps_max):
            if self.params.UPDATE_SCHEME == 'Néel': self.update(t_max=(t_max - dt), **kwargs) # A max time can be specified
            else: self.update(**kwargs) # No time
            yield
        return self.t - t_0, self.MCsteps - MCsteps_0

    # TODO: clean the minimize(), _minimize_all() and relax() functions (i.e. verbosity and standardize barrier calculation)
    def minimize(self, ignore_barrier=True, verbose=False, _frac=0.3):
        """ NOTE: this is basically the sequential version of `self.relax()`.
            NOTE: this is not deterministic.
            Switches the magnet which has the highest switching probability.
            Repeats this until there are no more magnets that can gain energy from switching.
            NOTE: it can take a while for large simulations to become fully relaxed, especially with this
                sequential procedure. Consider using `self.relax()` for large simulations.
        """
        visited = np.zeros_like(self.m) # Used to make sure we don't get an infinite loop
        while True:
            if ignore_barrier:
                barrier = self.switch_energy()
            else:
                barrier = self.E_barrier()
            if (n := (nobarrier := xp.where(barrier < -self.kBT))[0].size) != 0: # Then some magnet(s) simply has/have no barrier at all for switching
                # Randomly switch <frac> of the no-barrier magnets, this is because if we switch them all we might end up in an infinite loop
                rand = np.random.choice(n, size=math.ceil(n*_frac), replace=False)
                idx = (nobarrier[0][rand], nobarrier[1][rand])
            else: # No ultra-prone switchers, so choose the one that would gain the most energy
                idx = divmod(xp.argmin(barrier), self.nx) # The one with the most energy to gain from switching
                if barrier[idx] >= 0: break # Then energy would increase, so we have ended our minimization procedure.
                visited[idx] += 1
                if xp.any(visited[idx] >= 3): break
            if verbose:
                import matplotlib.pyplot as plt
                plt.scatter(idx[1], idx[0])
                plt.imshow(asnumpy(self.m), origin='lower')
                plt.show()
                from .plottools import show_m
                show_m(self)
            idx = xp.asarray(idx).reshape(2,-1)
            self.m[idx[0], idx[1]] *= -1
            self.switches += idx.shape[1]
            self.attempted_switches += idx.shape[1]
            self.update_energy(index=idx)

    def _minimize_all(self, ignore_barrier=True, _simultaneous=False):
        """ Switches all magnets that can gain energy by switching.
            By default, several subsets of the domain are updated in a random order, such that in total
            every cell got exactly one chance at switching. This randomness may increase the runtime, but
            reduces systematic bias and possible loops which would otherwise cause `self.relax()` to get stuck.
            @param _simultaneous [bool] (False): if True, all magnets are evaluated simultaneously (nonphysical, error-prone).
                If False, several subgrids are evaluated after each other in a random order to prevent systematic bias.
        """
        if _simultaneous:
            warnings.warn("_simultaneous=True in _minimize_all(). I hope you know what you are doing, as this is very nonphysical ;)")
            step_x = step_y = 1
        else:
            step_x, step_y = self.unitcell.x*2, self.unitcell.y*2
        order = np.random.permutation(step_x*step_y)
        for i in order:
            y, x = divmod(i, step_x)
            if not self.occupation[y, x]: continue
            subset_indices = xp.asarray(xp.meshgrid(xp.arange(y, self.ny, step_y), xp.arange(x, self.nx, step_x))).reshape(2, -1)
            if ignore_barrier:
                indices = subset_indices[:,xp.where(self.switch_energy(subset_indices) < 0)[0]]
            else:
                indices = subset_indices[:,self.switch_energy(subset_indices)/2 + self.E_B[subset_indices[0], subset_indices[1]] < 0]
            if indices.size > 0:
                self.m[indices[0], indices[1]] = -self.m[indices[0], indices[1]]
                self.switches += indices.shape[1]
                self.attempted_switches += indices.shape[1]
                self.update_energy(index=indices)

    def relax(self, verbose=False, **kwargs):
        # TODO: Idea for better relaxation: go through all the magnets sequentially (left to right, top to bottom) For each:
        # 1) Switch it.
        # 2) If this makes it the magnet with the highest switching energy, then switch it again and go to the next. (This means that it switching will unlikely cause another magnet to flip)
        # 3) If switching did not make it the magnet with the highest energy, then switch the other one and add all of its neighbors to the 'todo' list of magnets to switch and perform this check onto.
        # I have not thought much about how this could get stuck in an infinite loop, but I guess if the number of magnets to be checked starts to exceed several times mm.n, then just stop because it is exploding.
        # If such cases occur, I will perform further case studies to avoid those.
        """ NOTE: this is basically the parallel version of `self.minimize()`.
            Relaxes `self.m` to a (meta)stable state using multiple `self._minimize_all()` calls.
            NOTE: it can take a while for large simulations to become fully relaxed.
        """
        if verbose:
            import matplotlib.pyplot as plt
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            im = ax.imshow(asnumpy(self.m))

        # TODO: use a better stopping criterion than just "oh no we exceeded n_max steps", and remove verbose code when all is fixed
        n, n_max = 0, int(math.sqrt(self.nx**2 + self.ny**2)) # Maximum steps is to get signal across the whole grid
        previous_states = [self.m.copy(), self._zeros, self._zeros] # Current state, previous, and the one before
        switches = self.switches
        while not xp.allclose(previous_states[2], previous_states[0]) and n < n_max: # Loop exits once we detect a 2-cycle, i.e. the only thing happening is some magnets keep switching back and forth
            n += 1
            self._minimize_all(**kwargs)
            if switches == self.switches: break # Then nothing changed
            if verbose:
                im.set_array(asnumpy(self.m))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            switches = self.switches
            previous_states[2] = previous_states[1].copy()
            previous_states[1] = previous_states[0].copy()
            previous_states[0] = self.m.copy()
        if verbose: print(f"Used {n} (out of {n_max}) steps in relax().")


    def history_save(self, *, E_tot=None, t=None, T_avg=None, m_avg=None):
        """ Records `E_tot`, `t`, `T_avg` and `m_avg` as they were last calculated. This default behavior can be overruled: if
            passed as keyword parameters, their arguments will be saved instead of the `self.<E_tot|t|T_avg|m_avg>` value(s).
        """
        self.history.entry(self)
        if E_tot is not None: self.history.E[-1] = float(E_tot)
        if t is not None: self.history.t[-1] = float(t)
        if T_avg is not None: self.history.T[-1] = float(T_avg)
        if m_avg is not None: self.history.m[-1] = float(m_avg)

    # TODO: improve these autocorrelation and correlation length functions to take into account the non-uniform dx and dy
    def autocorrelation(self, *, normalize=True):
        """ In case of a 2D signal, this basically calculates the autocorrelation of the
            `self.m` matrix, while taking into account `self.orientation` for the in-plane case.
            Note that strictly speaking this assumes dx and dy to be the same for all cells.
        """
        boundary = 'wrap' if self.PBC else 'fill'
        # Now, convolve self.m with its point-mirrored/180°-rotated counterpart
        if self.in_plane:
            mx, my = self.m*self.orientation[:,:,0], self.m*self.orientation[:,:,1]
            corr_xx = signal.correlate2d(mx, mx, boundary=boundary)
            corr_yy = signal.correlate2d(my, my, boundary=boundary)
            corr = corr_xx + corr_yy # Basically trace of autocorrelation matrix
        else:
            corr = signal.correlate2d(self.m, self.m, boundary=boundary)

        if normalize:
            ones = xp.ones_like(self.m)
            corr_norm = signal.correlate2d(ones, ones, boundary=boundary) # Is this necessary? (this is number of occurences of each cell in correlation sum)
            corr *= self.m.size/self.n/corr_norm # Put between 0 and 1
        
        return corr[(self.ny-1):(2*self.ny-1),(self.nx-1):(2*self.nx-1)]**2

    def correlation_length(self, correlation=None):
        if correlation is None: correlation = self.autocorrelation()
        # First calculate the distance between all spins in the simulation.
        rr = (self.xx**2 + self.yy**2)**(1/2) # TODO: This only works correctly if dx, dy is the same for every cell!
        return float(xp.sum(xp.abs(correlation) * rr * rr)/xp.max(xp.abs(correlation))) # Do *rr twice, once for weighted avg, once for 'binning' by distance

    def get_appropriate_avg(self, n=0):
        ''' `n`=0,1,2,3... gives the `n`-th recommended averaging mask.
            `n=None` gives all allowed values, always including 'point'.
        '''
        allowed_averages = self._get_appropriate_avg()
        if isinstance(allowed_averages, str): allowed_averages = [allowed_averages]
        if n is None:
            if 'point' not in [s.lower() for s in allowed_averages]:
                allowed_averages = ['point'] + allowed_averages
            return allowed_averages
        return allowed_averages[min(n, len(allowed_averages) - 1)]

    ######## Now, some useful functions to overwrite when subclassing this class
    @abstractmethod # Not needed for out-of-plane ASI, but essential for in-plane ASI, therefore abstractmethod anyway
    def _get_angles(self):
        """ Returns a 2D array containing the angle (measured counterclockwise from the x-axis)
            of each spin. The angle of unoccupied cells (`self._get_occupation() == 0`) is ignored.
        """
        return xp.zeros_like(self.ixx) # Example

    @abstractmethod # TODO: should this really be an abstractmethod? Uniform and random can be defaults and subclasses can define more of course
    def _set_m(self, pattern: str):
        """ Directly sets `self.m`, depending on `pattern`. Usually, `pattern` is "uniform", "vortex", "AFM" or "random".
            `self.m` is a shape (`self.ny`, `self.nx`) array containing only -1, 0 or 1 indicating the magnetization direction.
            ONLY `self.m` should be set/changed/defined by this function.
        """
        match str(pattern).strip().lower():
            case 'uniform':
                self.m = self._get_m_uniform()
            case 'vortex': # This is not perfect, but probably the best I can do without manually specifying the center of the vortex
                # When using 'angle' property of Magnets.initialize_m:
                #   <angle> near 0 or math.pi: clockwise/anticlockwise vortex, respectively
                #   <angle> near math.pi/2 or -math.pi/2: bowtie configuration (top region: up/down, respectively)
                self.m = xp.ones_like(self.xx)
                if self.in_plane: # In-plane has a clear 'vortex' meaning
                    diff_x = self.xx - (self.x_max - self.x_min)/2
                    diff_y = self.yy - (self.y_max - self.y_min)/2
                    diff_complex = diff_x + 1j*diff_y
                    angle_desired = diff_complex*(-1j+1e-6) # Small offset from 90° to avoid weird boundaries in highly symmetric systems
                    dotprod = angle_desired.real*self.orientation[:,:,0] + angle_desired.imag*self.orientation[:,:,1]
                    dotprod[dotprod == 0] = 1
                    sign = xp.sign(dotprod)
                    occupied = xp.where(self.occupation == 1)
                    self.m[occupied] = sign[occupied]
                else: # for OOP, 'vortex' means 'up' on one side, 'down' on other side
                    half_side = xp.where(self.xx >= (self.x_max - self.x_min)/2)
                    self.m[half_side] *= -1
            case str(unknown_pattern):
                self.m = self.rng.integers(0, 2, size=self.xx.shape)*2 - 1
                if unknown_pattern != 'random': warnings.warn(f"Pattern '{unknown_pattern}'' not recognized, defaulting to 'random'.", stacklevel=2)

    @abstractmethod
    def _get_occupation(self):
        """ Returns a 2D array which contains 1 at the cells which are occupied by a magnet, and 0 elsewhere. """
        return xp.ones_like(self.ixx) # Example

    @abstractmethod
    def _get_groundstate(self):
        """ Returns `pattern` in `self.initialize_m(pattern)` which corresponds to a global ground state of the system.
            Use 'random' if no ground state is implemented in `self._set_m()`.
        """
        return 'random'

    @abstractmethod
    def _get_nearest_neighbors(self): # TODO: make this automatically calculated and unitcell-dependent
        """ Returns a small mask with the magnet at the center, and 1 at the positions of its nearest neighbors (elsewhere 0). """
        return xp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # Example

    @abstractmethod # TODO: this is a pretty archaic method at this point
    def _get_appropriate_avg(self):
        """ Returns the most appropriate averaging mask for a given type of ASI. """
        return 'cross' # Example

    @abstractmethod # TODO: this is a pretty archaic method at this point as well. Sould this really be abstract?
    def _get_AFMmask(self):
        """ Returns the (normalized) mask used to determine how anti-ferromagnetic the magnetization profile is. """
        return xp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float') # Example


@dataclass
class History:
    """ Stores the history of the energy, temperature, time, and average magnetization. """
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
    """ Stores x and y components, so we don't need to index [0] or [1] in a tuple, which would be unclear. """
    x: float
    y: float
