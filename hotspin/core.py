import math
import warnings

import cupy as cp
import numpy as np

from abc import ABC, abstractmethod
from cupyx.scipy import signal
from dataclasses import dataclass, field
from scipy.spatial.distance import pdist
from typing import List


kB = 1.380649e-23


@dataclass
class SimParams:
    SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF: int = 20 # If there are less than <this> switches in a single iteration, the energies are just summed, otherwise a convolution is used.
    REDUCED_KERNEL_SIZE: int = 20 # If the dipolar kernel is truncated, it becomes a shape (2*<this>-1, 2*<this>-1) array.
    UPDATE_SCHEME: str = 'Glauber' # Can be any of 'Néel', 'Glauber', 'Wolff'


class Magnets: # TODO: make this a behind-the-scenes class, and make ASI the abstract base class that is exposed to the world outside this file
    def __init__(self, nx, ny, dx, dy, *, T=1, E_b=5e-20, Msat=800e3, V=2e-22, in_plane=True, pattern='random', energies=None, PBC=False, angle=None, params: SimParams = None):
        '''
            !!! THIS CLASS SHOULD NOT BE INSTANTIATED DIRECTLY, USE AN ASI WRAPPER INSTEAD !!!
            The position of magnets is specified using <nx>, <ny>, <dx> and <dy>. Only rectilinear grids are allowed currently. 
            The initial configuration of a Magnets geometry consists of 3 parts:
             1) in_plane:  Magnets can be in-plane or out-of-plane: True or False, respectively.
             2) ASI type:  Defined through subclasses (pinwheel, kagome, Ising...).
             3) pattern: The initial magnetization direction (e.g. up/down) can be 'uniform', 'AFM' or 'random'.
            One can also specify which energy components should be considered: any of 'dipolar', 'Zeeman' and 'exchange'.
                If you want to adjust the parameters of these energies, than call energy_<type>_init(<parameters>) manually.
                By default, only the dipolar energy is active.
        '''
        if type(self) is Magnets: # Need to check for Magnets without inheritance, hence we use type() instead of isinstance()
            raise Exception("Magnets() class can not be instantiated directly, and should instead be subclassed. Consider using a class from the hotspin.ASI module, or writing your own custom ASI class instead.")

        self.params = SimParams() if params is None else params # This can just be edited and accessed normally since it is just a dataclass
        self.t = 0. # [s]
        self.Msat = Msat # [A/m]
        self.V = V # [m³]
        self.in_plane = in_plane
        energies: tuple[Energy] = (DipolarEnergy(),) if energies is None else tuple(energies) # [J]
        
        # initialize the coordinates based on nx, (ny) and L
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.xx, self.yy = cp.meshgrid(cp.linspace(0, dx*(nx-1), nx), cp.linspace(0, dy*(ny-1), ny)) # [m]
        self.index = range(self.xx.size)
        self.ixx, self.iyy = cp.meshgrid(cp.arange(0, self.nx), cp.arange(0, self.ny))
        self.x_min, self.y_min, self.x_max, self.y_max = float(self.xx[0,0]), float(self.yy[0,0]), float(self.xx[-1,-1]), float(self.yy[-1,-1])

        # initialize temperature and energy barrier arrays (!!! needs self.xx etc. to exist, since this is an array itself)
        self.T = T # [K]
        
        if isinstance(E_b, np.ndarray): # This detects both CuPy and NumPy arrays
            assert E_b.shape == self.xx.shape, f"Specified energy barriers (shape {E_b.shape}) does not match shape ({nx}, {ny}) of simulation domain."
            self.E_b = cp.asarray(E_b) # [J]
        else:
            self.E_b = cp.ones_like(self.xx)*E_b # [J]
            # TODO: this is not used in Glauber update scheme, but is that correct?

        # Unit cells and PBC
        self.unitcell = Vec2D(*self._get_unitcell())
        self.PBC = PBC
        if self.PBC:
            if nx % self.unitcell.x != 0 or ny % self.unitcell.y != 0:
                warnings.warn(f"Be careful with PBC, as there are not an integer number of unit cells in the simulation! Hence, the boundaries do not nicely fit together. Adjust nx or ny to alleviate this problem (unit cell has size {self.unitcell.x}x{self.unitcell.y}).", stacklevel=2)

        self.history = History()
        self.switches = 0

        # Main initialization steps that require calling other methods of this class
        self.occupation = self._get_occupation().astype(bool).astype(int) # Make sure that it is either 0 or 1
        self.n = int(cp.sum(self.occupation)) # Number of magnets in the simulation
        if self.in_plane: self._initialize_ip(angle=angle)

        self.initialize_m(pattern, update_energy=False)

        # Some energies might require self.orientation etc., so only initialize energy at the end
        self.energies = list[Energy]()
        for energy in energies: self.add_energy(energy)

    def _set_m(self, pattern, angle=0):
        if pattern == 'uniform': # PYTHONUPDATE_3.10: use structural pattern matching
            self.m = cp.ones_like(self.xx)
        elif pattern == 'AFM':
            self.m = ((self.ixx - self.iyy) % 2)*2 - 1
        else:
            self.m = cp.random.randint(0, 2, size=cp.shape(self.xx))*2 - 1
            if pattern != 'random': warnings.warn('Pattern not recognized, defaulting to "random".', stacklevel=2)

    def _get_unitcell(self):
        return Vec2D(1, 1)
    
    def _get_occupation(self):
        return cp.ones_like(self.ixx)
    
    def _get_groundstate(self):
        return 'random'
    
    def _get_closest_dist(self):
        ''' Returns the closest distance between two magnets in the simulation. '''
        slice = cp.where(self.occupation[:self.unitcell.y*2,:self.unitcell.x*2]) # We only need at most two unit cells
        pos_x, pos_y = self.xx[slice], self.yy[slice]
        return pdist(cp.asarray([pos_x, pos_y]).get().T).min()

    def initialize_m(self, pattern='random', *, angle=0, update_energy=True):
        ''' Initializes the self.m (array of -1, 0 or 1) and occupation.
            @param pattern [str]: can be any of "random", "uniform", "AFM".
        '''
        self._set_m(pattern, angle=angle)
        self.m = self.m.astype(float)
        self.m = cp.multiply(self.m, self.occupation)
        if update_energy: self.update_energy() # Have to recalculate all the energies since m changed completely
    
    def _set_orientation(self, angle: float = 0): # TODO: could perhaps make this 3D to avoid possible errors for OOP ASI where self.orientation is now undefined
        self.orientation = cp.ones((*self.xx.shape, 2))/math.sqrt(2)

    def _initialize_ip(self, angle: float = None):
        ''' Initialize the angles of all the magnets (only applicable in the in-plane case).
            This function should only be called by the Magnets() class itself, not by the user.
            @param angle [float] (0): optional parameter passed to self._set_orientation(), normally
                used to define an angle (in radians) by which each spin in the system is rotated.
        '''
        assert self.in_plane, "Can not _initialize_ip() if magnets are not in-plane (in_plane=False)."
        self._set_orientation(angle=angle) if angle else self._set_orientation()

    def add_energy(self, energy: 'Energy'):
        ''' Adds an Energy object to self.energies. This object is stored under its reduced name,
            e.g. ZeemanEnergy is stored under 'zeeman'.
            @param energy [Energy]: the energy to be added.
        '''
        energy.initialize(self)
        for i, e in enumerate(self.energies):
            if type(e) is type(energy):
                warnings.warn(f'An instance of {type(energy).__name__} was already included in the simulation, and has now been overwritten.', stacklevel=2)
                self.energies[i] = energy
                return
        self.energies.append(energy)

    def remove_energy(self, name: str):
        ''' Removes the specified energy from self.energies.
            @param name [str]: the name of the energy to be removed. Case-insensitive and may or may not include
                the 'energy' part of the class name. Valid options include e.g. 'dipolar', 'DipolarEnergy'...
        '''
        name = name.lower().replace('energy', '')
        for i, e in enumerate(self.energies):
            if name == e.shortname():
                self.energies.pop(i)
                return
        warnings.warn(f"There is no '{name}' energy associated with this Magnets object. Valid energies are: {[e.shortname() for e in self.energies]}", stacklevel=2)

    def get_energy(self, name: str):
        ''' Returns the specified energy from self.energies.
            @param name [str]: the name of the energy to be returned. Case-insensitive and may or may not include
                the 'energy' part of the class name. Valid options include e.g. 'dipolar', 'DipolarEnergy'...
            @returns [Energy]: the requested energy object.
        '''
        name = name.lower().replace('energy', '')
        for e in self.energies:
            if name == e.shortname(): return e
        warnings.warn(f"There is no '{name}' energy associated with this Magnets object. Valid energies are: {[e.shortname() for e in self.energies]}", stacklevel=2)
        return None

    def update_energy(self, index: np.ndarray=None):
        ''' Updates all the energies which are currently present in the simulation.
            @param index [np.array] (None): if specified, only the magnets at these indices are considered in the calculation.
                We need a NumPy or CuPy array (to easily determine its size: if =2, then only a single switch is considered.)
        '''
        if index is not None: index = Energy.clean_indices(index) # TODO: now we are cleaning twice, so remove this in Energy classes maybe?
        for energy in self.energies:
            # PYTHONUPDATE_3.10: use structural pattern matching
            if index is None: # No index specified, so update fully
                energy.update()
            elif index.size == 2: # Index contains 2 ints, so it represents the coordinates of a single magnet
                energy.update_single(Energy.clean_index(index))
            else: # Index is specified and does not contain 2 ints, so interpret it as coordinates of multiple magnets
                energy.update_multiple(index)
    
    def switch_energy(self, indices2D):
        ''' @return [cp.array]: the change in energy for each magnet in indices2D, in the same order, if they were to switch. '''
        return cp.sum(cp.asarray([energy.energy_switch(indices2D) for energy in self.energies]), axis=0)

    @property
    def MCsteps(self): # Number of Monte Carlo steps
        return self.switches/self.n

    @property
    def E(self) -> cp.ndarray:
        return cp.sum(cp.asarray([energy.E for energy in self.energies]), axis=0)

    @property
    def T(self) -> cp.ndarray:
        return self._T
    
    @T.setter
    def T(self, value):
        if isinstance(value, np.ndarray): # This detects both CuPy and NumPy arrays
            assert value.shape == self.xx.shape, f"Specified temperature profile (shape {value.shape}) does not match shape ({self.nx}, {self.ny}) of simulation domain."
            self._T = cp.asarray(value)
        else:
            self._T = cp.ones_like(self.xx)*value

    @property
    def kBT(self) -> cp.ndarray:
        return 1.380649e-23*self._T

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
    
    def set_T(self, f, /, *, center=False, noSIunits=False):
        ''' Sets the temperature field according to a spatial function. To simply set the temperature array, use self.T = assignment instead.
            @param f [function(x,y)->T]: x and y in meters yield T in Kelvin (should accept CuPy arrays as x and y).
            @param center [bool] (False): if True, the origin is put in the middle of the simulation,
                otherwise it is in the usual lower left corner.
            @param noSIunits [bool] (False): if True, x and y are assumed to be a number of cells instead of a distance in meters (but still float).
        '''
        if center:
            xx = self.xx + self.dx*self.nx/2
            yy = self.yy + self.dy*self.ny/2
        else:
            xx, yy = self.xx, self.yy
        if noSIunits: xx, yy = xx/self.dx, xx/self.dy
        self._T = f(xx, yy)


    def select(self, r=None):
        ''' @param r [float] (self.calc_r(0.01)): minimal distance between magnets, in meters
            @return [cp.array]: a 2xN array, where the 2 rows represent y- and x-coordinates (!), respectively.
                (this is because the indexing of 2D arrays is e.g. self.m[y,x])
        '''
        if r is None: r = self.calc_r(0.01)

        return self._select_grid(r=r)
        # return self._select_single()
    
    def _select_single(self):
        ''' Selects just a single magnet from the simulation domain. '''
        nonzero_y, nonzero_x = cp.nonzero(self.occupation)
        nonzero_idx = cp.random.choice(self.n, 1)
        return cp.asarray([nonzero_y[nonzero_idx], nonzero_x[nonzero_idx]]).reshape(2, -1)

    def _select_grid(self, r):
        ''' Uses a supergrid with supercells of size <r> to select multiple sufficiently-spaced magnets at once.
            ! <r> is a distance in meters, not a number of cells !
        '''
        Rx, Ry = math.ceil(r/self.dx) - 1, math.ceil(r/self.dy) - 1 # - 1 because effective minimal distance in grid-method is supercell_size + 1
        Rx, Ry = self.unitcell.x * math.ceil(Rx/self.unitcell.x), self.unitcell.y * math.ceil(Ry/self.unitcell.y) # To have integer number of unit cells in a supercell (necessary for occupation_supercell)
        Rx, Ry = max(min(Rx, self.nx), 1), max(min(Ry, self.ny), 1) # No need to make supercell larger than simulation itself
        supergrid_nx = (self.nx + Rx - 1)//(2*Rx) + 1
        supergrid_ny = (self.ny + Ry - 1)//(2*Ry) + 1
        move_grid = -np.random.randint([-Ry, -Rx], [Ry, Rx])
        if Rx >= self.nx: move_grid[1] = 0 # If the grid is perfectly fit along an axis, then
        if Ry >= self.ny: move_grid[0] = 0 # just don't move in that coordinate direction
        offsets_x = move_grid[1] + self.ixx[:supergrid_ny, :supergrid_nx].ravel()*2*Rx
        offsets_y = move_grid[0] + self.iyy[:supergrid_ny, :supergrid_nx].ravel()*2*Ry
        disp_x, disp_y = move_grid[1] % self.unitcell.x, move_grid[0] % self.unitcell.y
        occupation_supercell = self.occupation[disp_y:disp_y + Ry, disp_x:disp_x + Rx]
        occupation_nonzero = occupation_supercell.nonzero()
        random_nonzero_indices = cp.random.choice(occupation_nonzero[0].size, supergrid_ny*supergrid_nx)
        idx_x = offsets_x + occupation_nonzero[1][random_nonzero_indices]
        idx_y = offsets_y + occupation_nonzero[0][random_nonzero_indices]
        ok = cp.logical_and(cp.logical_and(idx_x >= 0, idx_y >= 0), cp.logical_and(idx_x < self.nx, idx_y < self.ny))
        if cp.sum(ok) == 0: return self._select_grid(r) # If no samples were taken, try again
        return cp.asarray([idx_y[ok], idx_x[ok]])
    
    def _select_cluster(self):
        ''' Selects a cluster based on the Wolff algorithm for the two-dimensional square Ising model. '''
        exchange: ExchangeEnergy = self.get_energy('exchange')
        assert exchange, "Can not select Wolff cluster if there is no exchange energy in the system."
        neighbors = exchange.local_interaction
        seed = tuple(self._select_single().flat)
        cluster = cp.zeros_like(self.m)
        cluster[seed] = 1
        checked = cp.copy(cluster)
        checked[cp.where(self.m != self.m[seed])] = 1
        while cp.any(current_frontier := (1-checked)*signal.convolve2d(cluster, neighbors, mode='same', boundary='wrap' if self.PBC else 'fill')):
            checked[current_frontier != 0] = 1
            bond_prob = 1 - cp.exp(-2*exchange.J/self.kBT*current_frontier) # Swendsen-Wang bond probability
            current_frontier[cp.random.random(cluster.shape) > bond_prob] = 0 # Rejected probabilistically
            cluster[current_frontier != 0] = 1
        return cp.where(cluster == 1)

    def update(self, *args, **kwargs):
        if cp.any(self.T == 0):
            warnings.warn('Temperature is zero somewhere, so no switch will be simulated to prevent DIV/0 errors.', stacklevel=2)
            return # We just warned that no switch will be simulated, so let's keep our word
        if self.params.UPDATE_SCHEME == 'Néel': # PYTHONUPDATE_3.10: use structural pattern matching
            idx = self._update_Néel(*args, **kwargs)
        elif self.params.UPDATE_SCHEME == 'Glauber':
            idx = self._update_Glauber(*args, **kwargs)
        elif self.params.UPDATE_SCHEME == 'Wolff':
            idx = self._update_Wolff(*args, **kwargs)
        else:
            raise ValueError(f"The Magnets.params object has an invalid value for UPDATE_SCHEME='{self.params.UPDATE_SCHEME}'.")
        return Energy.clean_indices(idx)

    def _update_Glauber(self, Q=0.05, idx=None):
        # 1) Choose a bunch of magnets at random
        if idx is None: idx = self.select(r=self.calc_r(Q))
        # 2) Compute the change in energy if they were to flip, and the corresponding Boltzmann factor.
        # TODO: can adapt this to work at T=0 by first checking if energy would drop
        exponential = cp.clip(cp.exp(-self.switch_energy(idx)/self.kBT[idx[0], idx[1]]), 1e-10, 1e10) # clip to avoid inf
        # 3) Flip the spins with a certain exponential probability. There are two commonly used and similar approaches:
        # idx = idx[:,cp.where(cp.random.random(exponential.shape) < (exponential/(1+exponential)))[0]] # https://en.wikipedia.org/wiki/Glauber_dynamics
        idx = idx[:,cp.where(cp.random.random(exponential.shape) < exponential)[0]] # http://bit-player.org/2019/glaubers-dynamics
        if idx.shape[1] > 0:
            self.m[idx[0], idx[1]] *= -1
            self.switches += idx.shape[1]
            self.update_energy(index=idx)
        return idx
    
    def _update_Néel(self):
        ''' Performs a single magnetization switch. '''
        barrier = (self.E_b - self.E)/self.occupation # Divide by occupation to make non-occupied grid cells have infinite barrier
        minBarrier = cp.min(barrier)
        barrier -= minBarrier # Energy is relative, so set min(E) to zero (this prevents issues at low T)
        with np.errstate(over='ignore'): # Ignore overflow warnings in the exponential: such high barriers wouldn't switch anyway
            # There is no real speed difference between XORWOW or MRG32k3a or Philox4x3210 random generators, so we just don't bother.
            taus = cp.random.uniform(size=barrier.shape)*cp.exp(barrier/self.kBT)
            indexmin2D = divmod(cp.argmin(taus), self.m.shape[1]) # cp.unravel_index(indexmin, self.m.shape) # The min(tau) index in 2D form for easy indexing
            self.m[indexmin2D] = -self.m[indexmin2D]
            self.switches += 1
            self.t += taus[indexmin2D]*(-cp.log(1-taus[indexmin2D]))*cp.exp(minBarrier/self.kBT[indexmin2D]) # This can become cp.inf quite quickly if T is small
        self.update_energy(index=indexmin2D)
        return indexmin2D

    def _update_Wolff(self):
        ''' Only works for Exchange-coupled 2D Ising system. '''
        cluster = self._select_cluster()
        self.m[cluster[0], cluster[1]] *= -1
        self.switches += cluster.shape[1]
        self.update_energy(index=cluster)
        return cluster


    def calc_r(self, Q): # r: a distance in meters
        ''' Calculates the minimal value of r (IN METERS). Considering two nearby sampled magnets, the switching probability
            of the first magnet will depend on the state of the second. For magnets further than <calc_r(Q)> apart, the switching
            probability of the first will not change by more than <Q> if the second magnet switches.
            @param Q [float]: (0<Q<1) the maximum allowed change in switching probability of a sample if any other sample switches.
            @return [float]: the minimal distance (in meters) between two samples, if their mutual influence on their switching
                probabilities is to be less than <Q>.
        '''
        return (8e-7*self.Msat**2*self.V**2/(Q*cp.min(self.kBT)))**(1/3)


    def minimize(self, N=1):
        ''' Switches the magnet which has the highest switching probability. Repeat this <N> times. '''
        all_indices = cp.asarray([self.iyy.reshape(-1), self.ixx.reshape(-1)])
        for _ in range(N):
            indexmin2D = divmod(cp.argmin(self.switch_energy(all_indices)), self.nx)
            self.m[indexmin2D] = -self.m[indexmin2D]
            self.update_energy(index=indexmin2D)


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

    ######## Now, some useful functions when subclassing this class
    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def _get_appropriate_avg(self):
        ''' Returns the most appropriate averaging mask for a given type of ASI '''
        return 'cross'

    def _get_AFMmask(self):
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')


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
            @param indices2D [list(2xN)]: A list containing two elements: an array containing the x-indices of each switched
                magnet, and a similar array for y (so indices2D is basically a 2xN array, with N the number of switches).
            @return [list(N)]: A list containing the local changes in energy for each magnet of <indices2D>, in the same order.
        '''

    @abstractmethod
    def update_single(self, index2D):
        ''' Updates self.E by only taking into account that a single magnet (at index2D) switched.
            @param index2D [tuple(2)]: A tuple containing two elements: the x- and y-index of the switched magnet.
        '''
    
    @abstractmethod
    def update_multiple(self, indices2D):
        ''' Updates self.E by only taking into account that some magnets (at indices2D) switched.
            This seems like it is just multiple times self.update_single(), but sometimes an optimization is possible,
            hence this required alternative function for updating multiple magnets at once.
            @param indices2D [list(2xN)]: A list containing two elements: an array containing the x-indices of each switched
                magnet, and a similar array for y (so indices2D is basically a 2xN array, with N the number of switches).
        '''

    @property
    @abstractmethod
    def E_tot(self):
        ''' Returns the total energy for this energy contribution. This function is necessary since this is not equal
            for all energies: e.g. sum(E) in the DipolarEnergy would count each interaction twice, while sum(E) is
            correct for ZeemanEnergy.
        '''

    def shortname(self):
        return type(self).__name__.lower().replace('energy', '')

    @classmethod
    def clean_indices(cls, indices2D):
        ''' Reshapes <indices2D> into a 2xN two-dimensional array.
            @param <indices2D> [iterable]: an iterable which can have any shape, as long as it contains
                at most 2 dimensions of a size greater than 1, of which exactly one has size 2. It is 
                this size-2 dimension which will become the primary dimension of the returned array.
            @return [np.array(2xN)]: A 2xN array, where the 1st sub-array indicates x-indices, the 2nd y-indices.
        '''
        indices2D = cp.atleast_2d(cp.asarray(indices2D).squeeze())
        assert len(indices2D.shape) <= 2, "An array with more than 2 non-empty dimensions can not be used to represent a list of indices."
        assert cp.any(cp.asarray(indices2D.shape) == 2), "The list of indices has an incorrect shape. At least one dimension should have length 2."
         # PYTHONUPDATE_3.10: use structural pattern matching
        if indices2D.shape[0] == 2: # The only ambiguous case is for a 2x2 input array, but in that case we assume the array is already correct.
            return indices2D
        elif indices2D.shape[1] == 2:
            return indices2D.T
        else:
            raise ValueError("Structure of attribute <indices2D> could not be recognized.")
    
    @classmethod
    def clean_index(cls, index2D):
        ''' Reshapes <index2D> into a length 2 one-dimensional array.
            @param <index2D> [iterable]: an iterable which can have any shape, as long as it contains exactly 2 elements.
            @return [tuple(2)]: A length 2 tuple, whose elements correspond to the x- and y-index, respectively.
        '''
        index2D = cp.asarray(index2D)
        assert index2D.size == 2, "The <index2D> argument should contain exactly two values: the x- and y-index."
        return tuple(index2D.reshape(2))
    
    @classmethod
    def J_to_eV(cls, E, /):
        return E/1.602176634e-19
    @classmethod
    def eV_to_J(cls, E, /):
        return E*1.602176634e-19


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
            if self.angle != 0: warnings.warn(f'You tried to set the angle of an out-of-plane field in ZeemanEnergy.set_field(), but this is not supported.', stacklevel=2)
        self.update()

    def update(self):
        if self.mm.in_plane:
            self.E = -self.mm.Msat*self.mm.V*cp.multiply(self.mm.m, self.B_ext[0]*self.mm.orientation[:,:,0] + self.B_ext[1]*self.mm.orientation[:,:,1])
        else:
            self.E = -self.mm.Msat*self.mm.V*self.mm.m*self.B_ext

    def energy_switch(self, indices2D):
        indices2D = Energy.clean_indices(indices2D)
        return -2*self.E[indices2D[0], indices2D[1]]

    def update_single(self, index2D):
        index2D = Energy.clean_index(index2D)
        self.E[index2D] *= -1
    
    def update_multiple(self, indices2D):
        indices2D = Energy.clean_indices(indices2D)
        self.E[indices2D[0], indices2D[1]] *= -1
    
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

        def mirror4(arr, /, *, negativex=False, negativey=False):
            ny, nx = arr.shape
            arr4 = cp.zeros((2*ny-1, 2*nx-1))
            xp = -1 if negativex else 1
            yp = -1 if negativey else 1
            arr4[ny-1:, nx-1:] = arr
            arr4[ny-1:, nx-1::-1] = xp*arr
            arr4[ny-1::-1, nx-1:] = yp*arr
            arr4[ny-1::-1, nx-1::-1] = xp*yp*arr
            return arr4
        
        # Let us first make the four-mirrored distance matrix rinv3
        # WARN: this four-mirrored technique only works if (dx, dy) is the same for every cell everywhere!
        # This could be generalized by calculating a separate rrx and rry for each magnet in a unit cell similar to toolargematrix_o{x,y}
        rrx = self.mm.xx - self.mm.xx[0,0]
        rry = self.mm.yy - self.mm.yy[0,0]
        rr_sq = rrx**2 + rry**2
        rr_sq[0,0] = cp.inf
        rr_inv = rr_sq**(-1/2) # Due to the previous line, this is now never infinite
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

                kernel *= 1e-7*(self.mm.Msat*self.mm.V)**2 # [J], 1e-7 is mu_0/4Pi
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

                    total_energy = total_energy + partial_m*signal.convolve2d(kernel, self.mm.m, mode='valid')
        self.E = self.prefactor*total_energy
    
    def energy_switch(self, indices2D):
        indices2D = Energy.clean_indices(indices2D)
        return -2*self.E[indices2D[0], indices2D[1]]
    
    def update_single(self, index2D):
        index2D = Energy.clean_index(index2D)
        # First we get the x and y coordinates of magnet <i> in its unit cell
        y, x = index2D
        x_unitcell = int(x) % self.unitcell.x
        y_unitcell = int(y) % self.unitcell.y
        # The kernel to use is then
        kernel = self.kernel_unitcell[y_unitcell][x_unitcell]
        if kernel is not None:
            # Multiply with the magnetization
            usefulkernel = kernel[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
            interaction = self.prefactor*self.mm.m[index2D]*cp.multiply(self.mm.m, usefulkernel)
        else:
            interaction = cp.zeros_like(self.mm.m)

        self.E += 2*interaction
        self.E[index2D] *= -1 # This magnet switched, so all its interactions are inverted
    
    def update_multiple(self, indices2D):
        indices2D = Energy.clean_indices(indices2D)
        self.E[indices2D[0], indices2D[1]] *= -1
        indices2D_unitcell = cp.empty_like(indices2D)
        indices2D_unitcell[0,:] = indices2D[0,:] % self.unitcell.y
        indices2D_unitcell[1,:] = indices2D[1,:] % self.unitcell.x
        indices2D_unitcell_raveled = indices2D_unitcell[1] + indices2D_unitcell[0]*self.unitcell.x
        binned_unitcell_raveled = cp.bincount(indices2D_unitcell_raveled)
        for i in binned_unitcell_raveled.nonzero()[0]: # Iterate over the unitcell indices present in indices2D
            y_unitcell, x_unitcell = divmod(int(i), self.unitcell.x)
            kernel = self.kernel_unitcell[y_unitcell][x_unitcell]
            if kernel is None: continue # This should never happen, but check anyway in case indices2D includes empty cells
            indices_here = indices2D[:,indices2D_unitcell_raveled == i]
            if indices_here.shape[1] > self.mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF: # THIS NUMBER IS EMPIRICALLY DETERMINED
                ### EITHER WE DO THIS (CONVOLUTION) (starts to be better at approx. 40 simultaneous switches for 41x41 kernel, taking into account the need for complete recalculation every <something> steps, so especially for large T this is good)
                switched_field = cp.zeros_like(self.mm.m)
                switched_field[indices_here[0], indices_here[1]] = self.mm.m[indices_here[0], indices_here[1]]
                n = self.mm.params.REDUCED_KERNEL_SIZE
                usefulkernel = signal.convolve2d(switched_field, kernel[self.mm.ny-1-n:self.mm.ny+n, self.mm.nx-1-n:self.mm.nx+n], mode='same', boundary='wrap' if self.mm.PBC else 'fill')
                # usefulkernel = signal.convolve2d(kernel, switched_field, mode='valid') # this one is extremely EXTREMELY slow
                interaction = self.prefactor*cp.multiply(self.mm.m, usefulkernel)
                self.E += 2*interaction
            else:
                ### OR WE DO THIS (BASICALLY self.update_single BUT SLIGHTLY PARALLEL AND SLIGHTLY NONPARALLEL) 
                interaction = cp.zeros_like(self.mm.m)
                for j in range(indices_here.shape[1]):
                    y, x = indices_here[0,j], indices_here[1,j]
                    interaction += self.mm.m[y,x]*kernel[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
                interaction = self.prefactor*cp.multiply(self.mm.m, interaction)
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
        indices2D = Energy.clean_indices(indices2D)
        return -2*self.E[indices2D[0], indices2D[1]]
    
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

@dataclass
class Vec2D:
    ''' Stores x and y components, so we don't need to index [0] or [1] in a tuple, which would be unclear. '''
    x: float
    y: float
