import math
import warnings

import cupy as cp
import numpy as np

from cupyx.scipy import signal
from dataclasses import dataclass, field

"""
TODO (summary):
(!: priority, -: should do at some point in time, .: perhaps implement perhaps not)
! use Glauber monte carlo markov chain on Ising model to determine which magnet will switch
  (this allows for switching multiple magnets at once, if we use some convolutions in a clever and fast way)
! develop the hotspin.io module
- update the animate_temp_rise function with the modern 'API' or however to call this
- sort out the AFM-ness and its normalization etc.
- organize plotting functions better
. can implement linear transformations if I want to
. can implement random defects if I want to
- make unit tests
"""


class Magnets: # TODO: make this a behind-the-scenes class, and make ASI the abstract base class that is exposed to the world outside this file
    def __init__(self, nx, ny, dx, dy, T=1, E_b=1, Msat=1, in_plane=True, pattern='random', energies=('dipolar',), PBC=False):
        '''
            !!! THIS CLASS SHOULD NOT BE INSTANTIATED DIRECTLY, USE AN ASI WRAPPER INSTEAD !!!
            The position of magnets is specified using <nx> and <a>. 
            The initial configuration of a Magnets geometry consists of 3 parts:
             1) in_plane:  Magnets can be in-plane or out-of-plane: True or False, respectively.
             2) ASI type:  Defined through subclasses (pinwheel, kagome, Ising...).
             3) pattern: The initial magnetization direction (e.g. up/down) can be 'uniform', 'AFM' or 'random'.
            One can also specify which energy components should be considered: any of 'dipolar', 'Zeeman' and 'exchange'.
                If you want to adjust the parameters of these energies, than call energy_<type>_init(<parameters>) manually.
            # TODO: linear transformations (e.g. skewing or squeezing) should be relatively easy to implement by acting on xx, yy
            #       see https://matplotlib.org/stable/gallery/images_contours_and_fields/affine_image.html for the imshows then
        '''
        if type(self) is Magnets:
            raise Exception("Magnets() class can not be instantiated directly, and should instead be subclassed. Consider using a class from the hotspin.ASI module, or writing your own custom ASI class instead.")

        self.t = 0. # TODO: decide if we are interested in the time, or not really
        self.Msat = Msat
        self.in_plane = in_plane
        self.energies = list(energies)
        
        # initialize the coordinates based on nx, (ny) and L
        self.nx, self.ny = nx, ny
        self.xx, self.yy = cp.meshgrid(cp.linspace(0, dx*(nx-1), nx), cp.linspace(0, dy*(ny-1), ny))
        self.index = range(self.xx.size)
        self.ixx, self.iyy = cp.meshgrid(cp.arange(0, self.xx.shape[1]), cp.arange(0, self.yy.shape[0]))
        self.x_min, self.y_min, self.x_max, self.y_max = float(self.xx[0,0]), float(self.yy[0,0]), float(self.xx[-1,-1]), float(self.yy[-1,-1])

        # initialize temperature and energy barrier arrays
        if isinstance(T, np.ndarray): # This detects both CuPy and NumPy arrays
            assert T.shape == self.xx.shape, f"Specified temperature profile (shape {T.shape}) does not match shape ({nx}, {ny}) of simulation domain."
            self.T = cp.asarray(T)
        else:
            self.T = cp.ones_like(self.xx)*T
        
        if isinstance(E_b, np.ndarray): # This detects both CuPy and NumPy arrays
            assert E_b.shape == self.xx.shape, f"Specified energy barriers (shape {E_b.shape}) does not match shape ({nx}, {ny}) of simulation domain."
            self.E_b = cp.asarray(E_b)
        else:
            self.E_b = cp.ones_like(self.xx)*E_b

        # Unit cells and PBC
        self.unitcell = Vec2D(*self._get_unitcell())
        self.PBC = PBC
        if self.PBC:
            if nx % self.unitcell.x != 0 or ny % self.unitcell.y != 0:
                warnings.warn(f"Be careful with PBC, as there are not an integer number of unit cells in the simulation! Hence, the boundaries do not nicely fit together. Adjust nx or ny to alleviate this problem (unit cell has size {self.unitcell.x}x{self.unitcell.y}).", stacklevel=2)

        self.history = History()

        # Main initialization steps that require calling other methods of this class
        self.occupation = self._get_occupation().astype(bool).astype(int) # Make sure that it is either 0 or 1
        self.n = int(cp.sum(self.occupation)) # Number of magnets in the simulation
        if self.in_plane: self._initialize_ip()
        self.energy_init() # This needs self.orientation
        self.initialize_m(pattern) # This needs energy kernels to be ready

    def _get_unitcell(self):
        return Vec2D(1, 1)
    
    def _get_occupation(self):
        return cp.ones_like(self.ixx)
    
    def _get_groundstate(self):
        return 'random'

    def initialize_m(self, pattern='random'):
        ''' Initializes the self.m (array of -1, 0 or 1) and occupation.
            @param pattern [str]: can be any of "random", "uniform", "AFM".
        '''
        self._set_m(pattern)
        self.m = cp.multiply(self.m, self.occupation)
        self.energy() # Have to recalculate all the energies since m changed completely
    
    def _set_orientation(self):
        self.orientation = cp.ones((*self.xx.shape, 2))/math.sqrt(2)

    def _initialize_ip(self):
        ''' Initialize the angles of all the magnets.
            This function should only be called by the Magnets() class itself, not by the user.
        '''
        # This sets the angle of all the magnets (this is of course only applicable in the in-plane case)
        assert self.in_plane, "Can not _initialize_ip() if magnets are not in-plane (in_plane=False)."
        self._set_orientation()


    @property
    def m_tot(self):
        if self.in_plane:
            self.m_tot_x = cp.mean(cp.multiply(self.m, self.orientation[:,:,0]))
            self.m_tot_y = cp.mean(cp.multiply(self.m, self.orientation[:,:,1]))
            return (self.m_tot_x**2 + self.m_tot_y**2)**(1/2)
        else:
            return cp.mean(self.m)



    def energy(self, single=False, index2D=None):
        assert not (single and index2D is None), "Provide the latest switch index to energy(single=True)"
        E = cp.zeros_like(self.xx)
        if 'exchange' in self.energies:
            self.energy_exchange_update()
            E = E + self.E_exchange
        if 'dipolar' in self.energies:
            if single:
                self.energy_dipolar_update(index2D)
            else:
                self.energy_dipolar_full()
            E = E + self.E_dipolar
        if 'Zeeman' in self.energies:
            self.energy_Zeeman_update()
            E = E + self.E_Zeeman
        self.E_int = E
        self.E_tot = cp.sum(E, axis=None)
        return self.E_tot
    
    def energy_init(self):
        if 'dipolar' in self.energies:
            self.energy_dipolar_init()
        if 'Zeeman' in self.energies:
            self.energy_Zeeman_init()
        if 'exchange' in self.energies:
            self.energy_exchange_init(1)

    def energy_Zeeman_init(self):
        if 'Zeeman' not in self.energies: self.energies.append('Zeeman')
        self.E_Zeeman = cp.empty_like(self.xx)
        if self.in_plane:
            self.H_ext = cp.zeros(2)
        else:
            self.H_ext = 0
        self.energy_Zeeman_update()
    
    def energy_Zeeman_setField(self, magnitude=0, angle=0): # TODO: is this a good name for setting the external field?
        if self.in_plane:
            self.H_ext = magnitude*cp.array([math.cos(angle), math.sin(angle)])
        else:
            self.H_ext = magnitude
            if angle != 0: warnings.warn('You tried to set the angle of an out-of-plane field in Magnets.energy_Zeeman_setField().')

    def energy_Zeeman_update(self):
        if self.in_plane:
            self.E_Zeeman = -cp.multiply(self.m, self.H_ext[0]*self.orientation[:,:,0] + self.H_ext[1]*self.orientation[:,:,1])
        else:
            self.E_Zeeman = -self.m*self.H_ext


    def energy_dipolar_init(self):
        if 'dipolar' not in self.energies: self.energies.append('dipolar')
        self.E_dipolar = cp.zeros_like(self.xx)
        # Let us first make the four-mirrored distance matrix rinv3
        # WARN: this four-mirrored technique only works if (dx, dy) is the same for every cell everywhere!
        # This could be generalized by calculating a separate rrx and rry for each magnet in a unit cell similar to toolargematrix_o{x,y}
        rrx = self.xx - self.xx[0,0]
        rry = self.yy - self.yy[0,0]
        rr_sq = rrx**2 + rry**2
        rr_sq[0,0] = cp.inf
        rr_inv = rr_sq**(-1/2) # Due to the previous line, this is now never infinite
        rr_inv3 = rr_inv**3
        rinv3 = _mirror4(rr_inv3)
        # Now we determine the normalized rx and ry
        ux = _mirror4(rrx*rr_inv, negativex=True)
        uy = _mirror4(rry*rr_inv, negativey=True)
        # Now we initialize the full ox
        if self.in_plane:
            unitcell_ox = self.orientation[:self.unitcell.y,:self.unitcell.x,0]
            unitcell_oy = self.orientation[:self.unitcell.y,:self.unitcell.x,1]
        else:
            unitcell_ox = unitcell_oy = cp.zeros((self.unitcell.y, self.unitcell.x))
        num_unitcells_x = 2*math.ceil(self.nx/self.unitcell.x) + 1
        num_unitcells_y = 2*math.ceil(self.ny/self.unitcell.y) + 1
        toolargematrix_ox = cp.tile(unitcell_ox, (num_unitcells_y, num_unitcells_x)) # This is the maximum that we can ever need (this maximum
        toolargematrix_oy = cp.tile(unitcell_oy, (num_unitcells_y, num_unitcells_x)) # occurs when the simulation does not cut off any unit cells)
        # Now comes the part where we start splitting the different cells in the unit cells
        self.Dipolar_unitcell = [[None for _ in range(self.unitcell.x)] for _ in range(self.unitcell.y)]
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                if self.in_plane:
                    ox1, oy1 = unitcell_ox[y,x], unitcell_oy[y,x] # Scalars
                    if ox1 == oy1 == 0:
                        continue # Empty cell in the unit cell, so keep self.Dipolar_unitcell[y][x] equal to None
                    # Get the useful part of toolargematrix_o{x,y} for this (x,y) in the unit cell
                    slice_startx = (self.unitcell.x - ((self.nx-1)%self.unitcell.x) + x) % self.unitcell.x # Final % not strictly necessary because
                    slice_starty = (self.unitcell.y - ((self.ny-1)%self.unitcell.y) + y) % self.unitcell.y # toolargematrix_o{x,y} large enough anyway
                    ox2 = toolargematrix_ox[slice_starty:slice_starty+2*self.ny-1,slice_startx:slice_startx+2*self.nx-1]
                    oy2 = toolargematrix_oy[slice_starty:slice_starty+2*self.ny-1,slice_startx:slice_startx+2*self.nx-1]
                    kernel1 = ox1*ox2*(3*ux**2 - 1)
                    kernel2 = oy1*oy2*(3*uy**2 - 1)
                    kernel3 = 3*(ux*uy)*(ox1*oy2 + oy1*ox2)
                    kernel = -(kernel1 + kernel2 + kernel3)*rinv3
                else:
                    kernel = rinv3 # 'kernel' for out-of-plane is very simple
                    self.Dipolar_unitcell[y][x] = rinv3 # 'kernel' for out-of-plane is very simple
                if self.PBC: # Just copy the kernel 8 times, for the 8 'nearest simulations'
                    kernelcopy = kernel.copy()
                    kernel[:,self.nx:] += kernelcopy[:,:self.nx-1]
                    kernel[self.ny:,self.nx:] += kernelcopy[:self.ny-1,:self.nx-1]
                    kernel[self.ny:,:] += kernelcopy[:self.ny-1,:]
                    kernel[self.ny:,:self.nx-1] += kernelcopy[:self.ny-1,self.nx:]
                    kernel[:,:self.nx-1] += kernelcopy[:,self.nx:]
                    kernel[:self.ny-1,:self.nx-1] += kernelcopy[self.ny:,self.nx:]
                    kernel[:self.ny-1,:] += kernelcopy[self.ny:,:]
                    kernel[:self.ny-1,self.nx:] += kernelcopy[self.ny:,:self.nx-1]
                self.Dipolar_unitcell[y][x] = kernel
    
    def energy_dipolar_single(self, index2D):
        ''' This calculates the dipolar interaction energy between magnet <i> and j,
            where j is the index in the output array. '''
        # First we get the x and y coordinates of magnet <i> in its unit cell
        y, x = index2D
        x_unitcell = int(x) % self.unitcell.x
        y_unitcell = int(y) % self.unitcell.y
        # The kernel to use is then
        kernel = self.Dipolar_unitcell[y_unitcell][x_unitcell]
        if kernel is not None:
            # Multiply with the magnetization
            usefulkernel = kernel[self.ny-1-y:2*self.ny-1-y,self.nx-1-x:2*self.nx-1-x]
            return self.m[index2D]*cp.multiply(self.m, usefulkernel)
        else:
            return cp.zeros_like(self.m)
    
    def energy_dipolar_update(self, index2D):
        ''' <i> is the index of the magnet that was switched. '''
        interaction = self.energy_dipolar_single(index2D)
        self.E_dipolar += 2*interaction
        self.E_dipolar[index2D] *= -1 # This magnet switched, so all its interactions are inverted

    def energy_dipolar_full(self):
        ''' Calculates (from scratch!) the interaction energy of each magnet with all others. '''
        total_energy = cp.zeros_like(self.m)
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                kernel = self.Dipolar_unitcell[y][x]
                if kernel is None:
                    continue
                else:
                    partial_m = cp.zeros_like(self.m)
                    partial_m[y::self.unitcell.y, x::self.unitcell.x] = self.m[y::self.unitcell.y, x::self.unitcell.x]

                    total_energy = total_energy + partial_m*signal.convolve2d(kernel, self.m, mode='valid')
        self.E_dipolar = total_energy
        
    def energy_exchange_init(self, J):
        if 'exchange' not in self.energies: self.energies.append('exchange')
        self.Exchange_J = J
        self.Exchange_interaction = self._get_nearest_neighbors()

    def energy_exchange_update(self):
        if self.in_plane:
            mx = self.orientation[:,:,0]*self.m
            my = self.orientation[:,:,1]*self.m
            sum_mx = signal.convolve2d(mx, self.Exchange_interaction, mode='same', boundary='wrap' if self.PBC else 'fill')
            sum_my = signal.convolve2d(my, self.Exchange_interaction, mode='same', boundary='wrap' if self.PBC else 'fill')
            self.E_exchange = -self.Exchange_J*(cp.multiply(sum_mx, mx) + cp.multiply(sum_my, my))
        else:
            self.E_exchange = -self.Exchange_J*cp.multiply(signal.convolve2d(self.m, self.Exchange_interaction, mode='same', boundary='wrap' if self.PBC else 'fill'), self.m)


    def update(self):
        if self.T == 0:
            warnings.warn('Temperature is zero, so no switch will be simulated.', stacklevel=2)
            return # We just warned that no switch will be simulated, so let's keep our word
        
        self._update_old()
        # self._update_Glauber()
    
    def _update_Glauber(self):
        # WARN: THIS ONLY WORKS IF self.E_int JUST BECOMES NEGATIVE WHEN SWITCHING THE MAGNET!!!
        # 1) Choose a magnet at random
        nonzero_x, nonzero_y = cp.nonzero(self.occupation)
        nonzero_idx = np.random.choice(self.n, 1)
        idx = (nonzero_x[nonzero_idx], nonzero_y[nonzero_idx])
        # 2) Compute the change in energy if it were to flip.
        energy_change = -2*self.E_int[idx] # TODO: revisit energy code, to make it easier to calculate the complete energy change if a single magnet would switch
        # 3) Flip the spin with a certain exponential probability
        exponential = math.exp(-energy_change/self.T)
        prob = exponential/(1+exponential)
        if cp.random.random() < prob: # Time is not defined in the Glauber model.
            self.m[idx] = - self.m[idx]
            self.energy(single=True, index2D=idx)
        # TODO: glauber can easily be expanded to switch multiple at once, by using superposed supergrid
    
    def _update_old(self):
        ''' Performs a single magnetization switch. '''
        self.barrier = (self.E_b - self.E_int)/self.occupation # Divide by occupation to make non-occupied grid cells have infinite barrier
        minBarrier = cp.min(self.barrier)
        self.barrier -= minBarrier # Energy is relative, so set min(E) to zero (this prevents issues at low T)
        with np.errstate(over='ignore'): # Ignore overflow warnings in the exponential: such high barriers wouldn't switch anyway
            # There is no real speed difference between XORWOW or MRG32k3a or Philox4x3210 random generators, so we just don't bother.
            taus = cp.random.uniform(size=self.barrier.shape)*cp.exp(self.barrier/self.T)
            indexmin2D = divmod(cp.argmin(taus), self.m.shape[1]) # cp.unravel_index(indexmin, self.m.shape) # The min(tau) index in 2D form for easy indexing
            self.m[indexmin2D] = -self.m[indexmin2D]
            self.t += taus[indexmin2D]*(-cp.log(1-taus[indexmin2D]))*cp.exp(minBarrier/self.T) # This can become cp.inf quite quickly if T is small
        
        self.energy(single=True, index2D=indexmin2D)

    def minimize(self):
        self.energy()
        indexmax = cp.argmax(self.E_int, axis=None)
        self.m.flat[indexmax] = -self.m.flat[indexmax]
    

    def save_history(self, *, E_tot=None, t=None, T=None, m_tot=None):
        ''' Records E_tot, t, T and m_tot as they were last calculated. This default behavior can be overruled: if
            passed as keyword parameters, their arguments will be saved instead of the self.<E_tot|t|T|m_tot> value(s).
        '''
        self.history.E.append(float(self.E_tot) if E_tot is None else float(E_tot))
        self.history.t.append(float(self.t) if t is None else float(t))
        self.history.T.append(float(self.T) if T is None else float(T))
        self.history.m.append(float(self.m_tot) if m_tot is None else float(m_tot))
    
    def clear_history(self):
        self.history.clear()
    

    def autocorrelation_fast(self, max_distance):
        max_distance = round(max_distance)
        s = cp.shape(self.xx)
        if not(hasattr(self, 'Distances')):
            # First calculate the distance between all spins in the simulation.
            self.Distances = (self.xx**2 + self.yy**2)**(1/2)
            self.Distance_range = math.ceil(cp.max(self.Distances))
            self.Distances_floor = cp.floor(self.Distances)
            # Then, calculate how many multiplications hide behind each cell in the convolution matrix, so we can normalize.
            self.corr_norm = 1/signal.convolve2d(cp.ones_like(self.m), cp.ones_like(self.m), mode='full', boundary='fill')
            # Then, calculate the correlation of the occupation, since not each position contains a spin
            fullcorr = signal.convolve2d(self.occupation, cp.flipud(cp.fliplr(self.occupation)), mode='full', boundary='fill')*self.corr_norm
            self.corr_occupation = fullcorr[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)] # Lower right quadrant of occupationcor because the other quadrants should be symmetrical
            self.corr_occupation[self.corr_occupation > 0] = 1
        # Now, convolve self.m with its point-mirrored/180°-rotated counterpart
        if self.in_plane:
            corr_x = signal.convolve2d(self.m*self.orientation[:,:,0], cp.flipud(cp.fliplr(self.m*self.orientation[:,:,0])), mode='full', boundary='fill')*self.corr_norm
            corr_y = signal.convolve2d(self.m*self.orientation[:,:,1], cp.flipud(cp.fliplr(self.m*self.orientation[:,:,1])), mode='full', boundary='fill')*self.corr_norm
            corr = corr_x + corr_y
        else:
            corr = signal.convolve2d(self.m, cp.flipud(cp.fliplr(self.m)), mode='full', boundary='fill')*self.corr_norm
        corr = corr*cp.size(self.m)/cp.sum(self.corr_occupation) # Put between 0 and 1
        self.correlation = corr[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)]**2
        self.correlation = cp.multiply(self.correlation, self.corr_occupation)
        
        # Prepare distance bins etc.
        corr_binned = cp.zeros(max_distance + 1) # How much the magnets correlate over a distance [i]
        distances = cp.linspace(0, max_distance, num=max_distance+1) # Use cp.linspace to get float, cp.arange to get int
        # Now loop over all the interesting distances
        for i, d in enumerate(distances):
            corr_binned[i] = cp.mean(self.correlation[cp.where(cp.isclose(self.Distances_floor, d))])
        corr_length = cp.sum(cp.multiply(abs(corr_binned), distances))
        return corr_binned.get(), distances.get(), float(corr_length)

    ######## Now, some useful functions when subclassing this class
    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def _get_appropriate_avg(self):
        ''' Returns the most appropriate averaging mask for a given type of ASI '''
        return 'cross'

    def _get_plotting_params(self):
        return {
            'quiverscale': 0.7
        }

    def _get_AFMmask(self):
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')


def _mirror4(arr, negativex=False, negativey=False):
    ny, nx = arr.shape
    arr4 = cp.zeros((2*ny-1, 2*nx-1))
    xp = -1 if negativex else 1
    yp = -1 if negativey else 1
    arr4[ny-1:, nx-1:] = arr
    arr4[ny-1:, nx-1::-1] = xp*arr
    arr4[ny-1::-1, nx-1:] = yp*arr
    arr4[ny-1::-1, nx-1::-1] = xp*yp*arr
    return arr4


@dataclass
class History:
    ''' Stores the history of the energy, temperature, time, and average magnetization. '''
    E: list = field(default_factory=list)
    T: list = field(default_factory=list)
    t: list = field(default_factory=list)
    m: list = field(default_factory=list)

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