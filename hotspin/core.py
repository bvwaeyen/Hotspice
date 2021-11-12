import ctypes
import math
import matplotlib
import warnings

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cupyx.scipy import signal
from dataclasses import dataclass, field
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import MultiCursor


ctypes.windll.shcore.SetProcessDpiAwareness(2) # (For Windows 10/8/7) this makes the matplotlib plots smooth on high DPI screens
matplotlib.rcParams["image.interpolation"] = 'none' # 'none' works best for large images scaled down, 'nearest' for the opposite

"""
TODO (summary):
(!: priority, -: should do at some point in time, .: perhaps implement perhaps not)
! use Glauber monte carlo markov chain on Ising model to determine which magnet will switch
  (will not improve performance by much I think, since this will still only switch 1 magnet each time)
- update the animate_temp_rise function with the modern 'API' or however to call this
- in-plane exchange energy
- sort out the AFM-ness and its normalization etc.
. can implement linear transformations if I want to
. can implement random defects if I want to
- move plotting functions to a separate module 'hotspin.plottools'
- make unit tests
"""

class Magnets:
    def __init__(self, nx, ny, dx, dy, T=1, E_b=1, Msat=1, m_type='ip', pattern='random', energies=('dipolar',), PBC=False):
        '''
            The position of magnets is specified using <nx> and <a>. 
            The initial configuration of a Magnets geometry consists of 3 parts:
             1) m_type:  Magnets can be in-plane or out-of-plane: 'ip' or 'op', respectively.
             2) config:  Now defined through subclasses. # TODO: deprecate this
             3) pattern: The initial magnetization direction (e.g. up/down) can be 'uniform', 'AFM' or 'random'.
            One can also specify which energy components should be considered: any of 'dipolar', 'Zeeman' and 'exchange'.
                If you want to adjust the parameters of these energies, than call energy_<type>_init(<parameters>) manually.
            # TODO: linear transformations (e.g. skewing or squeezing) should be relatively easy to implement by acting on xx, yy
            #       see https://matplotlib.org/stable/gallery/images_contours_and_fields/affine_image.html for the imshows then
        '''
        self.T = T
        self.t = 0.
        self.E_b = E_b
        self.Msat = Msat
        self.m_type = m_type
        self.energies = list(energies)
        
        # initialize the coordinates based on nx, (ny) and L
        self.xx, self.yy = cp.meshgrid(cp.linspace(0, dx*(nx-1), nx), cp.linspace(0, dy*(ny-1), ny))
        self.index = range(self.xx.size)
        self.ixx, self.iyy = cp.meshgrid(cp.arange(0, self.xx.shape[1]), cp.arange(0, self.yy.shape[0]))
        self.x_min, self.y_min, self.x_max, self.y_max = float(self.xx[0,0]), float(self.yy[0,0]), float(self.xx[-1,-1]), float(self.yy[-1,-1])

        self.unitcell = Vec2D(*self._get_unitcell())
        self.PBC = PBC
        if self.PBC:
            if nx % self.unitcell.x != 0 or ny % self.unitcell.y != 0:
                warnings.warn(f"""Be careful with PBC, as there are not an integer number of unit cells in the simulation! Hence, the boundaries do not nicely fit together. Adjust nx or ny to alleviate this problem (unit cell has size {self.unitcell.x}x{self.unitcell.y}).""", stacklevel=2)

        self.history = History()

        self.mask = self._get_mask().astype(bool).astype(int) # Make sure that it is either 0 or 1
        if self.m_type == 'ip':
            self._initialize_ip()
        self.energy_init() # This needs self.orientation
        self.initialize_m(pattern) # This needs energy kernels to be ready

    def _get_unitcell(self):
        return Vec2D(1, 1)
    
    def _get_mask(self):
        return cp.array([[1]])

    def initialize_m(self, pattern='random'):
        ''' Initializes the self.m (array of -1, 0 or 1) and mask.
            @param pattern [str]: can be any of "random", "uniform", "AFM".
        '''
        self._set_m(pattern)
        self.m = cp.multiply(self.m, self.mask)
        self.energy() # Have to recalculate all the energies since m changed completely
    
    def _set_orientation(self):
        self.orientation = cp.ones((*self.xx.shape, 2))/math.sqrt(2)

    def _initialize_ip(self):
        ''' Initialize the angles of all the magnets.
            This function should only be called by the Magnets() class itself, not by the user.
        '''
        # This sets the angle of all the magnets (this is of course only applicable in the in-plane case)
        assert self.m_type == 'ip', "Can not _initialize_ip() if m_type != 'ip'."
        self._set_orientation()


    @property
    def m_tot(self):
        if self.m_type == 'op':
            return cp.mean(self.m)
        elif self.m_type == 'ip':
            self.m_tot_x = cp.mean(cp.multiply(self.m, self.orientation[:,:,0]))
            self.m_tot_y = cp.mean(cp.multiply(self.m, self.orientation[:,:,1]))
            return (self.m_tot_x**2 + self.m_tot_y**2)**(1/2)


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
        if self.m_type == 'op':
            self.H_ext = 0.
        elif self.m_type == 'ip':
            self.H_ext = cp.zeros(2)
        self.energy_Zeeman_update()

    def energy_Zeeman_update(self):
        if self.m_type == 'op':
            self.E_Zeeman = -self.m*self.H_ext
        elif self.m_type == 'ip':
            self.E_Zeeman = -cp.multiply(self.m, self.H_ext[0]*self.orientation[:,:,0] + self.H_ext[1]*self.orientation[:,:,1])

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
        if self.m_type == 'ip':
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
                if self.m_type == 'op':
                    kernel = rinv3 # 'kernel' for out-of-plane is very simple
                    self.Dipolar_unitcell[y][x] = rinv3 # 'kernel' for out-of-plane is very simple
                elif self.m_type == 'ip':
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
            E_now = self.m[index2D]*cp.multiply(self.m, usefulkernel)
        else:
            E_now = cp.zeros_like(self.m)
        return E_now
    
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
        if self.m_type == 'op':
            self.E_exchange = -self.Exchange_J*cp.multiply(signal.convolve2d(self.m, self.Exchange_interaction, mode='same', boundary='wrap' if self.PBC else 'fill'), self.m)
        elif self.m_type == 'ip':
            mx = self.orientation[:,:,0]*self.m
            my = self.orientation[:,:,1]*self.m
            sum_mx = signal.convolve2d(mx, self.Exchange_interaction, mode='same', boundary='wrap' if self.PBC else 'fill')
            sum_my = signal.convolve2d(my, self.Exchange_interaction, mode='same', boundary='wrap' if self.PBC else 'fill')
            self.E_exchange = -self.Exchange_J*(cp.multiply(sum_mx, mx) + cp.multiply(sum_my, my))

    def update(self):
        """ Performs a single magnetization switch. """
        if self.T == 0:
            warnings.warn('Temperature is zero, so no switch will be simulated.', stacklevel=2)
            return # We just warned that no switch will be simulated, so let's keep our word
        self.barrier = (self.E_b - self.E_int)/self.mask # Divide by mask to make non-occupied grid cells have infinite barrier
        minBarrier = cp.min(self.barrier)
        self.barrier -= minBarrier # Energy is relative, so set min(E) to zero (this solves issues at low T)
        with np.errstate(over='ignore'): # Ignore overflow warnings in the exponential: such high barriers wouldn't switch anyway
            # There is no real speed difference between XORWOW or MRG32k3a or Philox4x3210 random generators, so we just don't bother.
            taus = cp.random.uniform(size=self.barrier.shape) # TODO: use Glauber monte carlo model (i.e. choose random magnet, and switch if energy would decrease)
            taus *= cp.exp(self.barrier/self.T)
            indexmin = cp.argmin(taus)
            indexmin2D = cp.unravel_index(indexmin, self.m.shape) # The min(tau) index in 2D form for easy indexing
            self.m[indexmin2D] = -self.m[indexmin2D]
            self.t += taus[indexmin2D]*(-cp.log(1-taus[indexmin2D]))*cp.exp(minBarrier/self.T) # This can become cp.inf quite quickly if T is small
        
        self.energy(single=True, index2D=indexmin2D)

    def minimize(self):
        self.energy()
        indexmax = cp.argmax(self.E_int, axis=None)
        self.m.flat[indexmax] = -self.m.flat[indexmax]
    

    def save_history(self, *, E_tot=None, t=None, T=None, m_tot=None):
        """ Records E_tot, t, T and m_tot as they were last calculated. This default behavior can be overruled: if
            passed as keyword parameters, their arguments will be saved instead of the self.<E_tot|t|T|m_tot> value(s).
        """
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
            # Then, calculate the correlation of the mask, since not each position contains a spin
            maskcor = signal.convolve2d(self.mask, cp.flipud(cp.fliplr(self.mask)), mode='full', boundary='fill')*self.corr_norm
            self.corr_mask = maskcor[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)] # Lower right quadrant of maskcor because the other quadrants should be symmetrical
            self.corr_mask[self.corr_mask > 0] = 1
        # Now, convolve self.m with its point-mirrored/180°-rotated counterpart
        if self.m_type == 'op':
            corr = signal.convolve2d(self.m, cp.flipud(cp.fliplr(self.m)), mode='full', boundary='fill')*self.corr_norm
        elif self.m_type == 'ip':
            corr_x = signal.convolve2d(self.m*self.orientation[:,:,0], cp.flipud(cp.fliplr(self.m*self.orientation[:,:,0])), mode='full', boundary='fill')*self.corr_norm
            corr_y = signal.convolve2d(self.m*self.orientation[:,:,1], cp.flipud(cp.fliplr(self.m*self.orientation[:,:,1])), mode='full', boundary='fill')*self.corr_norm
            corr = corr_x + corr_y
        corr = corr*cp.size(self.m)/cp.sum(self.corr_mask) # Put between 0 and 1
        self.correlation = corr[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)]**2
        self.correlation = cp.multiply(self.correlation, self.corr_mask)
        
        # Prepare distance bins etc.
        corr_binned = cp.zeros(max_distance + 1) # How much the magnets correlate over a distance [i]
        distances = cp.linspace(0, max_distance, num=max_distance+1) # Use cp.linspace to get float, cp.arange to get int
        # Now loop over all the interesting distances
        for i, d in enumerate(distances):
            corr_binned[i] = cp.mean(self.correlation[cp.where(cp.isclose(self.Distances_floor, d))])
        corr_length = cp.sum(cp.multiply(abs(corr_binned), distances))
        return corr_binned.get(), distances.get(), float(corr_length)

    def _get_nearest_neighbors(self):
        return cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # Below here are some graphical functions (plot magnetization profile etc.)
    def _get_averaged_extent(self, avg):
        ''' Returns the extent that can be used in imshow when plotting an averaged quantity. '''
        avg = self._resolve_avg(avg)
        mask = self._get_avg_mask(avg=avg)
        if self.PBC:
            movex, movey = 0.5*self.dx, 0.5*self.dy
        else:
            movex, movey = mask.shape[1]/2*self.dx, mask.shape[0]/2*self.dy # The averaged imshow should be displaced by this much
        return [self.x_min-self.dx+movex,self.x_max-movex+self.dx,self.y_min-self.dy+movey,self.y_max-movey+self.dy]
        
    def _get_appropriate_avg(self):
        ''' Returns the most appropriate averaging mask for a given type of ASI '''
        return 'cross'

    def _resolve_avg(self, avg):
        ''' If avg is str then determine if it is valid, otherwise auto-determine which averaging method is appropriate. '''
        if isinstance(avg, str):
            assert avg in ['point', 'cross', 'square', 'hexagon', 'triangle'], "Unsupported averaging mask: %s" % avg
        else: # It is something which can be truthy or falsy
            avg = self._get_appropriate_avg() if avg else 'point'
        return avg

    def _get_avg_mask(self, avg=None):
        ''' Returns the raw averaging mask as a 2D array. Note that this obviously does not include
            any removal of excess zero-rows etc., that is the task of self.get_m_angles.
            Note that this returns a CuPy array for performance reasons, since this is a 'hidden' _function anyway.
        '''
        avg = self._resolve_avg(avg)
        if avg == 'point':
            mask = [[1]]
        elif avg == 'cross': # cross ⁛
            mask = [[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]]
        elif avg == 'square': # square □
            mask = [[1, 1, 1], 
                    [1, 0, 1], 
                    [1, 1, 1]]
        elif avg == 'hexagon':
            mask = [[0, 1, 0, 1, 0], 
                    [1, 0, 0, 0, 1], 
                    [0, 1, 0, 1, 0]]
        elif avg == 'triangle':
            mask = [[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]]
        return cp.array(mask, dtype='float') # If mask would be int, then precision of convolve2d is also int instead of float

    def get_m_polar(self, m=None, avg=True):
        '''
            Returns the magnetization angle and magnitude (can be averaged using the averaging method specified by <avg>).
            If the local average magnetization is zero, the corresponding angle is NaN.
            If there are no magnets to average around a given cell, then the angle and magnitude are both NaN.
            @param m [2D array] (self.m): The magnetization profile that should be averaged.
            @param avg [str|bool] (True): can be any of True, False, 'point', 'cross', 'square', 'triangle', 'hexagon':
                True: automatically determines the appropriate averaging method.
                False|'point': no averaging at all, just calculates the angle of each individual spin.
                'cross': averages the spins north, east, south and west of each position.
                'square': averages the 8 nearest neighbors of each cell.
                'triangle': averages the three magnets connected to a corner of a hexagon in the kagome geometry.
                'hexagon:' averages each hexagon in kagome ASI, or each star in triangle ASI.
            @return [(2D np.array, 2D np.array)]: a tuple containing two arrays, namely the (averaged) magnetization
                angle and magnitude, respecively, for each relevant position in the simulation.
                Angles lay between 0 and 2*pi, magnitudes between 0 and self.Msat.
                !! This does not necessarily have the same shape as <m> !!
        '''
        if m is None: m = self.m
        avg = self._resolve_avg(avg)

        if self.m_type == 'ip':
            x_comp = cp.multiply(m, self.orientation[:,:,0])
            y_comp = cp.multiply(m, self.orientation[:,:,1])
        else:
            x_comp = m
            y_comp = cp.zeros_like(m)
        mask = self._get_avg_mask(avg=avg)
        if self.PBC:
            magnets_in_avg = signal.convolve2d(cp.abs(m), mask, mode='same', boundary='wrap')
            x_comp_avg = signal.convolve2d(x_comp, mask, mode='same', boundary='wrap')/magnets_in_avg
            y_comp_avg = signal.convolve2d(y_comp, mask, mode='same', boundary='wrap')/magnets_in_avg
        else:
            magnets_in_avg = signal.convolve2d(cp.abs(m), mask, mode='valid', boundary='fill')
            x_comp_avg = signal.convolve2d(x_comp, mask, mode='valid', boundary='fill')/magnets_in_avg
            y_comp_avg = signal.convolve2d(y_comp, mask, mode='valid', boundary='fill')/magnets_in_avg
        angles_avg = cp.arctan2(y_comp_avg, x_comp_avg) % (2*math.pi)
        magnitudes_avg = cp.sqrt(x_comp_avg**2 + y_comp_avg**2)*self.Msat
        useless_angles = cp.where(cp.logical_and(cp.isclose(x_comp_avg, 0), cp.isclose(y_comp_avg, 0)), cp.NaN, 1) # No well-defined angle
        useless_magnitudes = cp.where(magnets_in_avg == 0, cp.NaN, 1) # No magnet (the NaNs here will be a subset of useless_angles)
        angles_avg *= useless_angles
        magnitudes_avg *= useless_magnitudes
        if avg == 'triangle':
            angles_avg = angles_avg[1::2,1::2]
            magnitudes_avg = magnitudes_avg[1::2,1::2]
        elif avg == 'hexagon': # Only keep the centers of hexagons, throw away the rest
            angles_avg = angles_avg[::2,::2]
            magnitudes_avg = magnitudes_avg[::2,::2]
            ixx, iyy = cp.meshgrid(cp.arange(0, angles_avg.shape[1]), cp.arange(0, angles_avg.shape[0])) # DO NOT REMOVE THIS, THIS IS NOT THE SAME AS self.ixx, self.iyy!
            NaN_mask = (ixx + iyy) % 2 == 1 # These are not the centers of hexagons, so dont draw these
            angles_avg[NaN_mask] = cp.NaN
            magnitudes_avg[NaN_mask] = cp.NaN
        return angles_avg, magnitudes_avg
    
    def polar_to_rgb(self, angles=None, magnitudes=None, m=None, avg=True, fill=False, autoscale=True):
        ''' Returns the rgb values for the polar coordinates defined by angles and magnitudes. 
            TAKES CUPY ARRAYS AS INPUT, YIELDS NUMPY ARRAYS AS OUTPUT
            @param angles [2D cp.array()] (None): The averaged angles.
        '''
        if angles is None or magnitudes is None:
            angles, magnitudes = self.get_m_polar(m=m, avg=avg)
            if autoscale:
                avgmethod = self._resolve_avg(avg) # TODO: make this independent of config
                if self.config in ['pinwheel', 'square'] and avgmethod in ['cross', 'square']:
                    magnitudes *= math.sqrt(2)*.999
                elif self.config in ['kagome', 'triangle'] and avgmethod in ['hexagon', 'triangle']:
                    magnitudes *= 1.5*.999
        assert angles.shape == magnitudes.shape, "polar_to_hsv() did not receive angle and magnitude arrays of the same shape."
        
        # Normalize to ranges between 0 and 1 and determine NaN-positions
        angles = angles/2/math.pi
        magnitudes = magnitudes/self.Msat
        NaNangles = cp.isnan(angles)
        NaNmagnitudes = cp.isnan(magnitudes)
        # Create hue, saturation and value arrays
        hue = cp.zeros_like(angles)
        saturation = cp.ones_like(angles)
        value = cp.zeros_like(angles)
        # Situation 1: angle and magnitude both well-defined (an average => color (hue=angle, saturation=1, value=magnitude))
        affectedpositions = cp.where(cp.logical_and(cp.logical_not(NaNangles), cp.logical_not(NaNmagnitudes)))
        hue[affectedpositions] = angles[affectedpositions]
        value[affectedpositions] = magnitudes[affectedpositions]
        # Situation 2: magnitude is zero, so angle is NaN (zero average => black (hue=anything, saturation=anything, value=0))
        affectedpositions = cp.where(cp.logical_and(NaNangles, magnitudes == 0))
        value[affectedpositions] = 0
        # Situation 3: magnitude is NaN, so angle is NaN (no magnet => white (hue=0, saturation=0, value=1))
        affectedpositions = cp.where(cp.logical_and(NaNangles, NaNmagnitudes))
        saturation[affectedpositions] = 0
        value[affectedpositions] = 1
        # Create the hsv matrix with correct axes ordering for matplotlib.color.hsv_to_rgb:
        hsv = np.array([hue.get(), saturation.get(), value.get()]).swapaxes(0, 2).swapaxes(0, 1)
        if fill: hsv = fill_neighbors(hsv, cp.logical_and(NaNangles, NaNmagnitudes))
        rgb = hsv_to_rgb(hsv)
        return rgb
    
    def _get_plotting_params(self):
        return {
            'quiverscale': 0.7
        }

    def show_m(self, m=None, avg=True, show_energy=True, fill=False):
        ''' Shows two (or three if <show_energy> is True) figures displaying the direction of each spin: one showing
            the (locally averaged) angles, another quiver plot showing the actual vectors. If <show_energy> is True,
            a third and similar plot, displaying the interaction energy of each spin, is also shown.
            @param m [2D array] (self.m): the direction (+1 or -1) of each spin on the geometry. Default is the current
                magnetization profile. This is useful if some magnetization profiles have been saved manually, while 
                self.update() has been called since: one can then pass these saved profiles as the <m> parameter to
                draw them onto the geometry stored in <self>.
            @param avg [str|bool] (True): can be any of True, False, 'point', 'cross', 'square', 'triangle', 'hexagon'.
            @param show_energy [bool] (True): if True, a 2D plot of the energy is shown in the figure as well.
            @param fill [bool] (False): if True, empty pixels are interpolated if all neighboring averages are equal.
        '''
        avg = self._resolve_avg(avg)
        if m is None: m = self.m
        show_quiver = self.m.size < 1e5 and self.m_type == 'ip' # Quiver becomes very slow for more than 100k cells, so just dont show it then
        averaged_extent = self._get_averaged_extent(avg)
        full_extent = [self.x_min-self.dx/2,self.x_max+self.dx/2,self.y_min-self.dy/2,self.y_max+self.dx/2]

        num_plots = 1
        num_plots += 1 if show_energy else 0
        num_plots += 1 if show_quiver else 0
        axes = []
        fig = plt.figure(figsize=(3.5*num_plots, 3))
        ax1 = fig.add_subplot(1, num_plots, 1)
        im = self.polar_to_rgb(m=m, avg=avg, fill=fill)
        im1 = ax1.imshow(im, cmap='hsv' if self.m_type == 'ip' else 'gray', origin='lower', vmin=0, vmax=2*cp.pi,
                            extent=averaged_extent) # extent doesnt work perfectly with triangle or kagome but is still ok
        c1 = plt.colorbar(im1)
        c1.ax.get_yaxis().labelpad = 30
        c1.ax.set_ylabel(f"Averaged magnetization angle [rad]\n('{avg}' average{', PBC' if self.PBC else ''})", rotation=270, fontsize=12)
        axes.append(ax1)
        if show_quiver:
            ax2 = fig.add_subplot(1, num_plots, 2, sharex=ax1, sharey=ax1)
            ax2.set_aspect('equal')
            nonzero = self.m.get().nonzero()
            quiverscale = self._get_plotting_params()['quiverscale']
            ax2.quiver(self.xx.get()[nonzero], self.yy.get()[nonzero], 
                    cp.multiply(m, self.orientation[:,:,0]).get()[nonzero], cp.multiply(m, self.orientation[:,:,1]).get()[nonzero],
                    pivot='mid', scale=quiverscale, headlength=17, headaxislength=17, headwidth=7, units='xy') # units='xy' makes arrows scale correctly when zooming
            ax2.set_title(r'$m$')
            axes.append(ax2)
        if show_energy:
            ax3 = fig.add_subplot(1, num_plots, num_plots, sharex=ax1, sharey=ax1)
            im3 = ax3.imshow(self.E_int.get(), origin='lower',
                             extent=full_extent)
            plt.colorbar(im3)
            ax3.set_title(r'$E_{int}$')
            axes.append(ax3)
        multi = MultiCursor(fig.canvas, axes, color='black', lw=1, linestyle='dotted', horizOn=True, vertOn=True) # Assign to variable to prevent garbage collection
        plt.gcf().tight_layout()
        plt.show()

    def show_history(self, y_quantity=None, y_label=r'Average magnetization'):
        ''' Plots <y_quantity> (default: average magnetization (self.history.m)) and total energy (self.history.E)
            as a function of either the time or the temperature: if the temperature (self.history.T) is constant, 
            then the x-axis will represent the time (self.history.t), otherwise it represents the temperature.
            @param y_quantity [1D array] (self.m): The quantity to be plotted as a function of T or t.
            @param y_label [str] (r'Average magnetization'): The y-axis label in the plot.
        '''
        if y_quantity is None:
            y_quantity = self.history.m
        if cp.all(cp.isclose(self.history.T, self.history.T[0])):
            x_quantity, x_label = self.history.t, 'Time [a.u.]'
        else:
            x_quantity, x_label = self.history.T, 'Temperature [a.u.]'
        assert len(y_quantity) == len(x_quantity), "Error in show_history: <y_quantity> has different length than %s history." % x_label.split(' ')[0].lower()

        fig = plt.figure(figsize=(4, 6))
        ax1 = fig.add_subplot(211)
        ax1.plot(x_quantity, y_quantity)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax2 = fig.add_subplot(212)
        ax2.plot(x_quantity, self.history.E)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Total energy [a.u.]')
        plt.gcf().tight_layout()
        plt.show()
    
    def _get_AFMmask(self):
        return cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')

    def get_AFMness(self, AFM_mask=None):
        ''' Returns the average AFM-ness of self.m at the current time step, normalized to 1.
            For a perfectly uniform configuration this is 0, while for random it is 0.375.
            Note that the boundaries are not taken into account for the normalization, so the
            AFM-ness will often be slightly lower than the ideal values mentioned above.
            @param AFM_mask [2D array] (None): The mask used to determine the AFM-ness. If not
                provided explicitly, it is determined automatically based on self.config.
            @return [float]: The average normalized AFM-ness.
        '''
        AFM_mask = self._get_AFMmask() if AFM_mask is None else cp.asarray(AFM_mask)
        AFM_ness = cp.mean(cp.abs(signal.convolve2d(self.m, AFM_mask, mode='same', boundary='wrap' if self.PBC else 'fill')))
        return float(AFM_ness/cp.sum(cp.abs(AFM_mask))/cp.sum(self.mask)*self.m.size)


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

def fill_neighbors(hsv, replaceable, fillblack=True): # TODO: make this cupy if possible
    ''' THIS FUNCTION ONLY WORKS FOR GRIDS WHICH HAVE A CHESS-LIKE OCCUPATION OF THE CELLS! (cross ⁛)
        THIS FUNCTION OPERATES ON HSV VALUES, AND RETURNS HSV AS WELL!!! NOT RGB HERE!
        The 2D array <replaceable> is True at the positions of hsv which can be overwritten by this function.
        The 3D array <hsv> has the same first two dimensions as <replaceable>, with the third dimension having size 3 (h, s, v).
        Then this function overwrites the replaceables with the surrounding values at the nearest neighbors (cross neighbors ⁛),
        but only if all those neighbors are equal. This is useful for very large simulations where each cell
        occupies less than 1 pixel when plotted: by removing the replaceables, visual issues can be prevented.
        @param fillblack [bool] (True): If True, white pixels next to black pixels are colored black regardless of other neighbors.
        @return [2D np.array]: The interpolated array.
    '''
    hsv = hsv.get() if type(hsv) == cp.ndarray else np.asarray(hsv)
    replaceable = replaceable if type(replaceable) == cp.ndarray else cp.asarray(replaceable)

    # Extend arrays a bit to fill NaNs near boundaries as well
    a = np.insert(hsv, 0, hsv[1], axis=0)
    a = np.insert(a, 0, a[:,1], axis=1)
    a = cp.append(a, a[-2].reshape(1,-1,3), axis=0)
    a = cp.append(a, a[:,-2].reshape(-1,1,3), axis=1)

    N = a[:-2, 1:-1, :]
    E = a[1:-1, 2:, :]
    S = a[2:, 1:-1, :]
    W = a[1:-1, :-2, :]
    equal_neighbors = cp.logical_and(cp.logical_and(cp.isclose(N, E), cp.isclose(E, S)), cp.isclose(S, W))
    equal_neighbors = cp.logical_and(cp.logical_and(equal_neighbors[:,:,0], equal_neighbors[:,:,1]), equal_neighbors[:,:,2])

    result = cp.where(cp.repeat(cp.logical_and(replaceable, equal_neighbors)[:,:,cp.newaxis], 3, axis=2), N, cp.asarray(hsv))

    if fillblack:
        blacks = cp.where(cp.logical_and(cp.logical_or(cp.logical_or(N[:,:,2] == 0, E[:,:,2] == 0), cp.logical_or(S[:,:,2] == 0, W[:,:,2] == 0)), replaceable))
        result[blacks[0], blacks[1], 2] = 0

    return result.get()


@dataclass
class History:
    """ Stores the history of the energy, temperature, time, and average magnetization. """
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
    """ Stores x and y components, so we don't need to index [0] or [1] in a tuple, which would be unclear. """
    x: float
    y: float