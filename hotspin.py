import ctypes
import math
import matplotlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from dataclasses import dataclass, field
from matplotlib.widgets import MultiCursor
from cupyx.scipy import signal


ctypes.windll.shcore.SetProcessDpiAwareness(2) # (For Windows 10/8/7) this makes the matplotlib plots smooth on high DPI screens
matplotlib.rcParams["image.interpolation"] = 'none' # 'none' works best for large images scaled down, 'nearest' for the opposite

class Magnets:
    def __init__(self, xx, yy, T, E_b, m_type='ip', config='square', pattern='random', energies=('dipolar'), PBC=False):
        '''
            The initial configuration of a Magnets geometry consists of 3 parts:
             1) m_type:  Magnets can be in-plane or out-of-plane: 'ip' or 'op', respectively.
             2) config:  The placement of magnets on the grid can be
                    if m_type is 'op': 'full', 'chess',
                    if m_type is 'ip': 'square', 'pinwheel', 'kagome' or 'triangle'.
             3) pattern: The initial magnetization direction (e.g. up/down) can be 'uniform', 'AFM' or 'random'.
            One can also specify which energy components should be considered: any of 'dipolar', 'Zeeman' and 'exchange'.
                If you want to adjust the specifics of these energies, than call <energy>_init(<parameters>) manually.
        '''
        assert cp.shape(xx) == cp.shape(yy), "Error: xx and yy should have the same shape. Please obtain xx and yy using cp.meshgrid(x,y) to avoid this issue."
        self.xx = cp.asarray(xx)
        self.yy = cp.asarray(yy)
        self.ny, self.nx = self.xx.shape
        self.dx, self.dy = float(self.xx[1,1] - self.xx[0,0]), float(self.yy[1,1] - self.yy[0,0])
        self.x_min, self.y_min, self.x_max, self.y_max = float(self.xx[0,0]), float(self.yy[0,0]), float(self.xx[-1,-1]), float(self.yy[-1,-1])
        self.T = T
        self.t = 0.
        self.E_b = E_b
        self.m_type = m_type
        self.energies = list(energies)
        self.PBC = PBC

        self.index = range(self.xx.size)
        ix = cp.arange(0, self.xx.shape[1])
        iy = cp.arange(0, self.yy.shape[0])
        self.ixx, self.iyy = cp.meshgrid(ix, iy)

        # config disambiguation
        if m_type == 'op':
            if config in ['full']:
                self.config = 'full'
            elif config in ['chess']:
                self.config = 'chess'
            else:
                raise AssertionError(f"Invalid argument: config='{config}' not valid if m_type is 'op'.")
        elif m_type == 'ip':
            if config in ['square', 'squareASI']:
                self.config = 'square'
            elif config in ['pinwheel', 'pinwheelASI']:
                self.config = 'pinwheel'
            elif config in ['kagome', 'kagomeASI']:
                self.config = 'kagome'
            elif config in ['triangle', 'triangleASI']:
                self.config = 'triangle'
            else:
                raise AssertionError(f"Invalid argument: config='{config}' not valid if m_type is 'ip'.")

        if self.config == 'full':
            self.mask = cp.ones_like(self.xx)
            self.unitcell = Vec2D(1,1)
        elif self.config in ['chess', 'square', 'pinwheel']:
            self.mask = cp.zeros_like(self.xx)
            self.mask[(self.xx + self.yy) % 2 == 1] = 1
            self.unitcell = Vec2D(2,2)
        elif self.config in ['kagome', 'triangle']:
            self.mask = cp.zeros_like(self.xx)
            self.mask[(self.ixx + self.iyy) % 4 == 1] = 1 # One bunch of diagonals \
            self.mask[(self.ixx - self.iyy) % 4 == 3] = 1 # Other bunch of diagonals /
            self.unitcell = Vec2D(4,4)

        # Set the orientation of the islands corresponding to config
        if m_type == 'ip': 
            if self.config == 'square':
                self._initialize_ip('square', 0)
            elif self.config == 'pinwheel':
                self._initialize_ip('square', cp.pi/4)
            elif self.config == 'kagome':
                self._initialize_ip('kagome', 0)
            elif self.config == 'triangle':
                self._initialize_ip('kagome', cp.pi/2)

        # Initialize the specified energy components
        if 'dipolar' in energies:
            self.energy_dipolar_init()
        if 'Zeeman' in energies:
            self.energy_Zeeman_init()
        if 'exchange' in energies:
            self.energy_exchange_init(1)

        # Initialize self.m and the correct self.mask, this also calculates the initial energy
        self.initialize_m(pattern)

        self.history = History()


    def initialize_m(self, pattern):
        ''' Initializes the magnetization (-1, 0 or 1), mask and unit cell dimensions.
            @param pattern [str]: can be any of "random", "uniform", "AFM".
        '''
        if pattern == 'uniform':
            self.m = cp.ones(cp.shape(self.xx)) # For full, chess, square, pinwheel: this is already ok
            if self.config in ['kagome', 'triangle']:
                self.m[(self.ixx - self.iyy) % 4 == 1] = -1
        elif pattern == 'AFM':
            if self.config in ['full']:
                self.m = ((self.xx + self.yy) % 2)*2 - 1
            elif self.config in ['chess', 'square', 'pinwheel']:
                self.m = ((self.xx - self.yy)//2 % 2)*2 - 1
            elif self.config in ['kagome', 'triangle']:
                self.m = cp.ones(cp.shape(self.xx))
                self.m[(self.ixx + self.iyy) % 4 == 3] = -1
        else:
            self.m = cp.random.randint(0, 2, size=cp.shape(self.xx))*2 - 1 # Yields random -1 or 1
            if pattern != 'random': warnings.warn('Config not recognized, defaulting to "random".', stacklevel=2)

        self.m = cp.multiply(self.m, self.mask)
        self.m_tot = cp.mean(self.m)
        self.energy() # Have to recalculate all the energies since m changed completely
            
      
    def _initialize_ip(self, config, angle=0.):
        ''' Initialize the angles of all the magnets.
            This function should only be called by the Magnets() class itself, not by the user.
        '''
        # This sets the angle of all the magnets (this is of course only applicable in the in-plane case)
        assert self.m_type == 'ip', "Can not _initialize_ip() if m_type != 'ip'."
        self.orientation = np.zeros(np.shape(self.xx) + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
        mask = self.mask.get()
        yy = self.yy.get()
        if config == 'square':
            self.orientation[yy % 2 == 0,0] = math.cos(angle)
            self.orientation[yy % 2 == 0,1] = math.sin(angle)
            self.orientation[yy % 2 == 1,0] = math.cos(angle + math.pi/2)
            self.orientation[yy % 2 == 1,1] = math.sin(angle + math.pi/2)
            self.orientation[mask == 0,0] = 0
            self.orientation[mask == 0,1] = 0
        elif config == 'kagome':
            self.orientation[:,:,0] = math.cos(angle + math.pi/2)
            self.orientation[:,:,1] = math.sin(angle + math.pi/2)
            self.orientation[cp.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1).get(),0] = math.cos(angle - math.pi/6)
            self.orientation[cp.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1).get(),1] = math.sin(angle - math.pi/6)
            self.orientation[cp.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1).get(),0] = math.cos(angle + math.pi/6)
            self.orientation[cp.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1).get(),1] = math.sin(angle + math.pi/6)
            self.orientation[mask == 0,0] = 0
            self.orientation[mask == 0,1] = 0
        self.orientation = cp.asarray(self.orientation)
    

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
        # WARN: this four-mirrored technique only works if dx and dy is the same for every cell everywhere! so,
        # TODO: make the initialization of Magnets() object only take an nx, ny, dx, dy which is also much easier to use
        rrx = self.xx - self.xx[0,0]
        rry = self.yy - self.yy[0,0]
        rr_sq = rrx**2 + rry**2
        rr_sq[0,0] = cp.inf
        rr_inv = rr_sq**(-1/2) # Due to the previous line, this is now never infinite
        rr_inv3 = rr_inv**3
        rinv3 = _mirror4(rr_inv3)
        # Now we determine the normalized rx and ry
        ux = _mirror4(rrx*rr_inv, negativex=True) # THE BUG WAS HERE OMG
        uy = _mirror4(rry*rr_inv, negativey=True) # HOLY FLYING GUACAMOLE
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
                    if self.PBC:
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

                    total_energy += partial_m*signal.convolve2d(kernel, self.m, mode='valid')
        self.E_dipolar = total_energy
        
    def energy_exchange_init(self, J):
        if 'exchange' not in self.energies: self.energies.append('exchange')
        self.Exchange_J = J
        # self.Exchange_interaction is the mask for nearest neighbors
        if self.m_type == 'op': 
            self.Exchange_interaction = cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        elif self.m_type == 'ip':
            # if self.config in ['square', 'pinwheel']:
            #     self.Exchange_interaction = cp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
            # elif self.config in ['kagome', 'triangle']:
            #     self.Exchange_interaction = cp.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])
            self.Exchange_interaction = cp.array([[0]]) # Exchange E doesn't have much meaning for differently oriented spins

    def energy_exchange_update(self):
        self.E_exchange = -self.Exchange_J*cp.multiply(signal.convolve2d(self.m, self.Exchange_interaction, mode='same', boundary='fill'), self.m)
        if self.PBC:
            pass # TODO: implement PBC


    def update(self):
        """ Performs a single magnetization switch. """
        if self.T == 0:
            warnings.warn('Temperature is zero, so no switch will be simulated.', stacklevel=2)
            return # We just warned that no switch will be simulated, so let's keep our word
        self.barrier = (self.E_b - self.E_int)/self.mask # Divide by mask to make non-occupied grid cells have infinite barrier
        minBarrier = cp.min(self.barrier)
        self.barrier -= minBarrier # Energy is relative, so set min(E) to zero (this solves issues at low T)
        with np.errstate(over='ignore'): # Ignore overflow warnings in the exponential: such high barriers wouldn't switch anyway
            taus = cp.random.exponential(cp.exp(self.barrier/self.T))
            indexmin = cp.argmin(taus)
            indexmin2D = cp.unravel_index(indexmin, self.m.shape) # The min(tau) index in 2D form for easy indexing
            self.m[indexmin2D] = -self.m[indexmin2D]
            self.t += taus[indexmin2D]*cp.exp(minBarrier/self.T) # This can become cp.inf quite quickly if T is small
        if self.m_type == 'op':
            self.m_tot = cp.mean(self.m)
        elif self.m_type == 'ip':
            self.m_tot_x = cp.mean(cp.multiply(self.m, self.orientation[:,:,0]))
            self.m_tot_y = cp.mean(cp.multiply(self.m, self.orientation[:,:,1]))
            self.m_tot = (self.m_tot_x**2 + self.m_tot_y**2)**(1/2)
        
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
            self.corr_mask = maskcor[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)] # Lower right quadrant of maskcor because the other quadrants are symmetrical
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


    # Below here are some graphical functions (plot magnetization profile etc.)
    def _get_averaged_extent(self, avg):
        ''' Returns the extent that can be used in imshow when plotting an averaged quantity. '''
        avg = self._resolve_avg(avg)
        mask = self._get_mask(avg=avg)
        movex, movey = mask.shape[1]/2*self.dx, mask.shape[0]/2*self.dy # The averaged imshow should be displaced by this much
        return [self.x_min-self.dx+movex,self.x_max-movex+self.dx,self.y_min-self.dy+movey,self.y_max-movey+self.dy]
        
    def _get_appropriate_avg(self):
        ''' Auto-detect the most appropriate averaging mask based on self.config '''
        if self.config in ['full']:
            avg = 'point'
        elif self.config in ['chess', 'square', 'pinwheel']:
            avg = 'cross'
        elif self.config in ['kagome']:
            avg = 'hexagon'
        elif self.config in ['triangle']:
            avg = 'triangle'
        return avg

    def _resolve_avg(self, avg):
        ''' If avg is str then determine if it is valid, otherwise auto-determine which averaging method is appropriate. '''
        if isinstance(avg, str):
            assert avg in ['point', 'cross', 'square', 'hexagon', 'triangle'], "Unsupported averaging mask: %s" % avg
        else: # It is something which can be truthy or falsy
            avg = self._get_appropriate_avg() if avg else 'point'
        return avg

    def _get_mask(self, avg=None):
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
        elif avg == 'square': # square ⸬
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

    def get_m_angles(self, m=None, avg=True):
        '''
            Returns the magnetization angle (can be averaged using the averaging method specified by <avg>). If the local
            average magnetization is zero, the corresponding angle is NaN, such that those regions are white in imshow.
            @param m [2D array] (self.m): The magnetization profile that should be averaged.
            @param avg [str|bool] (True): can be any of True, False, 'point', 'cross', 'square', 'triangle', 'hexagon':
                True: automatically determines the appropriate averaging method corresponding to self.config.
                False|'point': no averaging at all, just calculates the angle of each individual spin.
                'cross': averages the spins north, east, south and west of each position.
                'square': averages the spins northeast, southeast, southwest and northwest of each position.
                'triangle': averages the three magnets connected to a corner of a hexagon in the kagome geometry.
                'hexagon:' averages each hexagon in 'kagome' config, or each star in 'triangle' config.
            @return [2D np.array]: the (averaged) magnetization angle at each position.
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
        mask = self._get_mask(avg=avg)
        x_comp_avg = signal.convolve2d(x_comp, mask, mode='valid', boundary='fill')
        y_comp_avg = signal.convolve2d(y_comp, mask, mode='valid', boundary='fill')
        angles_avg = cp.arctan2(y_comp_avg, x_comp_avg) % (2*cp.pi)
        useless_angles = cp.where(cp.logical_and(cp.isclose(x_comp_avg, 0), cp.isclose(y_comp_avg, 0)), cp.nan, 1)
        angles_avg *= useless_angles
        if avg == 'triangle':
            angles_avg = angles_avg[1::2,1::2]
        elif avg == 'hexagon':
            angles_avg = angles_avg[::2,::2]
            ix = cp.arange(0, angles_avg.shape[1])
            iy = cp.arange(0, angles_avg.shape[0])
            ixx, iyy = cp.meshgrid(ix, iy) # DO NOT REMOVE THIS, THIS IS NOT THE SAME AS self.ixx, self.iyy!
            angles_avg[(ixx + iyy) % 2 == 1] = cp.nan # These are not the centers of hexagons, so dont draw these
        return angles_avg.get()

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
        show_quiver = self.m.size < 1e5 # Quiver becomes very slow for more than 100k cells, so just dont show it then
        averaged_extent = self._get_averaged_extent(avg)
        full_extent = [self.x_min-self.dx/2,self.x_max+self.dx/2,self.y_min-self.dy/2,self.y_max+self.dx/2]

        num_plots = 2 if show_energy else 1
        axes = []
        if self.m_type == 'op':
            fig = plt.figure(figsize=(3.5*num_plots, 3))
            ax1 = fig.add_subplot(1, num_plots, 1)
            mask = self._get_mask(avg=avg)
            im1 = ax1.imshow(signal.convolve2d(m, mask, mode='valid', boundary='fill').get(),
                             cmap='gray', origin='lower', vmin=-cp.sum(mask), vmax=cp.sum(mask),
                             extent=averaged_extent)
            ax1.set_title(r'Averaged magnetization $\vert m \vert$')
            plt.colorbar(im1)
            axes.append(ax1)
        elif self.m_type == 'ip':
            num_plots += 1 if show_quiver else 0
            fig = plt.figure(figsize=(3.5*num_plots, 3))
            ax1 = fig.add_subplot(1, num_plots, 1)
            im = fill_nan_neighbors(self.get_m_angles(m=m, avg=avg)) if fill else self.get_m_angles(m=m, avg=avg)
            im1 = ax1.imshow(im, cmap='hsv', origin='lower', vmin=0, vmax=2*cp.pi,
                             extent=averaged_extent) # extent doesnt work perfectly with triangle or kagome but is still ok
            ax1.set_title('Averaged magnetization angle' + ('\n("%s" average)' % avg if avg != 'point' else ''), font={"size":"10"})
            plt.colorbar(im1)
            axes.append(ax1)
            if show_quiver:
                ax2 = fig.add_subplot(1, num_plots, 2, sharex=ax1, sharey=ax1)
                ax2.set_aspect('equal')
                nonzero = self.m.get().nonzero()
                quiverscale = 0.9 if self.config in ['kagome'] else 0.7
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
        multi = MultiCursor(fig.canvas, axes, color='black', lw=1, linestyle='dotted', horizOn=True, vertOn=True)
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
    
    def get_AFMness(self, AFM_mask=None):
        ''' Returns the average AFM-ness of self.m at the current time step, normalized to 1.
            For a perfectly uniform configuration this is 0, while for random it is 0.375.
            Note that the boundaries are not taken into account for the normalization, so the
            AFM-ness will often be slightly lower than the ideal values mentioned above.
            @param AFM_mask [2D array] (None): The mask used to determine the AFM-ness. If not
                provided explicitly, it is determined automatically based on self.config.
            @return [float]: The average normalized AFM-ness.
        '''
        if AFM_mask is None:
            if self.config in ['full']:
                AFM_mask = cp.array([[1, -1], [-1, 1]], dtype='float')
            elif self.config in ['chess', 'square', 'pinwheel']:
                AFM_mask = cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')
            elif self.config in ['kagome', 'triangle']:
                AFM_mask = cp.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype='float')
        else:
            AFM_mask = cp.asarray(AFM_mask)
        AFM_ness = cp.mean(cp.abs(signal.convolve2d(self.m, AFM_mask, mode='same', boundary='fill')))
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

def fill_nan_neighbors(arr): # TODO: find a better place and name for this function, maybe create separate graphical helper function file
    ''' THIS FUNCTION ONLY WORKS FOR GRIDS WHICH HAVE A CHESS-LIKE OCCUPATION OF THE CELLS! (cross ⁛)
        Assume an array <arr> has np.nan at every other diagonal (i.e. chess pattern of np.nan's). Then this
        function fills in these NaNs with the surrounding values at its nearest neighbors (cross neighbors ⁛),
        but only if all those neighbors are equal. This is useful for very large simulations where each cell
        occupies less than 1 pixel when plotted: by removing the NaNs, visual issues can be prevented.
        @return [2D np.array]: The interpolated array.
    '''
    if type(arr) == cp.ndarray:
        arr = arr.get() # We need numpy for this, to np.insert the desired rows (cp.insert does not exist)
    else:
        arr = np.asarray(arr)
    replaceable = np.isnan(arr)

    # Extend arrays a bit to fill nans near boundaries as well
    a = np.insert(arr, 0, arr[1], axis=0)
    a = np.insert(a, 0, a[:,1], axis=1)
    a = np.append(a, a[-2].reshape(1,-1), axis=0)
    a = np.append(a, a[:,-2].reshape(-1,1), axis=1)

    N = a[:-2, 1:-1]
    E = a[1:-1, 2:]
    S = a[2:, 1:-1]
    W = a[1:-1, :-2]
    equal_neighbors = np.logical_and(np.logical_and(np.isclose(N, E), np.isclose(E, S)), np.isclose(S, W))

    return np.where(np.logical_and(replaceable, equal_neighbors), N, arr)


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