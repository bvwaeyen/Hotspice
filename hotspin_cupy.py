import ctypes
import math
import matplotlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from dataclasses import dataclass, field
from numpy.core.numeric import Inf
from cupyx.scipy import signal


ctypes.windll.shcore.SetProcessDpiAwareness(2) # (For Windows 10/8/7) this makes the matplotlib plots smooth on high DPI screens
matplotlib.rcParams["image.interpolation"] = 'none' # 'none' works best for large images scaled down, 'nearest' for the opposite

class Magnets:
    def __init__(self, xx, yy, T, E_b, m_type='ip', config='square', pattern='random', energies=('dipolar')):
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
        self.T = T
        self.t = 0.
        self.E_b = E_b
        self.m_type = m_type

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

        # Initialize self.m and the correct self.mask
        self.Initialize_m(pattern)

        # Set the orientation of the islands corresponding to config
        if m_type == 'ip': 
            if self.config == 'square':
                self._Initialize_ip('square', 0)
            elif self.config == 'pinwheel':
                self._Initialize_ip('square', cp.pi/4)
            elif self.config == 'kagome':
                self._Initialize_ip('kagome', 0)
            elif self.config == 'triangle':
                self._Initialize_ip('kagome', cp.pi/2)

        # Initialize the specified energy components
        if 'dipolar' in energies:
            self.Dipolar_energy_init()
        if 'Zeeman' in energies:
            self.Zeeman_energy_init()
        if 'exchange' in energies:
            self.Exchange_energy_init()
        self.Energy()

        self.history = History()


    def Initialize_m(self, pattern):
        ''' Pattern can be any of "random", "uniform", "chess", "AFM".
            This function should detect which kind of geometry this ASI is without
            the user specifying it.
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

        if self.config == 'full':
            self.mask = cp.ones_like(self.m)
        elif self.config in ['chess', 'square', 'pinwheel']:
            self.mask = cp.zeros_like(self.m)
            self.mask[(self.xx + self.yy) % 2 == 1] = 1
        elif self.config in ['kagome', 'triangle']:
            self.mask = cp.zeros_like(self.m)
            self.mask[(self.ixx + self.iyy) % 4 == 1] = 1 # One bunch of diagonals \
            self.mask[(self.ixx - self.iyy) % 4 == 3] = 1 # Other bunch of diagonals /

        self.m = cp.multiply(self.m, self.mask)
        self.m_tot = cp.mean(self.m)
            
      
    def _Initialize_ip(self, config, angle=0.):
        ''' This function should only be called by the Magnets() class itself, since
            it initializes the angles of all the spins, which is something the user should not
            adjust themselves.
        '''
        # This sets the angle of all the magnets (this is of course only applicable in the in-plane case)
        assert self.m_type == 'ip', "Can not _Initialize_ip() if m_type != 'ip'."
        self.orientation = np.zeros(np.shape(self.m) + (2,)) # Keep this a numpy array for now since boolean indexing is broken in cupy
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
            self.orientation[np.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1),0] = math.cos(angle - math.pi/6)
            self.orientation[np.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1),1] = math.sin(angle - math.pi/6)
            self.orientation[np.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1),0] = math.cos(angle + math.pi/6)
            self.orientation[np.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1),1] = math.sin(angle + math.pi/6)
            self.orientation[mask == 0,0] = 0
            self.orientation[mask == 0,1] = 0
        self.orientation = cp.asarray(self.orientation)
    
    def Energy(self):
        E = cp.zeros_like(self.xx)
        if hasattr(self, 'E_exchange'):
            self.Exchange_energy_update()
            E = E + self.E_exchange
        if hasattr(self, 'E_dipolar'):
            self.Dipolar_energy_update()
            E = E + self.E_dipolar
        if hasattr(self, 'E_Zeeman'):
            self.Zeeman_energy_update()
            E = E + self.E_Zeeman
        self.E_int = E
        self.E_tot = cp.sum(E, axis=None)
        return self.E_tot  

    def Zeeman_energy_init(self):
        self.E_Zeeman = cp.empty_like(self.xx)
        if self.m_type == 'op':
            self.H_ext = 0.
        elif self.m_type == 'ip':
            self.H_ext = cp.zeros(2)
        self.Zeeman_energy_update()

    def Zeeman_energy_update(self):
        if self.m_type == 'op':
            self.E_Zeeman = -self.m*self.H_ext
        elif self.m_type == 'ip':
            self.E_Zeeman = -cp.multiply(self.m, self.H_ext[0]*self.orientation[:,:,0] + self.H_ext[1]*self.orientation[:,:,1])
            
    def Dipolar_energy_init(self, strength=1): # TODO: shrink this demag kernel
        self.Dipolar_interaction = cp.empty((self.xx.size, self.xx.size))
        self.E_dipolar = cp.empty_like(self.xx)
        for i in self.index:
            rrx = cp.reshape(self.xx.flat[i] - self.xx, -1)
            rry = cp.reshape(self.yy.flat[i] - self.yy, -1)
            rr_sq = rrx**2 + rry**2
            cp.place(rr_sq, rr_sq == 0.0, Inf)
            rr_inv = rr_sq**(-1/2)
            rrx_u = rrx*rr_inv
            rry_u = rry*rr_inv
            self.Dipolar_interaction[i] = rr_inv**3 # Distance^-3 of magnet <i> to every other magnet
            if self.m_type == 'ip':
                mx2 = cp.reshape(self.orientation[:,:,0], -1) # Vector: the mx of all the other magnets
                my2 = cp.reshape(self.orientation[:,:,1], -1) # Vector: the my of all the other magnets
                mx1 = mx2[i] # Scalar: the mx of this magnet
                my1 = my2[i] # Scalar: the my of this magnet
                self.Dipolar_interaction[i] = mx1*mx2 + my1*my2
                self.Dipolar_interaction[i] -= 3*(mx1*rrx_u + my1*rry_u)*(cp.multiply(mx2, rrx_u) + cp.multiply(my2, rry_u))
                self.Dipolar_interaction[i] = cp.multiply(self.Dipolar_interaction[i], rr_inv**3)
        cp.place(self.Dipolar_interaction, self.Dipolar_interaction == Inf, 0.0) # Magnet does not interact with itself
        self.Dipolar_interaction *= strength
        self.Dipolar_energy_update()

    def Dipolar_energy_update(self):
        # All we still need to do is multiply self.Dipolar_interaction by the correct current values of m1*m2.
        temp = cp.dot(self.Dipolar_interaction, cp.reshape(self.m, self.m.size)) # This adds the columns of self.Dipolar_interaction together with weights self.m (i.e. m2)
        self.E_dipolar = cp.multiply(self.m, cp.reshape(temp, self.xx.shape)) # This multiplies each row (which is now only 1 element long due to the sum from the previous line of code) with m1

    def Exchange_energy_init(self, J):
        # self.Exchange_interaction is the mask for nearest neighbors
        if self.m_type == 'op': 
            self.Exchange_interaction = cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        elif self.m_type == 'ip':
            if self.config in ['square', 'pinwheel']:
                self.Exchange_interaction = cp.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
            elif self.config in ['kagome', 'triangle']:
                self.Exchange_interaction = cp.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])

        self.Exchange_J = J
        self.Exchange_energy_update()

    def Exchange_energy_update(self):
        self.E_exchange = -self.Exchange_J*cp.multiply(signal.convolve2d(self.m, self.Exchange_interaction, mode='same', boundary='fill'), self.m)

    def Update(self):
        """ Performs a single magnetization switch. """
        if self.T == 0:
            warnings.warn('Temperature is zero, so no switch will be simulated.', stacklevel=2)
            return # We just warned that no switch will be simulated, so let's keep our word
        self.Energy()
        self.barrier = (self.E_b - self.E_int)/self.mask # Divide by mask to make non-occupied grid cells have infinite barrier
        minBarrier = cp.min(self.barrier)
        self.barrier -= minBarrier # Energy is relative, so set min(E) to zero (this solves issues at low T)
        with np.errstate(over='ignore'): # Ignore overflow warnings in the exponential: such high barriers wouldn't switch anyway
            taus = cp.random.exponential(cp.exp(self.barrier/self.T))
            indexmin = cp.unravel_index(cp.argmin(taus), self.m.shape) # WARN: this might not be optimal
            self.m[indexmin] = -self.m[indexmin]
            self.t += taus[indexmin]*cp.exp(minBarrier/self.T) # This can become cp.inf quite quickly if T is small
        if self.m_type == 'op':
            self.m_tot = cp.mean(self.m)
        elif self.m_type == 'ip':
            self.m_tot_x = cp.mean(cp.multiply(self.m, self.orientation[:,:,0]))
            self.m_tot_y = cp.mean(cp.multiply(self.m, self.orientation[:,:,1]))
            self.m_tot = (self.m_tot_x**2 + self.m_tot_y**2)**(1/2)
    
    def Run(self, N=1, save_history=1, T=None):
        ''' Perform <N> self.Update() steps, which defaults to only 1 step.
            @param N [int] (1): the number of update steps to run.
            @param save_history [int] (1): the number of steps between two recorded 
                entries in self.history. If 0, the history is not recorded.
            @param T [float] (self.T): the temperature at which to run the <N> steps.
                If not specified, the current temperature is kept.
        '''
        if T is not None:
            self.T = T
        for i in range(int(N)):
            self.Update()
            if save_history:
                if i % save_history == 0:
                    self.Save_history()

    def Minimize(self):
        self.Energy()
        indexmax = cp.argmax(self.E_int, axis=None)
        self.m.flat[indexmax] = -self.m.flat[indexmax]
    
    def Save_history(self, *, E_tot=None, t=None, T=None, m_tot=None):
        """ Records E_tot, t, T and m_tot as they were last calculated. This default behavior can be overruled: if
            passed as keyword parameters, their arguments will be saved instead of the self.<E_tot|t|T|m_tot> value(s).
        """
        self.history.E.append(float(self.E_tot) if E_tot is None else float(E_tot))
        self.history.t.append(float(self.t) if t is None else float(t))
        self.history.T.append(float(self.T) if T is None else float(T))
        self.history.m.append(float(self.m_tot) if m_tot is None else float(m_tot))
    
    def Clear_history(self):
        self.history.clear()
    
    def Autocorrelation_fast(self, max_distance):
        max_distance = round(max_distance)
        s = cp.shape(self.xx)
        if not(hasattr(self, 'Distances')):
            # First calculate the distance between all spins in the simulation.
            self.Distances = (self.xx**2 + self.yy**2)**(1/2)
            self.Distance_range = math.ceil(cp.max(self.Distances))
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
        
        # Prepare distance bins etc.
        corr_binned = cp.zeros(max_distance + 1) # How much the magnets correlate over a distance [i]
        counts = cp.zeros(max_distance + 1)
        distances = cp.linspace(0, max_distance, num=max_distance+1) # Use cp.linspace to get float, cp.arange to get int
        # Now loop over all the spins, and record their correlation and counts
        for i in self.index:
            distbin = math.floor(self.Distances.flat[i])
            if distbin <= max_distance:
                corr_binned[distbin] += self.correlation.flat[i]*self.corr_mask.flat[i]
                counts[distbin] += self.corr_mask.flat[i]
        corr_binned = cp.divide(corr_binned, counts, where=(counts!=0), out=counts) # Where counts==0, just output '0' to prevent divide by 0
        corr_length = cp.sum(cp.multiply(abs(corr_binned), distances))
        return corr_binned, distances, corr_length

    # Below here are some graphical functions (plot magnetization profile etc.)
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
            any removal of excess zero-rows etc., that is the task of self.Get_magAngles. '''
        avg = self._resolve_avg(avg)
        if avg == 'point':
            mask = [[1]]
        elif avg == 'cross': # cross ⁛
            mask = [[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]]
        elif avg == 'square': # square ⸬
            mask = [[1, 0, 1], 
                    [0, 0, 0], 
                    [1, 0, 1]]
        elif avg == 'hexagon':
            mask = [[0, 1, 0, 1, 0], 
                    [1, 0, 0, 0, 1], 
                    [0, 1, 0, 1, 0]]
        elif avg == 'triangle':
            mask = [[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]]
        return cp.array(mask, dtype='float') # If mask would be int, then precision of convolve2d is also int instead of float

    def Get_magAngles(self, m=None, avg=True):
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
            @return [2D array]: the (averaged) magnetization angle at each position.
                !! This does not necessarily have the same shape as <m> !!
        '''
        assert self.m_type == 'ip', "Can not Get_magAngles of an out-of-plane spin ice (m_type='op')."
        if m is None: m = self.m
        avg = self._resolve_avg(avg)

        x_comp = cp.multiply(m, self.orientation[:,:,0])
        y_comp = cp.multiply(m, self.orientation[:,:,1])
        mask = self._get_mask(avg=avg)
        x_comp_avg = signal.convolve2d(x_comp, mask, mode='valid', boundary='fill')
        y_comp_avg = signal.convolve2d(y_comp, mask, mode='valid', boundary='fill')
        if avg == 'triangle':
            x_comp_avg = signal.convolve2d(x_comp, mask, mode='same', boundary='fill')
            y_comp_avg = signal.convolve2d(y_comp, mask, mode='same', boundary='fill')
        angles_avg = cp.arctan2(y_comp_avg, x_comp_avg) % (2*cp.pi)
        useless_angles = cp.where(cp.logical_and(cp.isclose(x_comp_avg, 0), cp.isclose(y_comp_avg, 0)), cp.nan, 1)
        angles_avg *= useless_angles
        if avg == 'triangle':
            angles_avg = angles_avg[::2,::2]
        elif avg == 'hexagon':
            angles_avg = angles_avg[::2,::2]
            ix = cp.arange(0, angles_avg.shape[1])
            iy = cp.arange(0, angles_avg.shape[0])
            ixx, iyy = cp.meshgrid(ix, iy) # DO NOT REMOVE THIS, THIS IS NOT THE SAME AS self.ixx, self.iyy!
            angles_avg[(ixx + iyy) % 2 == 1] = cp.nan # These are not the centers of hexagons, so dont draw these
        return angles_avg.get()

    def Show_m(self, m=None, avg=True, show_energy=True):
        ''' Shows two (or three if <show_energy> is True) figures displaying the direction of each spin: one showing
            the (locally averaged) angles, another quiver plot showing the actual vectors. If <show_energy> is True,
            a third and similar plot, displaying the interaction energy of each spin, is also shown.
            @param m [2D array] (self.m): the direction (+1 or -1) of each spin on the geometry. Default is the current
                magnetization profile. This is useful if some magnetization profiles have been saved manually, while 
                self.Update() has been called since: one can then pass these saved profiles as the <m> parameter to
                draw them onto the geometry stored in <self>.
            @param avg [str|bool] (True): can be any of True, False, 'point', 'cross', 'square', 'triangle', 'hexagon'.
            @param show_energy [bool] (True): if True, a 2D plot of the energy is shown in the figure as well.
        '''  # TODO: add possibility to fill up all the NaNs with the nearest color if all nearest colors are equal (or something like that)
        if m is None: m = self.m
        avg = self._resolve_avg(avg)
            
        if self.m_type == 'op':
            num_plots = 2 if show_energy else 1
            fig = plt.figure(figsize=(3.5*num_plots, 3))
            ax1 = fig.add_subplot(1, num_plots, 1)
            mask = self._get_mask(avg=avg)
            im1 = ax1.imshow(signal.convolve2d(m, mask, mode='valid', boundary='fill'),
                             cmap='gray', origin='lower', vmin=-cp.sum(mask), vmax=cp.sum(mask))
            ax1.set_title(r'Averaged magnetization $\vert m \vert$')
            plt.colorbar(im1)
            if show_energy:
                ax2 = fig.add_subplot(1, num_plots, 2)
                im2 = ax2.imshow(self.E_int, origin='lower')
                ax2.set_title(r'$E_{int}$')
                plt.colorbar(im2)
        elif self.m_type == 'ip':
            num_plots = 3 if show_energy else 2
            fig = plt.figure(figsize=(3.5*num_plots, 3))
            ax1 = fig.add_subplot(1, num_plots, 1)
            im1 = ax1.imshow(self.Get_magAngles(m=m, avg=avg),
                             cmap='hsv', origin='lower', vmin=0, vmax=2*cp.pi)
            ax1.set_title('Averaged magnetization angle' + ('\n("%s" average)' % avg if avg != 'point' else ''), font={"size":"10"})
            plt.colorbar(im1)
            ax2 = fig.add_subplot(1, num_plots, 2)
            ax2.set_aspect('equal')
            ax2.quiver(self.xx.get(), self.yy.get(), cp.multiply(m, self.orientation[:,:,0]).get(), cp.multiply(m, self.orientation[:,:,1]).get(),
                       pivot='mid', headlength=17, headaxislength=17, headwidth=7, units='xy') # units='xy' makes arrows scale correctly when zooming
            ax2.set_title(r'$m$')
            if show_energy:
                ax3 = fig.add_subplot(1, num_plots, 3)
                im3 = ax3.imshow(self.E_int.get(), origin='lower')
                plt.colorbar(im3)
                ax3.set_title(r'$E_{int}$')
        plt.gcf().tight_layout()
        plt.show()

    def Show_history(self, y_quantity=None, y_label=r'Average magnetization'):
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
        assert len(y_quantity) == len(x_quantity), "Error in Show_history: <y_quantity> has different length than %s history." % x_label.split(' ')[0].lower()

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
