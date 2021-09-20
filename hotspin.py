import ctypes
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field
from numpy.core.numeric import Inf
from scipy import signal


ctypes.windll.shcore.SetProcessDpiAwareness(2) # (For Windows 10/8/7) this makes the matplotlib plots smooth on high DPI screens


class Magnets:
    def __init__(self, xx, yy, T, E_b, m_type='ip', config='square', pattern='random'): # TODO: allow to specify the energy components here as well
        '''
            The initial configuration of a Magnets geometry consists of 3 parts:
             1) m_type:  Magnets can be in-plane or out-of-plane: 'ip' or 'op', respectively.
             2) config:  The placement of magnets on the grid can be 'full', 'square', 'pinwheel', 'kagome' or 'triangle'.
             3) pattern: The initial magnetization direction (e.g. up/down) can be 'uniform', 'random', 'chess' or 'AFM'.
        '''
        assert np.shape(xx) == np.shape(yy), "Error: xx and yy should have the same shape. Please obtain xx and yy using np.meshgrid(x,y) to avoid this issue."
        self.xx = xx
        self.yy = yy
        self.T = T
        self.t = 0.
        self.E_b = E_b
        self.m_type = m_type

        ix = np.arange(0, self.xx.shape[1])
        iy = np.arange(0, self.yy.shape[0])
        self.ixx, self.iyy = np.meshgrid(ix, iy)

        # config disambiguation
        # TODO: could use an Enum for the config, though this might make for a more difficult user experience
        if config in ['full']:
            assert m_type == 'op', f"Can not use config '{config}' if m_type is not 'op'."
            self.config = 'full'
        elif config in ['square', 'squareASI']:
            self.config = 'square'
        elif config in ['pinwheel', 'pinwheelASI']:
            assert m_type == 'ip', f"Can not use config '{config}' if m_type is not 'ip'."
            self.config = 'pinwheel'
        elif config in ['kagome', 'kagomeASI']:
            assert m_type == 'ip', f"Can not use config '{config}' if m_type is not 'ip'."
            self.config = 'kagome'
        elif config in ['triangle', 'triangleASI']:
            assert m_type == 'ip', f"Can not use config '{config}' if m_type is not 'ip'."
            self.config = 'triangle'
        else:
            raise AssertionError(f"Invalid argument for parameter 'config': '{config}'.")

        # Initialize self.m and the correct self.mask
        self.Initialize_m(pattern)

        # Set the orientation of the islands corresponding to config
        if m_type == 'ip': 
            if self.config == 'square':
                self._Initialize_ip('square', 0)
            elif self.config == 'pinwheel':
                self._Initialize_ip('square', np.pi/4)
            elif self.config == 'kagome':
                self._Initialize_ip('kagome', 0)
            elif self.config == 'triangle':
                self._Initialize_ip('kagome', np.pi/2)

        self.E_int = np.zeros_like(xx)
        self.E_tot = 0
        self.index = range(self.xx.size)
        self.history = History()


    def Initialize_m(self, pattern):
        ''' Pattern can be any of "random", "uniform", "chess", "AFM".
            This function should detect which kind of geometry this ASI is without
            the user specifying it.
        '''
        if pattern == 'uniform':
            self.m = np.ones(np.shape(self.xx)) # For full, square, pinwheel: this is already ok
            if self.config in ['kagome', 'triangle']:
                self.m[(self.ixx - self.iyy) % 4 == 1] = -1
        elif pattern == 'chess':
            assert self.config == 'full', "Can only use 'chess' pattern for 'full' out-of-plane magnetization." # TODO: maybe implement this for in-plane somehow
            self.m = ((self.xx + self.yy) % 2)*2 - 1
        elif pattern == 'AFM':
            if self.config in ['square', 'pinwheel']:
                self.m = ((self.xx - self.yy)//2 % 2)*2 - 1
            elif self.config in ['kagome', 'triangle']:
                self.m = np.ones(np.shape(self.xx))
                self.m[(self.ixx + self.iyy) % 4 == 3] = -1
        else:
            self.m = np.random.randint(0, 2, size=np.shape(self.xx))*2 - 1 # Yields random -1 or 1
            if pattern != 'random': warnings.warn('Config not recognized, defaulting to "random".', stacklevel=2)

        if self.config == 'full':
            self.mask = np.ones_like(self.m)
        elif self.config in ['square', 'pinwheel']:
            self.mask = np.zeros_like(self.m)
            self.mask[(self.xx + self.yy) % 2 == 1] = 1
        elif self.config in ['kagome', 'triangle']:
            self.mask = np.zeros_like(self.m)
            self.mask[(self.ixx + self.iyy) % 4 == 1] = 1 # One bunch of diagonals \
            self.mask[(self.ixx - self.iyy) % 4 == 3] = 1 # Other bunch of diagonals /

        self.m = np.multiply(self.m, self.mask)
        self.m_tot = np.mean(self.m)
            
      
    def _Initialize_ip(self, config, angle=0.):
        ''' This function should only be called by the Magnets() class itself, since
            it initializes the angles of all the spins, which is something the user should not
            adjust themselves.
        '''
        # This sets the angle of all the magnets (this is of course only applicable in the in-plane case)
        assert self.m_type == 'ip', "Error: can not Initialize_ip() if m_type != 'ip'."
        self.orientation = np.zeros(np.shape(self.m) + (2,))
        if config == 'square':
            self.orientation[self.yy % 2 == 0,0] = np.cos(angle)
            self.orientation[self.yy % 2 == 0,1] = np.sin(angle)
            self.orientation[self.yy % 2 == 1,0] = np.cos(angle + np.pi/2)
            self.orientation[self.yy % 2 == 1,1] = np.sin(angle + np.pi/2)
            self.orientation[self.mask == 0,0] = 0
            self.orientation[self.mask == 0,1] = 0
        elif config == 'kagome':
            self.orientation[:,:,0] = np.cos(angle + np.pi/2)
            self.orientation[:,:,1] = np.sin(angle + np.pi/2)
            self.orientation[np.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1),0] = np.cos(angle - np.pi/6)
            self.orientation[np.logical_and((self.ixx - self.iyy) % 4 == 1, self.ixx % 2 == 1),1] = np.sin(angle - np.pi/6)
            self.orientation[np.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1),0] = np.cos(angle + np.pi/6)
            self.orientation[np.logical_and((self.ixx + self.iyy) % 4 == 3, self.ixx % 2 == 1),1] = np.sin(angle + np.pi/6)
            self.orientation[self.mask == 0,0] = 0
            self.orientation[self.mask == 0,1] = 0
    
    def Energy(self):
        E = np.zeros_like(self.xx)
        if hasattr(self, 'E_exchange'):
            self.Exchange_update()
            E = E + self.E_exchange
        if hasattr(self, 'E_dipolar'):
            self.Dipolar_energy_update()
            E = E + self.E_dipolar
        if hasattr(self, 'E_Zeeman'):
            self.Zeeman_update()
            E = E + self.E_Zeeman
        self.E_int = E
        self.E_tot = np.sum(E, axis=None)
        return self.E_tot  

    def Zeeman_init(self):
        self.E_Zeeman = np.empty_like(self.xx)
        if self.m_type == 'op':
            self.H_ext = 0.
        elif self.m_type == 'ip':
            self.H_ext = np.zeros(2)
        self.Zeeman_update()

    def Zeeman_update(self):
        if self.m_type == 'op':
            self.E_Zeeman = -self.m*self.H_ext
        elif self.m_type == 'ip':
            self.E_Zeeman = -np.multiply(self.m, self.H_ext[0]*self.orientation[:,:,0] + self.H_ext[1]*self.orientation[:,:,1])
            
    def Dipolar_energy_init(self, strength=1):
        self.Dipolar_interaction = np.empty((self.xx.size, self.xx.size))
        self.E_dipolar = np.empty_like(self.xx)
        mx2 = np.reshape(self.orientation[:,:,0], -1) # Vector: the mx of all the other magnets
        my2 = np.reshape(self.orientation[:,:,1], -1) # Vector: the my of all the other magnets
        for i in self.index:
            rrx = np.reshape(self.xx.flat[i] - self.xx, -1)
            rry = np.reshape(self.yy.flat[i] - self.yy, -1)
            rr_sq = rrx**2 + rry**2
            np.place(rr_sq, rr_sq == 0.0, Inf)
            rr_inv = rr_sq**(-1/2)
            rrx_u = rrx*rr_inv
            rry_u = rry*rr_inv
            self.Dipolar_interaction[i] = rr_inv**3 # Distance^-3 of magnet <i> to every other magnet
            if self.m_type == 'ip':
                mx1 = mx2[i] # Scalar: the mx of this magnet
                my1 = my2[i] # Scalar: the my of this magnet
                self.Dipolar_interaction[i] = mx1*mx2 + my1*my2
                self.Dipolar_interaction[i] -= 3*(mx1*rrx_u + my1*rry_u)*(np.multiply(mx2, rrx_u) + np.multiply(my2, rry_u))
                self.Dipolar_interaction[i] = np.multiply(self.Dipolar_interaction[i], rr_inv**3)
        np.place(self.Dipolar_interaction, self.Dipolar_interaction == Inf, 0.0) # Magnet does not interact with itself
        self.Dipolar_interaction *= strength
        self.Dipolar_energy_update()

    def Dipolar_energy_update(self):
        # All we still need to do is multiply self.Dipolar_interaction by the correct current values of m1*m2.
        temp = np.dot(self.Dipolar_interaction, np.reshape(self.m, self.m.size)) # This adds the columns of self.Dipolar_interaction together with weights self.m (i.e. m2)
        self.E_dipolar = np.multiply(self.m, np.reshape(temp, self.xx.shape)) # This multiplies each row (which is now only 1 element long due to the sum from the previous line of code) with m1

    def Exchange_init(self, J):
        if self.m_type == 'op': # Mask for nearest neighbors
            self.Exchange_interaction = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        elif self.m_type == 'ip': # TODO: this only works for square and pinwheel ASI, maybe include other geometries too
            self.Exchange_interaction = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        self.Exchange_J = J
        self.Exchange_update()

    def Exchange_update(self):
        self.E_exchange = -self.Exchange_J*np.multiply(signal.convolve2d(self.m, self.Exchange_interaction, mode='same', boundary='fill'), self.m)

    def Update(self):
        """ Performs a single magnetization switch. """
        self.Energy()
        self.barrier = self.E_b - self.E_int
        self.rate = np.exp(self.barrier/self.T) # TODO: this can throw a divide by zero warning
        taus = np.random.exponential(scale=self.rate) # TODO: this can throw an overflow warning
        indexmin = np.argmin(taus, axis=None)
        self.m.flat[indexmin] = -self.m.flat[indexmin]
        self.t += taus.flat[indexmin]
        if self.m_type == 'op':
            self.m_tot = np.mean(self.m)
        elif self.m_type == 'ip':
            self.m_tot_x = np.mean(np.multiply(self.m, self.orientation[:,:,0]))
            self.m_tot_y = np.mean(np.multiply(self.m, self.orientation[:,:,1]))
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
        indexmax = np.argmax(self.E_int, axis=None)
        self.m.flat[indexmax] = -self.m.flat[indexmax]
    
    def Save_history(self, *, E_tot=None, t=None, T=None, m_tot=None):
        """ Records E_tot, t, T and m_tot as they were last calculated. This default behavior can be overruled: if
            passed as keyword parameters, their arguments will be saved instead of the self.<E_tot|t|T|m_tot> value(s).
        """
        self.history.E.append(self.E_tot if E_tot is None else E_tot)
        self.history.t.append(self.t if t is None else t)
        self.history.T.append(self.T if T is None else T)
        self.history.m.append(self.m_tot if m_tot is None else m_tot)
    
    def Clear_history(self):
        self.history.clear()
    
    def Autocorrelation_fast(self, max_distance):
        max_distance = round(max_distance)
        s = np.shape(self.xx)
        if not(hasattr(self, 'Distances')):
            # First calculate the distance between all spins in the simulation.
            self.Distances = (self.xx**2 + self.yy**2)**(1/2)
            self.Distance_range = math.ceil(np.max(self.Distances))
            # Then, calculate how many multiplications hide behind each cell in the convolution matrix, so we can normalize.
            self.corr_norm = 1/signal.convolve2d(np.ones_like(self.m), np.ones_like(self.m), mode='full', boundary='fill')
            # Then, calculate the correlation of the mask, since not each position contains a spin
            maskcor = signal.convolve2d(self.mask, np.flipud(np.fliplr(self.mask)), mode='full', boundary='fill')*self.corr_norm
            self.corr_mask = maskcor[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)] # Lower right quadrant of maskcor because the other quadrants are symmetrical
            self.corr_mask[self.corr_mask > 0] = 1
        # Now, convolve self.m with its point-mirrored/180°-rotated counterpart
        if self.m_type == 'op':
            corr = signal.convolve2d(self.m, np.flipud(np.fliplr(self.m)), mode='full', boundary='fill')*self.corr_norm
        elif self.m_type == 'ip':
            corr_x = signal.convolve2d(self.m*self.orientation[:,:,0], np.flipud(np.fliplr(self.m*self.orientation[:,:,0])), mode='full', boundary='fill')*self.corr_norm
            corr_y = signal.convolve2d(self.m*self.orientation[:,:,1], np.flipud(np.fliplr(self.m*self.orientation[:,:,1])), mode='full', boundary='fill')*self.corr_norm
            corr = corr_x + corr_y
        corr = corr*np.size(self.m)/np.sum(self.corr_mask) # Put between 0 and 1
        self.correlation = corr[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)]**2
        
        # Prepare distance bins etc.
        corr_binned = np.zeros(max_distance + 1) # How much the magnets correlate over a distance [i]
        counts = np.zeros(max_distance + 1)
        distances = np.linspace(0, max_distance, num=max_distance+1) # Use np.linspace to get float, np.arange to get int
        # Now loop over all the spins, and record their correlation and counts
        for i in self.index:
            distbin = math.floor(self.Distances.flat[i])
            if distbin <= max_distance:
                corr_binned[distbin] += self.correlation.flat[i]*self.corr_mask.flat[i]
                counts[distbin] += self.corr_mask.flat[i]
        corr_binned = np.divide(corr_binned, counts) # TODO: this can throw a divide by zero warning
        corr_length = np.sum(np.multiply(abs(corr_binned), distances))
        return corr_binned, distances, corr_length
    

    # Below here are some graphical functions (plot magnetization profile etc.)
    def Get_magAngles(self, m=None, avg='point'):
        '''
            Returns the magnetization angle (can be averaged using the averaging method specified by <avg>). If the local
            average magnetization is zero, the corresponding angle is NaN, such that those regions are white in imshow.
            @param m [2D array] (self.m): The magnetization profile that should be averaged.
            @param avg [str] ('point'): can be any of 'point', 'cross', 'square', 'triangle', 'hexagon'. These are:
                'point' does no averaging at all, so just calculates the angle of each individual spin.
                'cross' averages the spins north, east, south and west of each position.
                'square' averages the spins northeast, southeast, southwest and northwest of each position.
                'triangle' averages the three magnets connected to a corner of a hexagon in the kagome geometry.
                'hexagon' averages in a hexagon around each position, though this is not very clear.
            @return [2D array]: the (averaged) magnetization angle at each position. 
                !! This does not necessarily have the same shape as <m> !!
        '''
        assert avg in ['point', 'cross', 'square', 'triangle', 'hexagon'], "Unsupported averaging mask: %s" % avg
        if m is None: m = self.m

        x_comp = np.multiply(m, self.orientation[:,:,0])
        y_comp = np.multiply(m, self.orientation[:,:,1])
        if avg == 'point':
            mask = [[1]]
        elif avg == 'cross':
            mask = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        elif avg == 'square':
            mask = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
        elif avg == 'hexagon': # TODO: instead of a normal averaging mask, can also make a 'clockwiseness' graph
            mask = [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0]]
        elif avg == 'triangle':
            mask = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        x_comp_avg = signal.convolve2d(x_comp, mask, mode='valid', boundary='fill')
        y_comp_avg = signal.convolve2d(y_comp, mask, mode='valid', boundary='fill')
        if avg == 'triangle':
            x_comp_avg = signal.convolve2d(x_comp, mask, mode='same', boundary='fill')
            y_comp_avg = signal.convolve2d(y_comp, mask, mode='same', boundary='fill')
        angles_avg = np.arctan2(y_comp_avg, x_comp_avg) % (2*np.pi)
        useless_angles = np.where(np.logical_and(np.isclose(x_comp_avg, 0), np.isclose(y_comp_avg, 0)), np.nan, 1)
        angles_avg *= useless_angles
        if avg == 'triangle':
            angles_avg = angles_avg[::2,::2]
        elif avg == 'hexagon':
            angles_avg = angles_avg[::2,::2]
            ix = np.arange(0, angles_avg.shape[1])
            iy = np.arange(0, angles_avg.shape[0])
            ixx, iyy = np.meshgrid(ix, iy) # DO NOT REMOVE THIS, THIS IS NOT THE SAME AS self.ixx, self.iyy!
            angles_avg[(ixx + iyy) % 2 == 1] = np.nan # These are not the centers of hexagons, so dont draw these
        return angles_avg

    def Show_m(self, m=None, avg='point', show_energy=True):
        ''' Shows two (or three if <show_energy> is True) figures displaying the direction of each spin: one showing
            the (locally averaged) angles, another quiver plot showing the actual vectors. If <show_energy> is True,
            a third and similar plot, displaying the interaction energy of each spin, is also shown.
            @param m [2D array] (self.m): the direction (+1 or -1) of each spin on the geometry. Default is the current
                magnetization profile. This is useful if some magnetization profiles have been saved manually, while 
                self.Update() has been called since: one can then pass these saved profiles as the <m> parameter to
                draw them onto the geometry stored in <self>.
            @param average [str] ('point'): any of 'point', 'cross', 'square', 'triangle' and 'hexagon'. This adds together 
                the nearest neighbors according to the given shape. This is useful to see the boundaries between 
                antiferromagnetic domains. One can also just pass a bool: True -> 'cross', False -> 'point'.
            @param show_energy [bool] (True): if True, a 2D plot of the energy is shown in the figure as well.
        '''
        if m is None: m = self.m
        
        if not isinstance(avg, str): # TODO: can detect which type of ASI <self> is, but currently that information is not stored in a Magnets() object
            if not avg: # If average is falsey
                avg = 'point'
            else: # If average is truthy
                avg = 'cross'
            
        if self.m_type == 'op':
            num_plots = 2 if show_energy else 1
            fig = plt.figure(figsize=(3.5*num_plots, 3))
            ax1 = fig.add_subplot(1, num_plots, 1)
            im1 = ax1.imshow(self.m, cmap='gray', origin='lower')
            ax1.set_title(r'$m$')
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
            im1 = ax1.imshow(self.Get_magAngles(m=m, avg=avg), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
            ax1.set_title('Averaged magnetization angle' + ('\n("%s" average)' % avg if avg != 'point' else ''), font={"size":"10"})
            plt.colorbar(im1)
            ax2 = fig.add_subplot(1, num_plots, 2)
            ax2.set_aspect('equal')
            ax2.quiver(self.xx, self.yy, np.multiply(m, self.orientation[:,:,0]), np.multiply(m, self.orientation[:,:,1]), pivot='mid', headlength=17, headaxislength=17, headwidth=7)
            ax2.set_title(r'$m$')
            if show_energy:
                ax3 = fig.add_subplot(1, num_plots, 3)
                im3 = ax3.imshow(self.E_int, origin='lower')
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
        if np.all(np.isclose(self.history.T, self.history.T[0])):
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
