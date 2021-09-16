import math

import numpy as np

from dataclasses import dataclass, field
from numpy.core.numeric import Inf
from scipy import signal


class Magnets:
    def __init__(self, xx, yy, T, E_b, config='full', m_type='op'):
        '''
            The initial configuration of a Magnets geometry consists of 3 parts:
             1) Magnets can be in-plane or out-of-plane: m_type='ip' or m_type='op', respectively.
             2) Each spot on the grid can be occupied by a magnet (config='full'), or only a chess-like
                pattern can be occupied on the grid (config='square').
             3) The initial magnetization direction (e.g. up/down) can be 'uniform', 'random', or 'chess'. 
        '''
        assert np.shape(xx) == np.shape(yy), "Error: xx and yy should have the same shape. Please obtain xx and yy using np.meshgrid(x,y) to avoid this issue."
        self.xx = xx
        self.yy = yy
        self.T = T
        self.t = 0.
        self.E_b = E_b
        self.m_type = m_type
        if m_type == 'op': # Out of plane
            if config == 'full':
                self.Initialize_m('random')
            elif config == 'square':
                self.Initialize_m_square('random')
            else:
                print('Bad config')
        elif m_type == 'ip': # In plane
            if config == 'square':
                self.Initialize_m_square('random')
                self.Initialize_ip('square')
            elif config == 'kagome':
                self.Initialize_m_kagome('random')
                self.Initialize_ip('kagome')
            else:
                print('Bad config')
        else:
            print('Bad m_type')
        self.m_tot = np.mean(self.m)
        self.E_int = np.zeros_like(xx)
        self.index = range(self.xx.size)
        self.history = History()

    def Initialize_m(self, config):
        if config == 'uniform':
            self.m = np.ones(np.shape(self.xx))
        elif config == 'random':
            self.m = np.random.randint(0, 2, size=np.shape(self.xx))*2 - 1 # Yields random -1 or 1
        elif config == 'chess':
            self.m = ((self.xx + self.yy) % 2)*2 - 1
        else:
            self.m = np.random.randint(0, 2, size=np.shape(self.xx))*2 - 1
        self.m_tot = np.mean(self.m)
        self.mask = np.ones_like(self.m) # Necessary if you would need the 'mask' later on

    def Initialize_m_square(self, config):
        # Same as Initialize_m, but half of the magnets are removed to get a chess-like pattern
        if config == 'uniform':
            self.m = np.ones(np.shape(self.xx))
        elif config == 'random':
            self.m = np.random.randint(0, 2, size=np.shape(self.xx))*2 - 1
        elif config == 'chess': # TODO: Is this not exactly the same as 'uniform', since we will apply a mask at the end of this function anyway?
            self.m = ((self.xx + self.yy) % 2)*2 - 1
        elif config == 'AFM_squareASI':
            self.m = ((self.xx - self.yy)//2 % 2)*2 - 1
        else:
            self.m = np.random.randint(0, 2, size=np.shape(self.xx))*2 - 1
        self.mask = np.zeros_like(self.m)
        self.mask[(self.xx + self.yy) % 2 == 1] = 1
        self.m = np.multiply(self.m, self.mask)
        self.m_tot = np.mean(self.m)
        
    def Initialize_m_kagome(self, config):
        ix = np.arange(0, self.xx.shape[1])
        iy = np.arange(0, self.yy.shape[0])
        ixx, iyy = np.meshgrid(ix, iy)
        
        # Same as Initialize_m, but a lot of the magnets are removed to get a kagome-like pattern
        if config == 'uniform':
            self.m = np.ones(np.shape(self.xx))
            self.m[(ixx - iyy) %4 == 1] = -1
        elif config == 'random':
            self.m = np.random.randint(0, 2, size=np.shape(self.xx))*2 - 1 
        else:
            self.m = np.random.randint(0, 2, size=np.shape(self.xx))*2 - 1
            
        self.mask = np.zeros_like(self.m)
        self.mask[(ixx + iyy) % 4 == 1] = 1 # One bunch of diagonals \
        self.mask[(ixx - iyy) % 4 == 3] = 1 # Other bunch of diagonals /
        self.m = np.multiply(self.m, self.mask)
        self.m_tot = np.mean(self.m)
      
    def Initialize_ip(self, config, angle=0.):
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
            ix = np.arange(0, self.xx.shape[1])
            iy = np.arange(0, self.yy.shape[0])
            ixx, iyy = np.meshgrid(ix, iy)
            self.orientation[:,:,0] = np.cos(np.pi/2)
            self.orientation[:,:,1] = np.sin(np.pi/2)
            self.orientation[np.logical_and((ixx - iyy) % 4 == 1, ixx % 2 == 1),0] = np.cos(-np.pi/6)
            self.orientation[np.logical_and((ixx - iyy) % 4 == 1, ixx % 2 == 1),1] = np.sin(-np.pi/6)
            self.orientation[np.logical_and((ixx + iyy) % 4 == 3, ixx % 2 == 1),0] = np.cos(np.pi/6)
            self.orientation[np.logical_and((ixx + iyy) % 4 == 3, ixx % 2 == 1),1] = np.sin(np.pi/6)
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
            rr_inv = (rrx**2 + rry**2)**(-1/2)
            np.place(rr_inv, rr_inv == Inf, 0.0)
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
        elif self.m_type == 'ip':
            self.Exchange_interaction = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        self.Exchange_J = J
        self.Exchange_update()

    def Exchange_update(self):
        self.E_exchange = -self.Exchange_J*np.multiply(signal.convolve2d(self.m, self.Exchange_interaction, mode='same', boundary='fill'), self.m)

    def Update(self):
        self.Energy()
        self.barrier = self.E_b - self.E_int
        self.rate = np.exp(self.barrier/self.T)
        taus = np.random.exponential(scale=self.rate)
        indexmin = np.argmin(taus, axis=None)
        self.m.flat[indexmin] = -self.m.flat[indexmin]
        self.t += taus.flat[indexmin]
        if self.m_type == 'op':
            self.m_tot = np.mean(self.m)
        elif self.m_type == 'ip':
            self.m_tot_x = np.mean(np.multiply(self.m, self.orientation[:,:,0]))
            self.m_tot_y = np.mean(np.multiply(self.m, self.orientation[:,:,1]))
            self.m_tot = (self.m_tot_x**2 + self.m_tot_y**2)**(1/2)

    def Minimize(self):
        self.Energy()
        indexmax = np.argmax(self.E_int, axis=None)
        self.m.flat[indexmax] = -self.m.flat[indexmax]
    
    def Save_history(self):
        """ Updates all members of self.history """
        if hasattr(self, "E_tot"):
            self.history.E.append(self.E_tot)
        else:
            self.history.E.append(self.Energy())
        self.history.t.append(self.t)
        self.history.T.append(self.T)
        self.history.m.append(self.m_tot)
    
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
        corr_binned = np.divide(corr_binned, counts)
        corr_length = np.sum(np.multiply(abs(corr_binned), distances))
        return corr_binned, distances, corr_length


@dataclass
class History:
    """ Stores the history of the energy, temperature, time, and average magnetization. """
    E: list = field(default_factory=list)
    T: list = field(default_factory=list)
    t: list = field(default_factory=list)
    m: list = field(default_factory=list)