import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
from scipy import signal
import math

class Magnets:
    def __init__(self,xx,yy,T,E_b,config='full',m_type='op'):
        if np.shape(xx) != np.shape(yy):
            print('Error, xx and yy should have the same shape')
            return
        self.xx = xx
        self.yy = yy
        self.T = T
        self.t = 0.
        self.E_b = E_b
        self.m_type = m_type
        if m_type == 'op':
            if config == 'full':
                self.Initialize_m('random')
            elif config == 'square':
                self.Initialize_m_square('random')
            else :
                print('Bad config')
        elif m_type == 'ip':
            if config == 'square':
                self.Initialize_m_square('random')
                self.Initialize_ip('square')
            else :
                print('Bad config')
        else :
            print('Bad type')
        self.m_tot = np.mean(self.m)
        self.E_int = np.zeros_like(xx)
        self.index = range(nx*ny)

    def Initialize_m(self,config):
        if config == 'uniform':
            self.m = np.zeros(np.shape(xx))*2-1
        elif config == 'random':
            self.m = np.random.random_integers(0,high=1,size=np.shape(xx))*2-1
        elif config == 'chess':
            self.m = ((xx + yy ) % 2)*2-1
        else:
            self.m = np.random.random_integers(0,high=1,size=np.shape(xx))*2-1
        self.m_tot = np.mean(self.m)

    def Initialize_m_square(self,config):
        if config == 'uniform':
            self.m = np.ones(np.shape(xx))*2-1
        elif config == 'random':
            self.m = np.random.random_integers(0,high=1,size=np.shape(xx))*2-1
        elif config == 'chess':
            self.m = ((xx + yy ) % 2)*2-1
        else:
            self.m = np.random.random_integers(0,high=1,size=np.shape(xx))*2-1
        self.m_tot = np.mean(self.m)
        self.mask = np.zeros_like(self.m)
        self.mask[(xx+yy) % 2 == 1] = 1
        #self.mask[yy % 2 == 0] = 1
        self.m = np.multiply(self.m,self.mask)
        self.m_tot = np.mean(self.m)
      
    def Initialize_ip(self,config,angle=0.):
        self.orrientation = np.zeros(np.shape(self.m) + (2,))
        if config == 'square':
            self.orrientation[yy % 2 == 0,0] = np.cos(angle)
            self.orrientation[yy % 2 == 0,1] = np.sin(angle)
            self.orrientation[yy % 2 == 1,0] = np.cos(angle+np.pi/2)
            self.orrientation[yy % 2 == 1,1] = np.sin(angle+np.pi/2)
            self.orrientation[self.mask==0,0] =0
            self.orrientation[self.mask==0,1] =0
    
    def Energy(self):
        E = np.zeros_like(self.xx)
        if hasattr(self,'E_exchange'):
            self.Exchange_update()
            E = E + self.E_exchange
        if hasattr(self,'E_dipolar'):
            self.Dipolar_energy_update()
            E = E + self.E_dipolar
        if hasattr(self,'E_Zeeman'):
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
            self.E_Zeeman = - self.m * self.H_ext
        elif self.m_type == 'ip':
            self.E_Zeeman = - np.multiply(self.m,(self.H_ext[0] * self.orrientation[:,:,0] +self.H_ext[1] * self.orrientation[:,:,1]))
            
    def Dipolar_energy_init(self):
        self.Dipolar_interaction = np.empty((self.xx.size,self.xx.size))
        self.E_dipolar = np.empty_like(self.xx)
        for i in self.index:
            rrx = np.reshape(self.xx.flat[i]-self.xx,-1)
            rry = np.reshape(self.yy.flat[i]-self.yy,-1)
            rr_inv = (rrx**2+rry**2)**(-1/2)
            self.Dipolar_interaction[i] = rr_inv**3
            if self.m_type == 'ip':
                mxx = np.reshape(self.orrientation[:,:,0],-1)
                myy = np.reshape(self.orrientation[:,:,1],-1)
                self.Dipolar_interaction[i] += -3*np.multiply(rrx[i]*mxx+rry[i]*myy,rrx*mxx[i]+rry*myy[i])
        np.place(self.Dipolar_interaction,self.Dipolar_interaction==Inf,0.0)
        self.Dipolar_energy_update()

    def Dipolar_energy_update(self):
        temp = np.dot(self.Dipolar_interaction,np.reshape(self.m,self.m.size))
        self.E_dipolar = np.multiply(self.m,np.reshape(temp,self.xx.shape))

    def Exchange_init(self,J):
        self.Exchange_interaction = np.array([[0.,1.,0.],[1.,0,1.],[0.,1.,0.]])
        self.Exchange_J = J
        self.Exchange_update()

    def Exchange_update(self):
        self.E_exchange = - self.Exchange_J * np.multiply(signal.convolve2d(self.m,self.Exchange_interaction,mode='same',boundary='fill'),self.m)

    def Update(self):
        self.Energy()
        self.barrier = self.E_b-self.E_int
        self.rate = np.exp(self.barrier/self.T)
        taus = np.random.exponential(scale=self.rate)
        indexmin = np.argmin(taus,axis=None)
        self.m.flat[indexmin] = (-1)*self.m.flat[indexmin]
        self.t = self.t + taus.flat[indexmin]
        if self.m_type == 'op':
            self.m_tot = np.mean(self.m)
        elif self.m_type == 'ip':
            self.m_tot_x = np.mean(np.multiply(self.m,self.orrientation[:,:,0]))
            self.m_tot_y = np.mean(np.multiply(self.m,self.orrientation[:,:,1]))
            self.m_tot = (self.m_tot_x**2+self.m_tot_y**2)**(1/2)

    def Minimize(self):
        self.Energy()
        indexmax = np.argmax(self.E_int,axis=None)
        self.m.flat[indexmax] = -self.m.flat[indexmax]
    
    def Autocorrelation_fast(self,max_distance):
        max_distance = round(max_distance)
        s = np.shape(self.xx)
        if not(hasattr(self,'Distances')):
            self.Distances = (self.xx**2 + self.yy**2)**(1/2)
            self.Distance_range = math.ceil(np.max(self.Distances))
            self.corr_norm = 1/signal.convolve2d(np.ones_like(self.m),np.ones_like(self.m),mode='full',boundary='fill')
            maskcor = signal.convolve2d(self.mask,np.flipud(np.fliplr(self.mask)),mode='full',boundary='fill') * self.corr_norm
            self.corr_mask = maskcor[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)]
            self.corr_mask[self.corr_mask>0] = 1
        if self.m_type == 'op':
            corr=signal.convolve2d(self.m,np.flipud(np.fliplr(self.m)),mode='full',boundary='fill') * self.corr_norm
        elif self.m_type == 'ip':
            corr_x=signal.convolve2d(self.m*self.orrientation[:,:,0],np.flipud(np.fliplr(self.m*self.orrientation[:,:,0])),mode='full',boundary='fill')*self.corr_norm
            corr_y=signal.convolve2d(self.m*self.orrientation[:,:,1],np.flipud(np.fliplr(self.m*self.orrientation[:,:,1])),mode='full',boundary='fill')*self.corr_norm
            corr = (corr_x + corr_y)
        corr = corr*np.size(self.m)/np.sum(self.corr_mask)    
        self.correlation=corr[(s[0]-1):(2*s[0]-1),(s[1]-1):(2*s[1]-1)]**2
        corr_binned = np.zeros(max_distance+1)
        counts = np.zeros(max_distance+1)
        distances = np.linspace(0,max_distance,num=max_distance+1)
        for i in self.index:
            bin = math.floor(self.Distances.flat[i])
            if bin <= max_distance:
                corr_binned[bin] = corr_binned[bin] + self.correlation.flat[i]*self.corr_mask.flat[i]
                counts[bin] = counts[bin] + self.corr_mask.flat[i]
        corr_binned = np.divide(corr_binned, counts)
        corr_length = np.sum(np.multiply(abs(corr_binned),distances))
        return corr_binned, distances, corr_length
