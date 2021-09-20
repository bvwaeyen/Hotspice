# import math

import matplotlib.pyplot as plt
import numpy as np

from context import hotspin
from matplotlib import animation
from numpy.core.numeric import Inf
# from scipy import signal


## Parameters, meshgrid
T = 0.1
E_b = 10.
nx = 51 # Multiple of 4 + 1
ny = int(nx/np.sqrt(3))//4*4 - 1 # To have a nice square-like shape of hexagons
x = np.linspace(0, nx - 1, nx)/np.sqrt(3)
y = np.linspace(0, ny - 1, ny)
xx, yy = np.meshgrid(x, y)

## Initialize main Magnets object
mm = hotspin.Magnets(xx, yy, T, E_b, 'ip', 'triangle', 'AFM', energies=['dipolar'])


def run_a_bit(mm, N=50e3, T=0.2, show_m=True):
    ''' Simulates <N> consecutive switches at temperature <T> and plots the end result.
        This end plot can be disabled by setting <show_m> to False.
    '''
    mm.Run(N=N, T=T)
    print('Energy:', mm.Energy())
    if show_m:
        mm.Show_m(average=True)


def animate_quenching(mm, animate=1, speed=20, n_sweep=20000, T_low=0.01, T_high=4):
    """ Shows an animation of repeatedly sweeping the simulation between quite low and high temperatures,
        WITH a smooth temperature transition in between (exponential between T_low and T_high).
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (20): How many switches are simulated between each frame.
        @param n_sweep [int] (20000): The number of switches between the temperature extrema.
    """
    mm.Initialize_m('AFM')

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(111)
    h = ax1.imshow(mm.Get_magAngles(avg='triangle'), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
    ax1.set_title(r'Averaged magnetization angle')
    c1 = plt.colorbar(h)
    fig.suptitle('Temperature %.3f [a.u.]' % mm.T)

    # This is the function that gets called each frame
    def animate_quenching_update(i):
        currStep = i*speed
        prev_T = mm.T
        for j in range(currStep, currStep + speed):
            exponent = np.log(T_high/T_low)
            if j % (2*n_sweep) > n_sweep: # Then cool down
                mm.T = T_low*np.exp(exponent*(((n_sweep-j)%n_sweep)/n_sweep))
            else: # Then heat up
                mm.T = T_low*np.exp(exponent*((j%n_sweep)/n_sweep))
            mm.Update()
        h.set_array(mm.Get_magAngles(avg='triangle'))
        fig.suptitle('Temperature %.3f' % mm.T)
        return h, # This has to be an iterable!

    # Assign the animation to a variable, to prevent it from getting garbage-collected
    anim = animation.FuncAnimation(fig, animate_quenching_update, 
                                    frames=int((2*n_sweep)//speed), interval=speed/animate, 
                                    blit=False, repeat=True)
    plt.show()


def autocorrelation_dist_dependence(mm):
    ''' Shows the full 2D autocorrelation, as well as the binned autocorrelation
        as a function of distance. '''
    corr, d, corr_length = mm.Autocorrelation_fast(20)
    print("Correlation length:", corr_length)

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle('T=%.2f' % mm.T, font={"size":12})
    ax1 = fig.add_subplot(121)
    ax1.plot(d, corr)
    ax1.set_xlabel(r'Distance [a.u.]')
    ax1.set_ylabel(r'Autocorrelation')
    ax2 = fig.add_subplot(122)
    h1 = ax2.imshow(mm.correlation, origin='lower', cmap='bone')
    c2 = plt.colorbar(h1)
    c2.set_label(r'Correlation', rotation=270, labelpad=15)
    plt.gcf().tight_layout()
    plt.show()


def autocorrelation_temp_dependence(mm, N=31, M=50, L=500, T_min=1, T_max=1.3):
    ''' Shows how the correlation distance depends on the temperature.
        @param N [int] (31): Number of temperature steps between <T_min> and <T_max>.
        @param M [int] (50): How many times to do <L> switches at each temperature,
            which are then averaged to get the correlation length at each temperature.
        @param L [int] (500): Number of switches between each measurement of the
            correlation length.
    '''
    # Calculate the correlation distance as a function of temperature
    mm.Initialize_m('AFM')
    TT = np.linspace(T_min, T_max, N)
    T_step = TT[1] - TT[0]
    corr_length = np.empty((N, M))
    niter = np.ones(N, dtype=int)*10*L
    niter[0] = 40*L
    for j, T in enumerate(TT):
        print('Temperature step %d/%d (T = %.2f) ...' % (j+1, N, T))
        mm.T = T
        for _ in range(niter[j]): # Update for a while to ensure that the different temperatures are not very correlated
            mm.Update()
        for k in range(M):
            corr, d, corr_length[j,k] = mm.Autocorrelation_fast(20)
            if k < M - 1: # Always except the last step
                for i in range(L):
                    mm.Update()
    corr_ll = np.mean(corr_length, axis=1)

    # Draw a nice plot of all this
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(121)
    extent = [-0.5,corr_length.shape[1]-0.5,np.min(TT)-T_step/2,np.max(TT)+T_step/2] # Adding all these halves to place the pixels correctly
    im1 = ax1.imshow(corr_length, origin='lower', interpolation='nearest', cmap='bone', extent=extent, aspect='auto')
    c1 = plt.colorbar(im1) 
    c1.set_label(r'Correlation length [a.u.]', rotation=270, labelpad=15)
    ax1.set_xlabel('Step (x%d)' % L)
    ax1.set_ylabel('Temperature [a.u.]')
    ax2 = fig.add_subplot(122)
    ax2.plot(TT, corr_ll)
    ax2.set_xlabel('Temperature [a.u.]')
    ax2.set_ylabel(r'$\langle$Correlation length$\rangle$ [a.u.]')
    plt.gcf().tight_layout()
    plt.show()


if __name__ == "__main__":
    print('Initialization energy:', mm.Energy())

    # run_a_bit(mm, N=10e3, T=0.1)
    # animate_quenching(mm, animate=3, speed=50)
    # autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, N=31) # Since kagome is quite sparse behind-the-scenes, it is doubtable whether the autocorrelation has a significant meaning