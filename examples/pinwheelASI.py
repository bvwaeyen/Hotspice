# import math

import matplotlib.pyplot as plt
import numpy as np

from context import hotspin
from matplotlib import animation
from numpy.core.numeric import Inf
# from scipy import signal


## Parameters, meshgrid
T = 0.2
E_b = 10.
nx = 29
ny = 29
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
xx, yy = np.meshgrid(x, y)

## Initialize main Magnets object
mm = hotspin.Magnets(xx, yy, T, E_b, 'square', 'ip')
mm.Initialize_ip('square', np.pi/4)
mm.Initialize_m_square('chess') # For pinwheel, this is a uniform magnetization in positive y-direction

## Choose which energy components are taken into account
mm.Dipolar_energy_init()


def run_a_bit(mm, N=50e3, T=0.2, show_m=True):
    ''' Simulates <N> consecutive switches at temperature <T> and plots the end result.
        This end plot can be disabled by setting <show_m> to False.
    '''
    mm.Run(N=N, T=T)
    print('Energy:', mm.Energy())
    if show_m:
        mm.Show_m(avg=True)


def curieTemperature(mm, N=5000):
    ''' A naive attempt at determining the Curie temperature, by looking at the average magnetization.
        @param N [int] (5000): The number of simulated switches at each individual temperature
    '''
    mm.Clear_history()
    mm.Initialize_m_square('chess') # Re-initialize mm, because otherwise domains cancel out for m_tot
    for T in np.linspace(0, 1, 100):
        total_m = np.zeros_like(mm.m)
        total_energy = 0
        mm.T = T
        for i in range(int(N)):
            mm.Update()
            total_m += mm.m
            total_energy += mm.E_tot
        total_m = total_m/N
        m_tot_x = np.mean(np.multiply(total_m, mm.orientation[:,:,0]))
        m_tot_y = np.mean(np.multiply(total_m, mm.orientation[:,:,1]))
        mm.Save_history(E_tot=total_energy/N, m_tot=(m_tot_x**2 + m_tot_y**2)**(1/2))
    mm.Show_history()


def animate_quenching(mm, animate=1, speed=50, n_low=20000, n_high=2000, T_low=0.15, T_high=1):
    """ Shows an animation of repeatedly putting the simulation at quite low and high temperatures,
        WITHOUT a smooth temperature transition in between.
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (50): How many switches are simulated between each frame.
        @param n_low [int] (20000): The number of switches at low T before returning to high T.
        @param n_high [int] (2000): The number of switches at high T before returning to low T.
    """
    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(111)
    h = ax1.imshow(mm.Get_magAngles(avg='cross'), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
    ax1.set_title(r'Averaged magnetization angle')
    c1 = plt.colorbar(h)
    fig.suptitle('Temperature %.3f [a.u.]' % mm.T)

    # This is the function that gets called each frame
    def animate_quenching_update(i):
        currStep = i*speed
        prev_T = mm.T
        for j in range(currStep, currStep + speed):
            mm.T = T_high if j%(n_low + n_high) < n_high else T_low
            mm.Update()
        if mm.T != prev_T: # Update the temperature text if T changed
            fig.suptitle('Temperature %.3f' % mm.T)
        h.set_array(mm.Get_magAngles(avg='cross'))
        return h, # This has to be an iterable!

    # Assign the animation to a variable, to prevent it from getting garbage-collected
    anim = animation.FuncAnimation(fig, animate_quenching_update, 
                                    frames=(n_low + n_high)//speed, interval=speed/animate, 
                                    blit=False, repeat=True)
    plt.show()


def animate_temp_rise(mm, animate=1, speed=1000, T_step=0.000005, T_max=0.4):
    """ Shows an animation of increasing the temperature gradually from 0 to <T_max>, which could reveal
        information about the Curie temperature. Caution has to be taken, however, since the graph shows
        the average magnetization, which above the Curie temperature is expected to approach 0, but this
        can also happen well below the Curie temperature if the spins settle in a vortex-like geometry,
        which also has an average magnetization near 0 but is clearly still an orderly state. A better
        indicator for passing the Curie temperature is probably the autocorrelation, though even then one
        will still have to take care not to increase the temperature too fast, otherwise the phase
        transitions will lag behind anyway.
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (1000): How many switches are simulated between each frame.
    """
    mm.Initialize_m_square('chess')
    mm.Clear_history()

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    h = ax1.imshow(mm.Get_magAngles(avg='cross'), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
    ax1.set_title(r'Averaged magnetization angle')
    c1 = plt.colorbar(h)
    ax2 = fig.add_subplot(212)
    p,  = ax2.plot(mm.history.T, mm.history.m)
    ax2.set_xlim(0, T_max)
    ax2.set_ylim(0, 0.4)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Average magnetization')

    # This is the function that gets called each frame
    def animate_temp_rise_update(i):
        currStep = i*speed
        for j in range(currStep, min(currStep + speed, int(T_max//T_step)+1)):
            mm.T = j*T_step
            mm.Update()
            mm.Save_history()
        p.set_data(mm.history.T, mm.history.m)
        h.set_array(mm.Get_magAngles(avg='cross'))
        return h, p

    anim = animation.FuncAnimation(fig, animate_temp_rise_update, 
                                    frames=int(T_max/T_step//speed)+1, interval=speed/2/animate, 
                                    blit=False, repeat=False, init_func=lambda:0) # Provide empty init_func: otherwise the update func is used as init and is thus called twice for the 0th frame
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


def autocorrelation_temp_dependence(mm, N=31, M=50, L=500, T_min=0.1, T_max=0.4):
    ''' Shows how the correlation distance depends on the temperature. 
        @param N [int] (31): Number of temperature steps between <T_min> and <T_max>.
        @param M [int] (50): How many times to do <L> switches at each temperature,
            which are then averaged to get the correlation length at each temperature.
        @param L [int] (500): Number of switches between each measurement of the
            correlation length.
    '''
    # Calculate the correlation distance as a function of temperature
    mm.Initialize_m_square('random')
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

    # run_a_bit(mm)
    # curieTemperature(mm)
    # animate_quenching(mm, animate=3, speed=50)
    # animate_temp_rise(mm, animate=3, speed=1000)
    # autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, N=31)