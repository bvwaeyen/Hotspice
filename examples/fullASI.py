# import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from numpy.core.numeric import Inf
from scipy import signal

import examplefunctions
from context import hotspin


## Parameters, meshgrid
T = 0.1
E_b = 10.
nx = 29
ny = 29
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
xx, yy = np.meshgrid(x, y)

## Initialize main Magnets object
mm = hotspin.Magnets(xx, yy, T, E_b, 'op', 'full', 'uniform', energies=['dipolar']) # config='chess' is just the same but with different boundaries


def run_a_bit(mm, N=50e3, T=0.2, show_m=True):
    ''' Simulates <N> consecutive switches at temperature <T> and plots the end result.
        This end plot can be disabled by setting <show_m> to False.
    '''
    mm.Run(N=N, T=T)
    print('Energy:', mm.Energy())
    if show_m:
        mm.Show_m(average=True)


def neelTemperature(mm, N=200000):
    ''' A naive attempt at determining the Néel temperature, by looking at the antiferromagnetic-ness.
        @param N [int] (200000): The number of temperature steps (with 1 switch each) between T_min and T_max.
    '''
    mm.Clear_history()
    mm.Initialize_m('AFM')
    AFM_mask = [[1, -1], [-1, 1]]
    AFM_ness = []

    for T in np.linspace(0, 2, N):
        mm.T = T
        mm.Update()
        AFM_ness.append(np.mean(np.abs(signal.convolve2d(mm.m, AFM_mask, mode='same', boundary='fill')))/4)
        mm.Save_history()
    mm.Show_history(y_quantity=AFM_ness, y_label=r'AFM-ness')


def animate_quenching(mm, animate=1, speed=20, n_sweep=20000, T_low=0.01, T_high=4):
    """ Shows an animation of repeatedly sweeping the simulation between quite low and high temperatures,
        WITH a smooth temperature transition in between (exponential between T_low and T_high).
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (20): How many switches are simulated between each frame.
        @param n_sweep [int] (20000): The number of switches between the temperature extrema.
    """
    mm.Initialize_m('uniform')

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(111)
    mask = mm._get_mask()
    h = ax1.imshow(signal.convolve2d(mm.m, mask, mode='valid', boundary='fill'),
                             cmap='gray', origin='lower', vmin=-np.sum(mask), vmax=np.sum(mask))
    ax1.set_title(r'Averaged magnetization $\vert m \vert$')
    c1 = plt.colorbar(h)
    fig.suptitle('Temperature %.3f [a.u.]' % mm.T)

    # This is the function that gets called each frame
    def animate_quenching_update(i):
        currStep = i*speed
        for j in range(currStep, currStep + speed):
            exponent = np.log(T_high/T_low)
            if j % (2*n_sweep) > n_sweep: # Then cool down
                mm.T = T_low*np.exp(exponent*(((n_sweep-j)%n_sweep)/n_sweep))
            else: # Then heat up
                mm.T = T_low*np.exp(exponent*((j%n_sweep)/n_sweep))
            mm.Update()
        h.set_array(signal.convolve2d(mm.m, mask, mode='valid', boundary='fill'))
        fig.suptitle('Temperature %.3f' % mm.T)
        return h, # This has to be an iterable!

    # Assign the animation to a variable, to prevent it from getting garbage-collected
    anim = animation.FuncAnimation(fig, animate_quenching_update, 
                                    frames=int((2*n_sweep)//speed), interval=speed/animate, 
                                    blit=False, repeat=True)
    plt.show()


def animate_temp_rise(mm, animate=1, speed=1000, T_step=0.00005, T_max=4):
    """ Shows an animation of increasing the temperature gradually from 0 to <T_max>, which could reveal
        information about the Néel temperature. Caution has to be taken, however, not to increase the 
        temperature too fast, as otherwise the phase transitions will lag behind anyway. The dotted horizontal
        line indicates the AFM-ness of a perfectly random state.
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (1000): How many switches are simulated between each frame.
    """
    mm.Initialize_m('uniform')
    mm.Clear_history()
    AFM_mask = [[1, -1], [-1, 1]]
    AFM_ness = []

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    mask = mm._get_mask()
    h = ax1.imshow(signal.convolve2d(mm.m, mask, mode='valid', boundary='fill'),
                             cmap='gray', origin='lower', vmin=-np.sum(mask), vmax=np.sum(mask))
    ax1.set_title(r'Averaged magnetization angle')
    c1 = plt.colorbar(h)
    ax2 = fig.add_subplot(212)
    p,  = ax2.plot(mm.history.T, mm.history.m)
    ax2.axhline(3/8, linestyle=':', linewidth=1, color='grey')
    ax2.set_xlim(0, T_max)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Average AFM-ness')

    # This is the function that gets called each frame
    def animate_temp_rise_update(i):
        currStep = i*speed
        for j in range(currStep, min(currStep + speed, int(T_max//T_step)+1)):
            mm.T = j*T_step
            mm.Update()
            mm.Save_history()
            AFM_ness.append(np.mean(np.abs(signal.convolve2d(mm.m, AFM_mask, mode='same', boundary='fill')))/4)
        p.set_data(mm.history.T, AFM_ness)
        h.set_array(signal.convolve2d(mm.m, mask, mode='valid', boundary='fill'))
        return h, p

    anim = animation.FuncAnimation(fig, animate_temp_rise_update, 
                                    frames=int(T_max/T_step//speed)+1, interval=speed/2/animate, 
                                    blit=False, repeat=False, init_func=lambda:0) # Provide empty init_func: otherwise the update func is used as init and is thus called twice for the 0th frame
    plt.show()


def autocorrelation_temp_dependence(mm):
    mm.Initialize_m('AFM')
    examplefunctions.autocorrelation_temp_dependence(mm, N=31, M=50, L=500, T_min=0.9, T_max=1.2)


if __name__ == "__main__":
    print('Initialization energy:', mm.Energy())

    # run_a_bit(mm, N=10e3, T=1)
    # neelTemperature(mm)
    # animate_quenching(mm, animate=3, speed=50)
    # animate_temp_rise(mm, animate=3, speed=1000)
    # examplefunctions.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm)