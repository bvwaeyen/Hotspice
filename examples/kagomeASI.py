# import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from numpy.core.numeric import Inf
# from scipy import signal

import examplefunctions
from context import hotspin


## Parameters, meshgrid
T = 0.1
E_b = 10.
nx = 51 # Multiple of 4 + 1
ny = int(nx/np.sqrt(3))//4*4 - 1 # To have a nice square-like shape of hexagons
x = np.linspace(0, nx - 1, nx)/np.sqrt(3)
y = np.linspace(0, ny - 1, ny)
xx, yy = np.meshgrid(x, y)

## Initialize main Magnets object
mm = hotspin.Magnets(xx, yy, T, E_b, 'ip', 'kagome', 'uniform', energies=['dipolar'])


def run_a_bit(mm, N=50e3, T=0.2, show_m=True):
    ''' Simulates <N> consecutive switches at temperature <T> and plots the end result.
        This end plot can be disabled by setting <show_m> to False.
    '''
    mm.Run(N=N, T=T)
    print('Energy:', mm.Energy())
    if show_m:
        mm.Show_m()


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
    h = ax1.imshow(mm.Get_magAngles(avg='triangle'), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
    ax1.set_title(r'Averaged magnetization angle')
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
        h.set_array(mm.Get_magAngles(avg='triangle'))
        fig.suptitle('Temperature %.3f' % mm.T)
        return h, # This has to be an iterable!

    # Assign the animation to a variable, to prevent it from getting garbage-collected
    anim = animation.FuncAnimation(fig, animate_quenching_update, 
                                    frames=int((2*n_sweep)//speed), interval=speed/animate, 
                                    blit=False, repeat=True)
    plt.show()


def autocorrelation_temp_dependence(mm, N=31, M=50, L=500, T_min=0.1, T_max=0.3):
    mm.Initialize_m('uniform')
    examplefunctions.autocorrelation_temp_dependence(mm, N=N, M=M, L=L, T_min=T_min, T_max=T_max)


if __name__ == "__main__":
    print('Initialization energy:', mm.Energy())

    # run_a_bit(mm, N=10e3, T=0.1)
    # animate_quenching(mm, animate=3, speed=50)
    # examplefunctions.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm) # Since kagome is quite sparse behind-the-scenes, it is doubtable whether the autocorrelation has a significant meaning