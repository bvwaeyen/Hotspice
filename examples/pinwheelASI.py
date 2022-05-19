import math
import time

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
# from cupyx.scipy import signal

import examplefunctions as ef
from context import hotspin


## Parameters, meshgrid
T = 300 # [K]
E_B = 5e-22 # [J]
n = 400

## Initialize main Magnets object
t = time.perf_counter()
simparams = hotspin.SimParams(SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF=1000)
mm = hotspin.ASI.PinwheelASI(n, 2e-6, T=T, E_B=E_B, pattern='uniform', energies=[hotspin.DipolarEnergy()], PBC=True, params=simparams)
print(f'Initialization time: {time.perf_counter() - t} seconds.')


def animate_temp_rise(mm: hotspin.Magnets, animate=1, speed=100, T_step=0.05, T_max=160):
    ''' Shows an animation of increasing the temperature gradually from 0 to <T_max>, which could reveal
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
    '''
    mm.initialize_m('uniform')
    mm.history_clear()

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    h = ax1.imshow(hotspin.plottools.get_rgb(mm, fill=True), cmap='hsv', origin='lower', vmin=0, vmax=2*math.pi, interpolation_stage='rgba', interpolation='antialiased')
    ax1.set_title(r'Averaged magnetization angle [rad]')
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
            mm.update()
            mm.history_save()
        p.set_data(mm.history.T, mm.history.m)
        h.set_array(hotspin.plottools.get_rgb(mm, fill=True))
        return h, p

    anim = animation.FuncAnimation(fig, animate_temp_rise_update, 
                                    frames=int(T_max/T_step//speed)+1, interval=speed/2/animate, 
                                    blit=False, repeat=False, init_func=lambda:0) # Provide empty init_func: otherwise the update func is used as init and is thus called twice for the 0th frame
    plt.show()


if __name__ == "__main__":
    print('Initialization energy:', mm.E_tot)

    # ef.run_a_bit(mm, N=5000, T=60, verbose=True, fill=True)
    # ef.curieTemperature(mm)
    # ef.animate_quenching(mm, pattern='random', T_low=30, T_high=30, animate=1, speed=50, fill=True)
    # animate_temp_rise(mm, animate=3, speed=100, T_step=0.02)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=20, T_max=80)

    #### Commands which do some specific thing which yields nice saved figures or videos
    # hotspin.plottools.show_lattice(mm, 5, 5, save=True)
    # factor = 10 # Approximately how many switches occur per mm.update()
    # ef.animate_quenching(mm, pattern='random', T_low=30, T_high=30, animate=3, speed=500//factor, n_sweep=40000//factor, fill=True, save=25) # Optimized for nx = ny = 400
