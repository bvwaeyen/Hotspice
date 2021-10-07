import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
# from cupyx.scipy import signal

import examplefunctions as ef
from context import hotspin


## Parameters, meshgrid
T = 0.1
E_b = 10.
nx = ny = 200
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
xx, yy = np.meshgrid(x, y)

## Initialize main Magnets object
mm = hotspin.Magnets(xx, yy, T, E_b, 'ip', 'square', 'AFM', energies=['dipolar'])


def animate_temp_rise(mm, animate=1, speed=1000, T_step=0.00005, T_max=3):
    """ Shows an animation of increasing the temperature gradually from 0 to <T_max>, which could reveal
        information about the Néel temperature. Caution has to be taken, however, not to increase the 
        temperature too fast, as otherwise the phase transitions will lag behind anyway. The dotted horizontal
        line indicates the AFM-ness of a perfectly random state.
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (1000): How many switches are simulated between each frame.
    """
    mm.Initialize_m('AFM')
    mm.Clear_history()
    AFM_ness = []

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    h = ax1.imshow(mm.Get_magAngles(), cmap='hsv', origin='lower', vmin=0, vmax=2*math.pi)
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
            AFM_ness.append(mm.Get_AFMness())
        p.set_data(mm.history.T, AFM_ness)
        h.set_array(mm.Get_magAngles())
        return h, p

    anim = animation.FuncAnimation(fig, animate_temp_rise_update, 
                                    frames=int(T_max/T_step//speed)+1, interval=speed/2/animate, 
                                    blit=False, repeat=False, init_func=lambda:0) # Provide empty init_func: otherwise the update func is used as init and is thus called twice for the 0th frame
    plt.show()


if __name__ == "__main__":
    print('Initialization energy:', mm.Energy())

    # ef.run_a_bit(mm, N=4e3, T=100, show_m=False)
    # ef.run_a_bit(mm, N=20e3, T=0.2)
    # ef.neelTemperature(mm, T_max=2)
    # ef.animate_quenching(mm, animate=3, speed=50)
    # animate_temp_rise(mm, animate=3, speed=1000)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=0.1, T_max=1)

    #### Commands which do some specific thing which yields nice saved figures or videos
    # ef.animate_quenching(mm, animate=3, speed=50, n_sweep=40000, save=2) # Optimized for nx = ny = 200