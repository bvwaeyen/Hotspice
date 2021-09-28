# import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
# from cupyx.scipy import signal

import examplefunctions as ef
from context import hotspin


## Parameters, meshgrid
T = 0.2
E_b = 10.
nx = ny = 50
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
xx, yy = np.meshgrid(x, y)

## Initialize main Magnets object: pinwheel with uniform magnetization in +y direction
mm = hotspin.Magnets(xx, yy, T, E_b, 'ip', 'pinwheel', 'uniform', energies=['dipolar'])


def curieTemperature(mm, N=5000):
    ''' A naive attempt at determining the Curie temperature, by looking at the average magnetization.
        @param N [int] (5000): The number of simulated switches at each individual temperature
    '''
    mm.Clear_history()
    mm.Initialize_m('uniform') # Re-initialize mm, because otherwise domains cancel out for m_tot
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
    h = ax1.imshow(mm.Get_magAngles(), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
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
        h.set_array(mm.Get_magAngles())
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
    mm.Initialize_m('uniform')
    mm.Clear_history()

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    h = ax1.imshow(mm.Get_magAngles(), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
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
        h.set_array(mm.Get_magAngles())
        return h, p

    anim = animation.FuncAnimation(fig, animate_temp_rise_update, 
                                    frames=int(T_max/T_step//speed)+1, interval=speed/2/animate, 
                                    blit=False, repeat=False, init_func=lambda:0) # Provide empty init_func: otherwise the update func is used as init and is thus called twice for the 0th frame
    plt.show()


def autocorrelation_temp_dependence(mm, N=31, M=50, L=500, T_min=0.1, T_max=0.4):
    mm.Initialize_m('random')
    ef.autocorrelation_temp_dependence(mm, N=N, M=M, L=L, T_min=T_min, T_max=T_max)


if __name__ == "__main__":
    print('Initialization energy:', mm.E_tot)

    ef.run_a_bit(mm, N=5000, T=0.3, timeit=True)
    # curieTemperature(mm)
    # animate_quenching(mm, animate=3, speed=50)
    # animate_temp_rise(mm, animate=3, speed=1000)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm)