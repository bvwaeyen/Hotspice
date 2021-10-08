import os
import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation

from context import hotspin


def run_a_bit(mm: hotspin.Magnets, N=50e3, T=0.2, show_m=True, timeit=False, fill=False):
    ''' Simulates <N> consecutive switches at temperature <T> and plots the end result.
        This end plot can be disabled by setting <show_m> to False.
    '''
    if timeit: t = time.time()
    mm.Run(N=N, T=T)
    if timeit: print(f"Simulated {N} switches (on {mm.m.shape[0]}x{mm.m.shape[1]} grid) in {time.time() - t} seconds.")
    print('Energy:', mm.E_tot)
    if show_m:
        mm.Show_m(fill=fill)


def neelTemperature(mm: hotspin.Magnets, N=200000, T_min=0, T_max=1):
    ''' A naive attempt at determining the Néel temperature, by looking at the antiferromagnetic-ness.
        @param N [int] (200000): The number of temperature steps (with 1 switch each) between T_min and T_max.
    '''
    mm.Clear_history()
    mm.Initialize_m('AFM')
    AFM_ness = []

    for T in np.linspace(T_min, T_max, N):
        mm.T = T
        mm.Update()
        AFM_ness.append(mm.Get_AFMness())
        mm.Save_history()
    mm.Show_history(y_quantity=AFM_ness, y_label=r'AFM-ness')


def animate_quenching(mm: hotspin.Magnets, animate=1, speed=20, n_sweep=40000, T_low=0.01, T_high=4, save=False, fill=False, avg=True, pattern='uniform'):
    """ Shows an animation of repeatedly sweeping the simulation between quite low and high temperatures,
        WITH a smooth temperature transition in between (exponential between T_low and T_high).
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (20): How many switches are simulated between each frame.
        @param n_sweep [int] (40000): The number of switches in one low->high->low cycle.
        @param save [int] (0): If nonzero, this specifies the amount of cycles that are saved as a video.
        @param fill [bool] (False): If true, empty simulation cells are interpolated to prevent white pixels.
        @param avg [str] (True): Which averaging mask to use in the plot.
        @param pattern [str] ('uniform'): The initial magnetization pattern (any of 'random', 'uniform', 'AFM').
            Set to a False value to prevent initialization of the magnetization.
    """
    if pattern:
        mm.Initialize_m(pattern)
    n_sweep2 = n_sweep/2

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(6, 4.8))
    ax1 = fig.add_subplot(111)
    h = ax1.imshow(mm.Get_magAngles(avg=avg), cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi, extent=mm._get_averaged_extent(avg))
    c1 = plt.colorbar(h)
    c1.ax.get_yaxis().labelpad = 30
    c1.ax.set_ylabel('Averaged magnetization angle' + ('\n("%s" average)' % mm._resolve_avg(avg) if avg != 'point' else ''), rotation=270, fontsize=12)
    fig.suptitle('Temperature %.3f' % mm.T)

    # This is the function that gets called each frame
    def animate_quenching_update(i):
        currStep = i*speed
        exponent = np.log(T_high/T_low)
        for j in range(currStep, currStep + speed):
            if j % n_sweep < n_sweep2: # Then heat up
                mm.T = T_low*np.exp(exponent*((j % n_sweep)/n_sweep2))
            else: # Then cool down
                mm.T = T_low*np.exp(exponent*(((n_sweep - j) % n_sweep2)/n_sweep2))
            mm.Update()
        h.set_array(hotspin.fill_nan_neighbors(mm.Get_magAngles(avg=avg)) if fill else mm.Get_magAngles(avg=avg))
        fig.suptitle('Temperature %.3f' % mm.T)
        return h, # This has to be an iterable!

    # Assign the animation to a variable, to prevent it from getting garbage-collected
    anim = animation.FuncAnimation(fig, animate_quenching_update, 
                                    frames=int(n_sweep//speed)*max(1, save), interval=speed/animate, 
                                    blit=False, repeat=True)
    if save:
        mywriter = animation.FFMpegWriter(fps=30)
        if not os.path.exists('videos'): os.makedirs('videos')
        anim.save(f'videos/{mm.config}_{mm.nx}x{mm.ny}_T{T_low}-{T_high}_N{n_sweep}x{save}.mp4', writer=mywriter, dpi=300)

    plt.show()


def autocorrelation_dist_dependence(mm: hotspin.Magnets):
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
    h1 = ax2.imshow(mm.correlation.get(), origin='lower', cmap='bone')
    c2 = plt.colorbar(h1)
    c2.set_label(r'Correlation', rotation=270, labelpad=15)
    plt.gcf().tight_layout()
    plt.show()


def autocorrelation_temp_dependence(mm: hotspin.Magnets, N=41, M=50, L=500, T_min=0, T_max=2):
    ''' Shows how the correlation distance depends on the temperature. 
        @param N [int] (31): Number of temperature steps between <T_min> and <T_max>.
        @param M [int] (50): How many times to do <L> switches at each temperature,
            which are then averaged to get the correlation length at each temperature.
        @param L [int] (500): Number of switches between each measurement of the
            correlation length.
    '''
    # Initialize in the ground state
    if mm.config in ['pinwheel', 'kagome']:
        mm.Initialize_m('random')
    elif mm.config in ['full', 'square', 'triangle']:
        mm.Initialize_m('AFM')

    # Calculate the correlation distance as a function of temperature
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
    extent = [-0.5, corr_length.shape[1]-0.5, T_min-T_step/2, T_max+T_step/2] # Adding all these halves to place the pixels correctly
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
    nx = ny = 29
    x, y = np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny)
    xx, yy = np.meshgrid(x, y)
    mm = hotspin.Magnets(xx, yy, 0.2, 10., 'ip', 'pinwheel', 'uniform', energies=['dipolar'])
    autocorrelation_dist_dependence(mm)
    autocorrelation_temp_dependence(mm, N=41, T_min=0.05, T_max=0.45)