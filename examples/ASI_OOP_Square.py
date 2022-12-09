import time

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

import examplefunctions as ef
from context import hotspice
if hotspice.config.USE_GPU:
    import cupy as xp
    from cupyx.scipy import signal
else:
    import numpy as xp
    from scipy import signal


## Parameters
T = 300 # [K]
E_B = 5e-22 # [J]
n = 100

## Initialize main Magnets object
t = time.perf_counter()
mm = hotspice.ASI.OOP_Square(1e-6, n, T=T, E_B=E_B, pattern='uniform', energies=[hotspice.DipolarEnergy()], PBC=True)
print(f"Initialization time: {time.perf_counter() - t} seconds.")


def animate_temp_rise(mm: hotspice.Magnets, animate=1, speed=100, T_step=0.05, T_max=800):
    """ Shows an animation of increasing the temperature gradually from 0 to <T_max>, which could reveal
        information about the Néel temperature. Caution has to be taken, however, not to increase the 
        temperature too fast, as otherwise the phase transitions will lag behind anyway. The dotted horizontal
        line indicates the AFM-ness of a perfectly random state.
        @param animate [float] (1): How fast the animation will go: this is inversely proportional to the
            time between two frames.
        @param speed [int] (1000): How many switches are simulated between each frame.
    """
    mm.initialize_m('AFM')
    mm.history_clear()
    AFM_ness = []

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    mask = hotspice.utils.asnumpy(hotspice.plottools.Average.resolve(mm._get_appropriate_avg()).mask)
    image = signal.convolve2d(mm.m, xp.asarray(mask), mode='valid', boundary='wrap' if mm.PBC else 'fill')
    h = ax1.imshow(hotspice.utils.asnumpy(image), cmap='gray', origin='lower',
                   vmin=-np.sum(mask), vmax=np.sum(mask), interpolation_stage='rgba', interpolation='antialiased')
    ax1.set_title("Averaged magnetization")
    c1 = plt.colorbar(h)
    ax2 = fig.add_subplot(212)
    p,  = ax2.plot(mm.history.T, mm.history.m)
    ax2.axhline(3/8, linestyle=':', linewidth=1, color='grey')
    ax2.set_xlim(0, T_max)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Average AFM-ness")

    # This is the function that gets called each frame
    def animate_temp_rise_update(i):
        currStep = i*speed
        for j in range(currStep, min(currStep + speed, int(T_max//T_step)+1)):
            mm.T = j*T_step
            mm.update()
            mm.history_save()
            AFM_ness.append(hotspice.plottools.get_AFMness(mm))
        p.set_data(mm.history.T, AFM_ness)
        h.set_array(hotspice.utils.asnumpy(signal.convolve2d(mm.m, mask, mode='valid', boundary='fill')))
        return h, p

    anim = animation.FuncAnimation(fig, animate_temp_rise_update, 
                                    frames=int(T_max/T_step//speed)+1, interval=speed/2/animate, 
                                    blit=False, repeat=False, init_func=lambda:0) # Provide empty init_func: otherwise the update func is used as init and is thus called twice for the 0th frame
    plt.show()

def test(mm: hotspice.Magnets, T_low=200, T_high=400, T_steps=100, N=1e2, verbose=False):
    """ This function tests the time per step as a function of the # of switches per iteration. 
        Hence, this function serves to verify if the value mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF is well-chosen.
        Ideally, the plot rises to the left of the black line, and instantly becomes constant to the right of it.
        Any excessive curvature (either up OR down) directly to the right of the black line indicates a less-than-ideal value.
        NOTE: the curve is always quite smooth, since every iteration has a random # of samples so always
              a random amount of them are below or above the summation/convolution threshold which smooths things out.
        NOTE: the ideal value will depend on mm.params.REDUCED_KERNEL_SIZE, so this is gonna get quite complicated.
        (This is highly dependent on the performance of the computer, so the user should decide this threshold manually)
    """
    central = mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF # Alias this variable because its name is long af boii
    temps = np.logspace(np.log10(T_low), np.log10(T_high), T_steps)
    times = np.zeros_like(temps)
    switches = np.zeros_like(times)
    for i, temp in enumerate(temps):
        mm.initialize_m(mm._get_groundstate())
        if verbose: print(f"T={temp}")
        n = mm.switches
        times[i] = ef.run_a_bit(mm, N=N, T=temp, show_m=False)
        switches[i] = mm.switches - n
    plt.plot(switches/N, times/N)
    plt.scatter(switches/N, times/N, s=(temps/np.min(temps))**2*10) # Larger point means higher temperature
    plt.axvline(central, linestyle=':', color='black')
    plt.xlabel("Switches per iteration")
    plt.ylabel("Time per iteration [s]")
    plt.title(f"{type(mm).__name__}\nT={T_low:.1f}$\\rightarrow${T_high:.1f}\u2009K, {N:.0f} iterations per dot")
    plt.text(central-1, np.min(times/N), "Summation", ha='right', va='bottom', size=10, bbox=dict(boxstyle='larrow,pad=0.3', fc='white', ec='#00000080', lw=2))
    plt.text(central+1, np.min(times/N), "Convolution", ha='left', va='bottom', size=10, bbox=dict(boxstyle='rarrow,pad=0.3', fc='white', ec='#00000080', lw=2))
    plt.show()

def testWolff():
    mm = hotspice.ASI.OOP_Square(1, 400, energies=[hotspice.ExchangeEnergy(J=hotspice.kB*300)], PBC=True, pattern='uniform', T=481)
    fig = None
    while True: 
        mm._update_Wolff()
        fig = hotspice.plottools.show_m(mm, fill=False, figure=fig)


if __name__ == "__main__":
    print("Initialization energy:", mm.E_tot)
    # test(mm, T_low=200, T_high=400, T_steps=20, verbose=True) # Tailored to nx=ny=100
    # testWolff()

    # ef.run_a_bit(mm, N=10e2, T=160, verbose=True)
    # ef.neelTemperature(mm, T_max=400)
    # ef.animate_quenching(mm, avg='square', animate=3, speed=50)
    # animate_temp_rise(mm, animate=3, speed=100, T_step=.05)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=150, T_max=200)

    #### Commands which do some specific thing which yields nice saved figures or videos
    # hotspice.plottools.show_lattice(mm, 10, 10, save=True, fall_off=2, scale=.7)
    # factor = 1 # Approximately how many switches occur per mm.update()
    # ef.animate_quenching(mm, pattern='uniform', T_low=0.01, T_high=4, animate=3, speed=50//factor, n_sweep=80000//factor, avg='square', fill=True, save=2) # Optimized for nx = ny = 100
