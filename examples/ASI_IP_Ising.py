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
mm = hotspice.ASI.IP_Ising(1e-6, n, T=T, E_B=E_B, pattern='uniform', energies=[hotspice.DipolarEnergy()], PBC=True)
print(f"Initialization time: {time.perf_counter() - t} seconds.")
# mm.add_energy(hotspice.ExchangeEnergy(J=hotspice.utils.eV_to_J(0.0258/7)))
# mm.remove_energy('dipolar')


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
    mm.history.clear()
    AFM_ness = []

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    mask = hotspice.utils.asnumpy(hotspice.plottools.Average.resolve(mm.get_appropriate_avg()).mask)
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


if __name__ == "__main__":
    print("Initialization energy:", mm.E_tot)

    # ef.run_a_bit(mm, N=10e3, T=300)
    # ef.neelTemperature(mm, T_max=400)
    # ef.animate_quenching(mm, animate=3, speed=50, pattern='random')
    # animate_temp_rise(mm, animate=3, speed=100, T_step=0.05)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=150, T_max=200)

    #### Commands which do some specific thing which yields nice saved figures or videos
    # hotspice.plottools.show_lattice(mm, 7, 7, save=True, fall_off=2)
