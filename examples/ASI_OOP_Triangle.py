import time

import examplefunctions as ef
from context import hotspice


## Parameters
T = 300 # [K]
E_B = 5e-22 # [J]
n = 100

## Initialize main Magnets object
t = time.perf_counter()
mm = hotspice.ASI.OOP_Triangle(800e-9, n, T=T, E_B=E_B, pattern='uniform', energies=[hotspice.DipolarEnergy()], PBC=True)
print(f'Initialization time: {time.perf_counter() - t} seconds.')


if __name__ == "__main__":
    print('Initialization energy:', mm.E_tot)

    # ef.run_a_bit(mm, N=10e2, T=160, verbose=True)
    # ef.neelTemperature(mm, T_max=400)
    # ef.animate_quenching(mm, avg='point', pattern='afm', animate=3, speed=50, T_low=20, T_high=300)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=150, T_max=200)

    #### Commands which do some specific thing which yields nice saved figures or videos
    # hotspice.plottools.show_lattice(mm, 10, 10, save=True, fall_off=2, scale=.7)
    # factor = 1 # Approximately how many switches occur per mm.update()
    # ef.animate_quenching(mm, pattern='uniform', T_low=0.01, T_high=4, animate=3, speed=50//factor, n_sweep=80000//factor, avg='square', fill=True, save=2) # Optimized for nx = ny = 100
