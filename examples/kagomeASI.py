import math
import time

# import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import animation
# from cupyx.scipy import signal

import examplefunctions as ef
from context import hotspin


## Parameters, meshgrid
T = 300 # [K]
E_b = 5e-22 # [J]
nx = 25 *4+1 # Multiple of 4 + 1

## Initialize main Magnets object
t = time.perf_counter()
mm = hotspin.ASI.KagomeASI(nx, 4e-6, T=T, E_b=E_b, pattern='uniform', energies=[hotspin.DipolarEnergy()], PBC=False)
print(f'Initialization time: {time.perf_counter() - t} seconds.')


if __name__ == "__main__":
    print('Initialization energy:', mm.E_tot)

    # ef.run_a_bit(mm, N=10e3, T=20)
    # ef.animate_quenching(mm, animate=3, speed=50, avg='triangle')
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=20, T_max=60) # Since kagome is quite sparse behind-the-scenes, it is doubtable whether the autocorrelation has a significant meaning
    
    #### Commands which do some specific thing which yields nice saved figures or videos
    # hotspin.plottools.show_lattice(mm, 5, 3, save=True, fall_off=.5)
