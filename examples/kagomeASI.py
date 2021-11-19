import math
import time

# import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import animation
# from cupyx.scipy import signal

import examplefunctions as ef
from context import hotspin


## Parameters, meshgrid
T = 0.1
E_b = 10.
nx = 25 *4+1 # Multiple of 4 + 1

## Initialize main Magnets object
t = time.perf_counter()
mm = hotspin.ASI.KagomeASI(nx, 4, T=T, E_b=E_b, pattern='uniform', energies=['dipolar'], PBC=False)
print(f'Initialization time: {time.perf_counter() - t} seconds.')


if __name__ == "__main__":
    print('Initialization energy:', mm.E_tot)

    # ef.run_a_bit(mm, N=10e3, T=0.1)
    # ef.animate_quenching(mm, animate=3, speed=50, avg='triangle')
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=0.1, T_max=0.3) # Since kagome is quite sparse behind-the-scenes, it is doubtable whether the autocorrelation has a significant meaning