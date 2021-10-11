import math

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
ny = int(nx/math.sqrt(3))//4*4 - 1 # To have a nice square-like shape of hexagons
x = np.linspace(0, nx - 1, nx)/math.sqrt(3)
y = np.linspace(0, ny - 1, ny)
xx, yy = np.meshgrid(x, y)

## Initialize main Magnets object
mm = hotspin.Magnets(xx, yy, T, E_b, 'ip', 'triangle', 'AFM', energies=['dipolar'])


if __name__ == "__main__":
    print('Initialization energy:', mm.E_tot)

    # ef.run_a_bit(mm, N=10e3, T=0.1)
    # neelTemperature(mm, T_max=3)
    # ef.animate_quenching(mm, animate=3, speed=50)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=1, T_max=1.3) # Since kagome is quite sparse behind-the-scenes, it is doubtable whether the autocorrelation has a significant meaning