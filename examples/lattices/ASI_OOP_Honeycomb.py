import time

import examplefunctions as ef
import hotspice


## Parameters
T = 300 # [K]
E_B = 5e-22 # [J]
n = 100

## Initialize main Magnets object
t = time.perf_counter()
mm = hotspice.ASI.OOP_Honeycomb(200e-9, n, T=T, E_B=E_B, pattern='uniform', energies=[hotspice.DipolarEnergy()], PBC=True)
print(f"Initialization time: {time.perf_counter() - t} seconds.")


if __name__ == "__main__":
    print("Initialization energy:", mm.E_tot)

    #### Commands which do some specific thing which yields nice saved figures or videos
    hotspice.plottools.show_lattice(mm, 4, save=True, fall_off=.75, scale=.7, save_ext='.pdf')
