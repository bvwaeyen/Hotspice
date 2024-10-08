import time

import examplefunctions as ef
import hotspice


## Parameters
T = 300 # [K]
E_B = 5e-22 # [J]
nx = 25 *4+1 # Multiple of 4 + 1

## Initialize main Magnets object
t = time.perf_counter()
mm = hotspice.ASI.IP_Triangle(4e-6, nx, T=T, E_B=E_B, pattern='AFM', energies=[hotspice.DipolarEnergy()], PBC=False)
print(f"Initialization time: {time.perf_counter() - t} seconds.")


if __name__ == "__main__":
    print("Initialization energy:", mm.E_tot)

    # ef.run_a_bit(mm, N=10e3, T=20)
    # neelTemperature(mm, T_max=600)
    # ef.animate_quenching(mm, animate=3, speed=50)
    # ef.autocorrelation_dist_dependence(mm)
    # autocorrelation_temp_dependence(mm, T_min=200, T_max=300) # Since kagome is quite sparse behind-the-scenes, it is doubtable whether the autocorrelation has a significant meaning

    #### Commands which do some specific thing which yields nice saved figures or videos
    # hotspice.plottools.show_lattice(mm, 5, 3, save=True, fall_off=.5, scale=1)
