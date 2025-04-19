import numpy as np

import os
os.environ["HOTSPICE_USE_GPU"] = "false"
import hotspice

## System parameters
l = 470e-9 # Hexagon side length    [m]
m = 1.1278401e-15 # Magnetic moment [Am²]

for nx in range(5, 11): # Test various system sizes (and then trim to 1 hexagon) to catch bugs
    for ny in range(3, 11):
        ## Create 1 kagome hexagon
        mm = hotspice.ASI.IP_Kagome(a := l/np.tan(30*np.pi/180), nx=nx, ny=ny, moment=m, energies=[dd_energy := hotspice.energies.DipolarEnergy()], T=300, m_perp_factor=0, PBC=False)
        mm.occupation[3:,:] = mm.m[3:,:] = 0 # Remove magnets if ny > 3
        mm.occupation[:,5:] = mm.m[:,5:] = 0 # Remove magnets if nx > 5
        
        ## Set vortex state
        mm.m[(0,0,1),(1,3,4)] *= -1
        mm.update_energy()

        ## Check if simulation corresponds to theory
        E = -1e-7*m*m/a**3*(29+20*np.sqrt(3)/9) # Theoretical magnetostatic energy of all magnets in the hexagon
        DD_m = dd_energy.E[mm.m.astype(bool)] # DipolarEnergy of the magnets
        assert hotspice.xp.allclose(DD_m, E), f"Energy of magnets in a hexagon does not correspond to analytical solution (nx={nx},ny={ny})."

print(f"Successfully ran test: {hotspice.utils.get_caller_script().name}.")