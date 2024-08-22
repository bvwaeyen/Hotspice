from __future__ import annotations # Interpret all type annotations as strings, otherwise we have to write e.g. 'Magnets'
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import Magnets # Only need Magnets for type annotations

from .utils import as_2D_array

from abc import ABC, abstractmethod
from functools import cache
import math
import numpy as np

from . import config
if config.USE_GPU:
    import cupy as xp
    from cupyx.scipy import signal
else:
    import numpy as xp
    from scipy import signal


class Energy(ABC):
    def __init__(self):
        """ The __init__ method contains all initialization of variables which do not depend
            on a specific given `Magnets()` object. It is not required to override this method.
        """
        pass

    def energy_switch(self, indices2D=None):
        """ Returns the change in energy experienced by the magnets at `indices2D`, if they were to switch.
            @param indices2D [xp.array(2, -)]: A 2D array with the first row containing the y-coordinates,
                and the second row containing the corresponding x-coordinates of the sampled magnets.
            @return [list(N)]: A list containing the local changes in energy for each magnet of `indices2D`, in the same order.
        """
        return -2*self.E if indices2D is None else -2*self.E[indices2D[0], indices2D[1]]

    def initialize(self, mm: Magnets):
        """ (Re)calculate energy for `mm`. """
        self.mm = mm # Like this
        self.E, self.E_perp = xp.zeros_like(self.mm.m), xp.zeros_like(self.mm.m)
        self._initialize()
        self.update()

    @abstractmethod
    def _initialize(self):
        """ Calculate properties that should not be recalculated after every switch, like the dipolar kernels.
            It is assumed that `self.mm` already exists.
        """

    @abstractmethod
    def update(self):
        """ Calculates the entire `self.E` and/or `self.E_perp` array(s), for the state of `self.mm.m`.
            No approximations should be made here: this serves to (re)calculate the whole energy.
        """
        self.E = xp.zeros_like(self.mm.xx) # [J]

    @abstractmethod
    def update_single(self, index2D):
        """ Updates `self.E` by only taking into account that a single magnet (at `index2D`) switched.
            @param index2D [tuple(2)]: A tuple containing two size-1 arrays representing y- and x-index of the switched magnet.
        """

    @abstractmethod
    def update_multiple(self, indices2D):
        """ Updates `self.E` by only taking into account that some magnets (at `indices2D`) switched.
            This seems like it is just multiple times `self.update_single()`, but sometimes an optimization is possible,
            hence this required alternative function for updating multiple magnets at once.
            @param indices2D [tuple(2)]: A tuple containing two equal-size 1D arrays representing the y- and x-
                indices of the sampled magnets, such that this tuple can be used directly to index `self.E`.
        """

    @property
    @abstractmethod
    def E_tot(self):
        """ Returns the total energy for this energy contribution. This function is necessary since this is not equal
            for all energies: e.g. `sum(E)` in the DipolarEnergy would count each interaction twice, while `sum(E)` is
            correct for ZeemanEnergy.
        """

    @property
    @cache
    def shortname(self):
        return type(self).__name__.lower().replace('energy', '')


class ZeemanEnergy(Energy):
    def __init__(self, magnitude=0, angle=0):
        """ This `ZeemanEnergy` class implements the Zeeman energy for a spatially uniform external field, whose magnitude
            (and `angle`, if the magnetization is in-plane) can be set using the `set_field` method.
            @param magnitude [float] (0): The magnitude of the external field.
            @param angle [float] (0): The angle (in radians) of the external field.
        """
        # NOTE: self._magnitude and self._angle are not cleaned, and can be int, float, array...
        #       self.magnitude and self.angle, on the other hand, are always cast as_2D_array() upon calling.
        self._magnitude = magnitude # [T]
        self._angle = angle # [rad]

    def _initialize(self):
        self.set_field(magnitude=self._magnitude, angle=self._angle)

    def set_field(self, magnitude=None, angle=None):
        if magnitude is not None: self._magnitude = magnitude
        if angle is not None: self._angle = angle
        if not hasattr(self, 'mm'): return

        if self.mm.in_plane:
            B_ext = (self.magnitude*xp.cos(self.angle), self.magnitude*xp.sin(self.angle)) # [T] tuple(2) of 2D xp.ndarray
            self.E_factor = -self.mm.moment*(B_ext[0]*self.mm.orientation[:,:,0] + B_ext[1]*self.mm.orientation[:,:,1])
            self.E_factor_perp = -self.mm.moment*(B_ext[1]*self.mm.orientation[:,:,0] - B_ext[0]*self.mm.orientation[:,:,1]) # Don't put this behind an 'if self.mm.USE_PERP_ENERGY' block, to prevent desyncs if USE_PERP_ENERGY gets changed. This is not the most expensive function in the world.
        else:
            B_ext = xp.copy(self.magnitude) # [T] 2D xp.ndarray to allow spatially varying external field
            self.E_factor = -self.mm.moment*B_ext
            self.E_factor_perp = xp.zeros_like(self.mm.m)
        self.update() # Fields were (probably) changed, so recalculate the energy

    def update(self):
        self.E = self.mm.m*self.E_factor
        if self.mm.USE_PERP_ENERGY: self.E_perp = self.mm.m*self.E_factor_perp
        
    def update_single(self, index2D):
        self.E[index2D[0,0], index2D[1,0]] *= -1
        if self.mm.USE_PERP_ENERGY: self.E_perp[index2D[0,0], index2D[1,0]] *= -1

    def update_multiple(self, indices2D):
        self.E[indices2D[0], indices2D[1]] *= -1
        if self.mm.USE_PERP_ENERGY: self.E_perp[indices2D[0], indices2D[1]] *= -1
        
    @property
    def E_tot(self):
        return xp.sum(self.E)

    @property
    def magnitude(self): return as_2D_array(self._magnitude, self.mm.shape)
    @magnitude.setter
    def magnitude(self, value): self.set_field(magnitude=value)

    @property
    def angle(self): return as_2D_array(self._angle, self.mm.shape)
    @angle.setter
    def angle(self, value): self.set_field(angle=value)


class DipolarEnergy(Energy):
    def __init__(self, prefactor: float = 1, decay_exponent: float = -3):
        """ This `DipolarEnergy` class implements the interaction between the magnets of the simulation themselves.
            It should therefore always be included in the simulations.
            @param prefactor [float] (1): The relative strength of the dipolar interaction.
            @param decay_exponent [float] (-3): How fast the dipole interaction weakens with distance: DD ∝ 1/r^`decay_exponent`.
        """
        # TODO: a more intricate scaling with distance is needed to incorporate the effect of the magnets not being infinitely small magnetic spins.
        #       Further steps can then concern themselves with the nonpolynomial fall-off with distance, as the magnets start to see each other more as infinitesimal spins rather than a finite FM geometry.
        self._prefactor = prefactor
        self.decay_exponent = decay_exponent
    
    @property
    def prefactor(self):
        return self._prefactor
    @prefactor.setter
    def prefactor(self, value):
        self.E *= value/self._prefactor
        self.E_perp *= value/self._prefactor
        self._prefactor = value

    def _initialize(self):
        """ When `self.mm` has nonzero `major_axis` and `minor_axis`, the magnet
            is assumed to be elliptical, and the approximation presented in
                Politi, P., & Pini, M. G. (2002). Dipolar interaction between two-
                dimensional magnetic particles. Physical Review B, 66(21), 214414.
            is used to emulate the finite size of nanomagnets.
        """
        self.unitcell = self.mm.unitcell
        I = (self.mm.major_axis**2 + self.mm.minor_axis**2)/16 # "Moment of inertia" to emulate finite size of magnets
        
        # Now we initialize the full ox
        num_unitcells_x = 2*math.ceil(self.mm.nx/self.unitcell.x) + 1
        num_unitcells_y = 2*math.ceil(self.mm.ny/self.unitcell.y) + 1
        if self.mm.in_plane:
            unitcell_ox = self.mm.orientation[:self.unitcell.y,:self.unitcell.x,0]
            unitcell_oy = self.mm.orientation[:self.unitcell.y,:self.unitcell.x,1]
        else:
            unitcell_o = self.mm.occupation[:self.unitcell.y,:self.unitcell.x]
        unitcell_dx, unitcell_dy = xp.meshgrid(self.mm.dx[:self.unitcell.x], self.mm.dy[:self.unitcell.y]) # TODO: there is an issue with dx/dy having length nx-1/ny-1
        tlm_dx = xp.tile(unitcell_dx, (num_unitcells_y, num_unitcells_x))
        tlm_dy = xp.tile(unitcell_dy, (num_unitcells_y, num_unitcells_x))
        # Now comes the part where we start splitting the different cells in the unit cells
        self.kernel_unitcell_indices = -xp.ones((self.unitcell.y, self.unitcell.x), dtype=int) # unitcell (y,x) -> kernel (i). If no magnet present, the kernel index is -1 to indicate this.
        self.kernel_unitcell = []
        self.kernel_perpself_unitcell = [] # Initialize perp regardless of mm.USE_PERP_ENERGY, it only takes some unnecessary memory if it is not needed, initialization is very fast anyway
        self.kernel_perpother_unitcell = []
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                # Slices for slicing "toolargematrix"es (tlm). These are (num_unitcells_x, num_unitcells_y) arrays of tiled unitcells.
                slice_startx = (self.unitcell.x - ((self.mm.nx-1) % self.unitcell.x) + x) % self.unitcell.x # Final % not strictly necessary because
                slice_starty = (self.unitcell.y - ((self.mm.ny-1) % self.unitcell.y) + y) % self.unitcell.y # toolargematrix_o{x,y} large enough anyway
                tlm_slice = np.s_[slice_starty:slice_starty+2*self.mm.ny-1,slice_startx:slice_startx+2*self.mm.nx-1]
                
                ## DETERMINE DISTANCES
                # The unitcell's dx and dy
                dx, dy = tlm_dx[tlm_slice], tlm_dy[tlm_slice]
                xx, yy = xp.cumsum(dx, axis=1), xp.cumsum(dy, axis=0) # The central magnet in this whole matrix is the one we are interested in. Therefore...
                rrx = xx - xx[0,self.mm.nx - 1]
                rry = yy - yy[self.mm.ny - 1,0]
                
                # Let us first make the four-mirrored distance matrix rinv3
                rr_sq = (rrx**2 + rry**2).astype(float)
                rr_sq[self.mm.ny - 1,self.mm.nx - 1] = xp.inf
                rr_inv = rr_sq**-0.5 # Due to the previous line, this is now never infinite
                rinv3 = rr_inv**(-self.decay_exponent) # = 1/r^3 by default
                rinv5 = rr_inv**(-self.decay_exponent + 2) # = 1/r^5 by default
                # Now we determine the normalized rx and ry
                ux = rrx*rr_inv
                uy = rry*rr_inv
                ## NORMAL KERNEL
                if self.mm.in_plane:
                    if (ox1 := unitcell_ox[y,x]) == (oy1 := unitcell_oy[y,x]) == 0:
                        continue # Empty cell in the unit cell, so don't store a kernel
                    tlm_ox = xp.tile(unitcell_ox, (num_unitcells_y, num_unitcells_x)) # This is the maximum that we can ever need (this maximum
                    tlm_oy = xp.tile(unitcell_oy, (num_unitcells_y, num_unitcells_x)) # occurs when the simulation does not cut off any unit cells)
                    ox2, oy2 = tlm_ox[tlm_slice], tlm_oy[tlm_slice] # Get the useful part of toolargematrix_o{x,y} for this (x,y) in the unit cell
                    k = ox1*ox2*(3*ux**2 - 1) + oy1*oy2*(3*uy**2 - 1) + 3*(ux*uy)*(ox1*oy2 + oy1*ox2)
                    k_correction = ox1*ox2*(5*ux**2 - 1) + oy1*oy2*(5*uy**2 - 1) + 5*(ux*uy)*(ox1*oy2 + oy1*ox2) # Add finite-size correction
                    kernel = -rinv3*k - 3*I/2*rinv5*k_correction
                else:
                    if unitcell_o[y,x] == 0:
                        continue # Empty cell in the unit cell, so don't store a kernel
                    tlm_o = xp.tile(unitcell_o, (num_unitcells_y, num_unitcells_x)).astype(float)
                    o2 = tlm_o[tlm_slice] # Get the useful part of toolargematrix_o for this (x,y) in the unit cell
                    kernel = o2*(rinv3 + 9*I/2*rinv5) # 'kernel' for out-of-plane: 1/r³ plus correction for finite size
                
                ## PERPENDICULAR KERNEL SELF (so the magnet in the center of the kernel is 'perpendicular')
                if self.mm.in_plane: # "Perpendicular" is defined as rotated 90° counterclockwise (mathematical positive direction). AFAIK, this is only important here and nowhere else.
                    ox1_perp, oy1_perp = -oy1, ox1
                    # Get the useful part of toolargematrix_o{x,y} for this (x,y) in the unit cell
                    k_perpself = ox1_perp*ox2*(3*ux**2 - 1) + oy1_perp*oy2*(3*uy**2 - 1) + 3*(ux*uy)*(ox1_perp*oy2 + oy1_perp*ox2)
                    k_correction_perpself = ox1_perp*ox2*(5*ux**2 - 1) + oy1_perp*oy2*(5*uy**2 - 1) + 5*(ux*uy)*(ox1_perp*oy2 + oy1_perp*ox2) # Add finite-size correction
                    kernel_perpself = -rinv3*k_perpself - 3*I/2*rinv5*k_correction_perpself
                else:
                    kernel_perpself = np.zeros_like(kernel) # Perpendicular magnetization in OOP ASI is always zero-energy due to symmetry

                ## PERPENDICULAR KERNEL OTHER (so the magnet in the center is the only one that is still normal)
                if self.mm.in_plane:
                    ox2_perp, oy2_perp = -oy2, ox2
                    # Get the useful part of toolargematrix_o{x,y} for this (x,y) in the unit cell
                    k_perpother = ox1*ox2_perp*(3*ux**2 - 1) + oy1*oy2_perp*(3*uy**2 - 1) + 3*(ux*uy)*(ox1*oy2_perp + oy1*ox2_perp)
                    k_correction_perpother = ox1*ox2_perp*(5*ux**2 - 1) + oy1*oy2_perp*(5*uy**2 - 1) + 5*(ux*uy)*(ox1*oy2_perp + oy1*ox2_perp) # Add finite-size correction
                    kernel_perpother = -rinv3*k_perpother - 3*I/2*rinv5*k_correction_perpother
                else:
                    kernel_perpother = np.zeros_like(kernel) # Perpendicular magnetization in OOP ASI is always zero-energy due to symmetry

                ## PBC
                def apply_PBC(k): # <k> is a DD kernel (2N+1,2N+1)-array
                    kopy = k.copy()
                    k[:,self.mm.nx:] += kopy[:,:self.mm.nx-1]
                    k[self.mm.ny:,self.mm.nx:] += kopy[:self.mm.ny-1,:self.mm.nx-1]
                    k[self.mm.ny:,:] += kopy[:self.mm.ny-1,:]
                    k[self.mm.ny:,:self.mm.nx-1] += kopy[:self.mm.ny-1,self.mm.nx:]
                    k[:,:self.mm.nx-1] += kopy[:,self.mm.nx:]
                    k[:self.mm.ny-1,:self.mm.nx-1] += kopy[self.mm.ny:,self.mm.nx:]
                    k[:self.mm.ny-1,:] += kopy[self.mm.ny:,:]
                    k[:self.mm.ny-1,self.mm.nx:] += kopy[self.mm.ny:,:self.mm.nx-1]
                if self.mm.PBC: # Just copy the kernel 8 times, for the 8 'nearest simulations'
                    apply_PBC(kernel)
                    apply_PBC(kernel_perpself)
                    apply_PBC(kernel_perpother)

                kernel *= 1e-7 # [J/Am²], 1e-7 is mu_0/4Pi
                kernel_perpself *= 1e-7 # [J/Am²], 1e-7 is mu_0/4Pi
                kernel_perpother *= 1e-7 # [J/Am²], 1e-7 is mu_0/4Pi
                self.kernel_unitcell_indices[y,x] = len(self.kernel_unitcell)
                self.kernel_unitcell.append(kernel)
                self.kernel_perpself_unitcell.append(kernel_perpself)
                self.kernel_perpother_unitcell.append(kernel_perpother)
        self.kernel_unitcell = xp.asarray(self.kernel_unitcell)
        self.kernel_perpself_unitcell = xp.asarray(self.kernel_perpself_unitcell)
        self.kernel_perpother_unitcell = xp.asarray(self.kernel_perpother_unitcell)

    def update(self):
        total_energy = xp.zeros_like(self.mm.m)
        if self.mm.USE_PERP_ENERGY: total_energy_perp = xp.zeros_like(self.mm.m)
        mmoment = self.mm.m*self.mm.moment
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                if (n := self.kernel_unitcell_indices[y,x]) < 0:
                    continue
                else:
                    partial_m = xp.zeros_like(self.mm.m)
                    partial_m[y::self.unitcell.y, x::self.unitcell.x] = self.mm.m[y::self.unitcell.y, x::self.unitcell.x]

                    kernel = self.kernel_unitcell[n,::-1,::-1]
                    total_energy += partial_m*signal.convolve2d(kernel, mmoment, mode='valid') # Could probably be done faster by only convolving the nonzero partial_m elements but this is already fast enough anyway and such slicing is also not free
                    if self.mm.USE_PERP_ENERGY:
                        kernel_perpself = self.kernel_perpself_unitcell[n,::-1,::-1]
                        total_energy_perp += partial_m*signal.convolve2d(kernel_perpself, mmoment, mode='valid') # NOTE: partial_m is not strictly necessary if 'perpendicular' does not necessarily mean '90° counterclockwise'
        self.E = self.prefactor*self.mm.moment*total_energy
        if self.mm.USE_PERP_ENERGY: self.E_perp = self.prefactor*self.mm.moment*total_energy_perp

    def update_single(self, index2D):
        #! Call this AFTER self.mm.m[index2D] has been updated!
        # First we get the x and y coordinates of magnet <i> in its unit cell
        y, x = index2D[0,0], index2D[1,0]
        x_unitcell = x.astype(int) % self.unitcell.x
        y_unitcell = y.astype(int) % self.unitcell.y
        # Test if there is actually a kernel
        n = self.kernel_unitcell_indices[y_unitcell, x_unitcell]
        if n < 0: return # Then there is no magnet there, so nothing happens
        # Multiply with the magnetization
        usefulkernel = self.kernel_unitcell[n,self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
        mmoment = self.mm.m*self.mm.moment
        interaction = self.prefactor*mmoment[y, x]*xp.multiply(mmoment, usefulkernel)
        self.E += 2*interaction
        self.E[y, x] *= -1 # This magnet switched, so all its interactions are inverted
        if self.mm.USE_PERP_ENERGY:
            usefulkernel_perp = self.kernel_perpother_unitcell[n,self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
            interaction_perp = self.prefactor*mmoment[y, x]*xp.multiply(mmoment, usefulkernel_perp)
            self.E_perp += 2*interaction_perp
            self.E_perp[y, x] *= -1 # NOTE: not strictly necessary if 'perpendicular' does not necessarily mean '90° counterclockwise'

    def update_multiple(self, indices2D):
        self.E[indices2D[0], indices2D[1]] *= -1
        mmoment = self.mm.m*self.mm.moment
        if self.mm.USE_PERP_ENERGY: self.E_perp[indices2D[0], indices2D[1]] *= -1 # NOTE: not strictly necessary if 'perpendicular' does not necessarily mean '90° counterclockwise'
        indices2D_unitcell_raveled = (indices2D[1] % self.unitcell.x) + (indices2D[0] % self.unitcell.y)*self.unitcell.x
        binned_unitcell_raveled = xp.bincount(indices2D_unitcell_raveled)
        for i in binned_unitcell_raveled.nonzero()[0]: # Iterate over the unitcell indices present in indices2D
            y_unitcell, x_unitcell = divmod(int(i), self.unitcell.x)
            if (n := self.kernel_unitcell_indices[y_unitcell, x_unitcell]) < 0: continue # This should never happen, but check anyway in case indices2D includes empty cells
            kernel = self.kernel_unitcell[n,:,:]
            if self.mm.USE_PERP_ENERGY: kernel_perp = self.kernel_perpother_unitcell[n,:,:]
            indices2D_here = indices2D[:,indices2D_unitcell_raveled == i]
            if indices2D_here.shape[1] > self.mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF:
                ### EITHER WE DO THIS (CONVOLUTION) (starts to be faster at approx. 40 simultaneous switches for 41x41 kernel, so especially for large T this is good)
                switched_field = xp.zeros_like(self.mm.m)
                switched_field[indices2D_here[0], indices2D_here[1]] = mmoment[indices2D_here[0], indices2D_here[1]]
                k = self.mm.params.REDUCED_KERNEL_SIZE
                kx, ky = min(k, self.mm.nx-1), min(k, self.mm.ny-1)
                usefulkernel = kernel[self.mm.ny-1-ky:self.mm.ny+ky, self.mm.nx-1-kx:self.mm.nx+kx] if k else kernel
                convolvedkernel = signal.convolve2d(switched_field, usefulkernel, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
                if self.mm.USE_PERP_ENERGY:
                    usefulkernel_perp = kernel_perp[self.mm.ny-1-ky:self.mm.ny+ky, self.mm.nx-1-kx:self.mm.nx+kx] if k else kernel_perp
                    convolvedkernel_perp = signal.convolve2d(switched_field, usefulkernel_perp, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
            else:
                ### OR WE DO THIS (BASICALLY self.update_single BUT SLIGHTLY PARALLEL AND SLIGHTLY NONPARALLEL) 
                convolvedkernel = xp.zeros_like(self.mm.m) # Still is convolved, just not in parallel
                if self.mm.USE_PERP_ENERGY: convolvedkernel_perp = xp.zeros_like(self.mm.m)
                for j in range(indices2D_here.shape[1]): # Here goes the manual convolution
                    y, x = indices2D_here[0,j], indices2D_here[1,j]
                    convolvedkernel += mmoment[y,x]*kernel[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
                    if self.mm.USE_PERP_ENERGY: convolvedkernel_perp += mmoment[y,x]*kernel_perp[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
            interaction = self.prefactor*xp.multiply(mmoment, convolvedkernel)
            self.E += 2*interaction
            if self.mm.USE_PERP_ENERGY:
                interaction_perp = self.prefactor*xp.multiply(mmoment, convolvedkernel_perp)
                self.E_perp += 2*interaction_perp

    @property
    def E_tot(self):
        return xp.sum(self.E)/2

    def get_NN_interaction(self):
        """ An APPROXIMATE value for the nearest-neighbor dipolar interaction energy.
            Approximate means that the highest value over all NNs over all kernels is returned, using the average magnetic moment.
            Hence, for a highly symmetric system like a uniform Ising system, this will be exact, but not for IP ASI.
        """
        # Approximation: use highest value from all nearest-neighbors in all kernels, and average magnetic moment
        largest = 0
        for kernel in self.kernel_unitcell:
            ny, nx = kernel.shape
            middle_y, middle_x = ny//2, nx//2
            NN = self.mm._get_nearest_neighbors()
            dy, dx = NN.shape
            dy, dx = dy//2, dx//2
            value = xp.max(xp.abs(kernel[middle_y-dy:middle_y+dy+1,middle_x-dx:middle_x+dx+1]*NN))
            if value > largest: largest = value
        return largest*self.prefactor*(self.mm.moment_avg**2)

    def set_NN_interaction(self, value):
        """ Sets `self.prefactor` such that the nearest-neighbor dipolar interaction energy is `value` Joules. """
        self.prefactor = value/self.get_NN_interaction()


class DiMonopolarEnergy(DipolarEnergy): # Original author: Diego De Gusem
    def __init__(self, prefactor=1, d=200e-9, small_d=200e-9):
        """ This DimonopolarEnergy class implements the interaction between the magnets of the simulation themselves using the monopole approximation.
            @param prefactor [float] (1): The relative strength of the monopole interaction.
            @param d [float] (200e-9): The distance between two monopoles in one magnet in meters.
            @param small_d [float] (200e-9): The distance between two monopoles in one magnet for the E_perp calculation.
                The most physical interpretation of this would likely be the width of the magnet.
        """
        self._prefactor = prefactor
        self.d = d # [m]
        self.small_d = small_d # [m]

    def _initialize(self):
        if not isinstance(self.d, np.ndarray):
            self.dist_unitcell = as_2D_array(self.d, (self.mm.unitcell.y, self.mm.unitcell.x))
        else:
            self.dist_unitcell = self.d
        self.dist = np.tile(self.dist_unitcell,(math.ceil(self.mm.ny / self.mm.unitcell.y), math.ceil(self.mm.nx / self.mm.unitcell.x)))
        self.dist_exact = self.dist[:self.mm.ny, :self.mm.nx]
        self.unitcell = self.mm.unitcell
        self.E = xp.zeros_like(self.mm.xx)
        self.charge = self.mm.moment / self.dist_exact  # Magnetic charge [Am]
        self.charge_sq = self.charge ** 2

        # Create a grid that is too large. This is needed to place every magnet in the unitcell in the centrum to convolve
        x_arange = xp.arange(-(self.mm.nx - 1), self.mm.nx + self.mm.unitcell.x)
        dx = self.mm.dx[x_arange % self.mm.nx]
        x_large = xp.zeros(x_arange.size)
        x_large[1:] = xp.cumsum(dx)[:-1]
        x_large -= x_large[self.mm.nx - 1]

        y_arange = xp.arange(-(self.mm.ny - 1), self.mm.ny + self.mm.unitcell.y)
        dy = self.mm.dy[y_arange % self.mm.ny]
        y_large = xp.zeros(y_arange.size)
        y_large[1:] = xp.cumsum(dy)[:-1]
        y_large -= y_large[self.mm.ny - 1]

        xx, yy = xp.meshgrid(x_large, y_large)
        if not isinstance(self.d, np.ndarray):
            self.dist_too_big = as_2D_array(self.d, xx.shape)
        else:
            self.dist_too_big = xp.zeros(xx.shape)
            for y in range(xx.shape[0]):
                for x in range(xx.shape[1]):
                    self.dist_too_big[y,x] = self.dist_unitcell[(y-(self.mm.ny-1))%self.mm.unitcell.y,(x-(self.mm.nx-1))%self.mm.unitcell.x]

        # Determine the angle of each magnet in the too large grid
        angle_unitcell = self.mm.original_angles[:self.unitcell.y,:self.unitcell.x]
        angles = xp.zeros(xx.shape)
        for y in range(xx.shape[0]):
            for x in range(xx.shape[1]):
                angles[y,x] = angle_unitcell[(y-(self.mm.ny-1))%self.mm.unitcell.y, (x-(self.mm.nx-1))%self.mm.unitcell.x]

        # Let us now find our monopoles, given the position and angle of the magnets we just found
        charges_xx = np.zeros((2 * self.mm.ny - 1 + self.mm.unitcell.y, 2 * self.mm.nx - 1 + self.mm.unitcell.x, 2))
        charges_yy = np.zeros((2 * self.mm.ny - 1 + self.mm.unitcell.y, 2 * self.mm.nx - 1 + self.mm.unitcell.x, 2))
        charges_xx[:, :, 0] = xx + self.dist_too_big / 2 * xp.cos(angles)
        charges_xx[:, :, 1] = xx - self.dist_too_big / 2 * xp.cos(angles)
        charges_yy[:, :, 0] = yy + self.dist_too_big / 2 * xp.sin(angles)
        charges_yy[:, :, 1] = yy - self.dist_too_big / 2 * xp.sin(angles)

        # DETERMINE THE POSITIONS WHEN ONE UNITCELL MAGNET HAS FLIPPED (1 d should be small)
        charges_xx_perp_self = np.zeros((2 * self.mm.ny - 1 + self.mm.unitcell.y, 2 * self.mm.nx - 1 + self.mm.unitcell.x, 2, self.mm.unitcell.x*self.mm.unitcell.y))
        charges_yy_perp_self = np.zeros((2 * self.mm.ny - 1 + self.mm.unitcell.y, 2 * self.mm.nx - 1 + self.mm.unitcell.x, 2, self.mm.unitcell.x*self.mm.unitcell.y))
        magnet_n = 0
        for y in range(self.mm.ny - 1, self.mm.ny - 1 + self.unitcell.y):
            for x in range(self.mm.nx - 1, self.mm.nx - 1 + self.unitcell.x):
                charges_xx_perp_self[:, :, :, magnet_n] = charges_xx
                charges_xx_perp_self[y, x, 0, magnet_n] = xx[y,x] + self.small_d / 2 * xp.cos(angles[y,x]+np.pi/2)
                charges_xx_perp_self[y, x, 1, magnet_n] = xx[y,x] - self.small_d / 2 * xp.cos(angles[y,x]+np.pi/2)
                charges_yy_perp_self[:, :, :, magnet_n] = charges_yy
                charges_yy_perp_self[y, x, 0, magnet_n] = yy[y,x] + self.small_d / 2 * xp.sin(angles[y,x]+np.pi/2)
                charges_yy_perp_self[y, x, 1, magnet_n] = yy[y,x] - self.small_d / 2 * xp.sin(angles[y,x]+np.pi/2)
                magnet_n += 1

        # DETERMINE THE POSITIONS WHEN ALL EXCEPT ONE UNITCELL MAGNET HAS FLIPPED (all d except 1 should be small)
        charges_xx_perp_other = np.zeros((2 * self.mm.ny - 1 + self.mm.unitcell.y,2 * self.mm.nx - 1 + self.mm.unitcell.x, 2, self.mm.unitcell.x * self.mm.unitcell.y))
        charges_yy_perp_other = np.zeros((2 * self.mm.ny - 1 + self.mm.unitcell.y,2 * self.mm.nx - 1 + self.mm.unitcell.x, 2,self.mm.unitcell.x * self.mm.unitcell.y))
        magnet_n = 0
        for y in range(self.mm.ny - 1, self.mm.ny - 1 + self.unitcell.y):
            for x in range(self.mm.nx - 1, self.mm.nx - 1 + self.unitcell.x):
                charges_xx_perp_other[:, :, 0, magnet_n] = xx + self.small_d / 2 * xp.cos(angles + np.pi / 2)
                charges_xx_perp_other[:, :, 1, magnet_n] = xx - self.small_d / 2 * xp.cos(angles + np.pi / 2)
                charges_xx_perp_other[y, x, 0, magnet_n] = xx[y, x] + self.dist_too_big[y, x] / 2 * xp.cos(angles[y, x] - np.pi / 2)
                charges_xx_perp_other[y, x, 1, magnet_n] = xx[y, x] - self.dist_too_big[y, x] / 2 * xp.cos(angles[y, x] - np.pi / 2)
                charges_yy_perp_other[:, :, 0, magnet_n] = yy + self.small_d / 2 * xp.sin(angles + np.pi / 2)
                charges_yy_perp_other[:, :, 1, magnet_n] = yy - self.small_d / 2 * xp.sin(angles + np.pi / 2)
                charges_yy_perp_other[y, x, 0, magnet_n] = yy[y, x] + self.small_d / 2 * xp.sin(angles[y, x] - np.pi / 2)
                charges_yy_perp_other[y, x, 1, magnet_n] = yy[y, x] - self.small_d / 2 * xp.sin(angles[y, x] - np.pi / 2)
                magnet_n += 1

        # For each monopole in the unit cell, determine the relative position to every other monopole in the system
        # There are 4 interactions for each magnet and there are self.mm.unitcell.x*self.mm.unitcell.y magnets in a unitcell
        rinv = np.zeros((2*self.mm.ny-1, 2*self.mm.nx-1, 4, self.mm.unitcell.x*self.mm.unitcell.y))
        # DETERMINE rinv WHEN ONE UNITCELL MAGNET HAS FLIPPED
        rinv_perp_self = np.zeros((2*self.mm.ny-1, 2*self.mm.nx-1, 4, self.mm.unitcell.x*self.mm.unitcell.y))
        # DETERMINE rinv WHEN ALL EXCEPT ONE UNITCELL MAGNET HAS FLIPPED
        rinv_perp_other = np.zeros((2*self.mm.ny-1, 2*self.mm.nx-1, 4, self.mm.unitcell.x*self.mm.unitcell.y))

        magnet_n = 0
        for y in range(self.mm.ny-1, self.mm.ny-1 + self.unitcell.y):
            for x in range(self.mm.nx-1, self.mm.nx-1 + self.unitcell.x):

                # Now calculate the distance from two monopoles in a magnet to all the others
                rrxNN = charges_xx[:,:,0] - charges_xx[y,x,0]
                rryNN = charges_yy[:,:,0] - charges_yy[y,x,0]
                rrxSS = charges_xx[:,:,1] - charges_xx[y,x,1]
                rrySS = charges_yy[:,:,1] - charges_yy[y,x,1]
                rrxNS = charges_xx[:,:,0] - charges_xx[y,x,1]  # Cross terms
                rryNS = charges_yy[:,:,0] - charges_yy[y,x,1]
                rrxSN = charges_xx[:,:,1] - charges_xx[y,x,0]
                rrySN = charges_yy[:,:,1] - charges_yy[y,x,0]

                rrNN_sq = (rrxNN**2 + rryNN**2).astype(np.float32)
                rrSS_sq = (rrxSS**2 + rrySS**2).astype(np.float32)
                rrNS_sq = (rrxNS**2 + rryNS**2).astype(np.float32)
                rrSN_sq = (rrxSN**2 + rrySN**2).astype(np.float32)

                # It is possible that artificial monopoles overlap and also give a 0, for this reason:
                rrNN_sq[rrNN_sq==0] = xp.inf
                rrSS_sq[rrSS_sq==0] = xp.inf
                rrNS_sq[rrNS_sq==0] = xp.inf
                rrSN_sq[rrSN_sq==0] = xp.inf

                rrNN_inv = rrNN_sq ** -0.5  # Due to the previous line, this is now never infinite
                rrSS_inv = rrSS_sq ** -0.5
                rrNS_inv = rrNS_sq ** -0.5
                rrSN_inv = rrSN_sq ** -0.5

                # Now take the correct slice, so this unit cell magnet is at its center (this is needed to convolve)
                rinv[:,:,0,magnet_n] = rrNN_inv[y-self.mm.ny+1:y+self.mm.ny, x-self.mm.nx+1:x+self.mm.nx]
                rinv[:,:,1,magnet_n] = rrSS_inv[y-self.mm.ny+1:y+self.mm.ny, x-self.mm.nx+1:x+self.mm.nx]
                rinv[:,:,2,magnet_n] = rrNS_inv[y-self.mm.ny+1:y+self.mm.ny, x-self.mm.nx+1:x+self.mm.nx]
                rinv[:,:,3,magnet_n] = rrSN_inv[y-self.mm.ny+1:y+self.mm.ny, x-self.mm.nx+1:x+self.mm.nx]

                # DETERMINE rinv WHEN ONE UNITCELL MAGNET HAS FLIPPED
                rrxNN_perp_self = charges_xx_perp_self[:, :, 0, magnet_n] - charges_xx_perp_self[y, x, 0, magnet_n]
                rryNN_perp_self = charges_yy_perp_self[:, :, 0, magnet_n] - charges_yy_perp_self[y, x, 0, magnet_n]
                rrxSS_perp_self = charges_xx_perp_self[:, :, 1, magnet_n] - charges_xx_perp_self[y, x, 1, magnet_n]
                rrySS_perp_self = charges_yy_perp_self[:, :, 1, magnet_n] - charges_yy_perp_self[y, x, 1, magnet_n]
                rrxNS_perp_self = charges_xx_perp_self[:, :, 0, magnet_n] - charges_xx_perp_self[y, x, 1, magnet_n]  # Cross terms
                rryNS_perp_self = charges_yy_perp_self[:, :, 0, magnet_n] - charges_yy_perp_self[y, x, 1, magnet_n]
                rrxSN_perp_self = charges_xx_perp_self[:, :, 1, magnet_n] - charges_xx_perp_self[y, x, 0, magnet_n]
                rrySN_perp_self = charges_yy_perp_self[:, :, 1, magnet_n] - charges_yy_perp_self[y, x, 0, magnet_n]

                rrNN_sq_perp_self = rrxNN_perp_self ** 2 + rryNN_perp_self ** 2
                rrSS_sq_perp_self = rrxSS_perp_self ** 2 + rrySS_perp_self ** 2
                rrNS_sq_perp_self = rrxNS_perp_self ** 2 + rryNS_perp_self ** 2
                rrSN_sq_perp_self = rrxSN_perp_self ** 2 + rrySN_perp_self ** 2

                rrNN_sq_perp_self[y, x] = xp.inf
                rrSS_sq_perp_self[y, x] = xp.inf
                rrNS_sq_perp_self[y, x] = xp.inf
                rrSN_sq_perp_self[y, x] = xp.inf

                rrNN_inv_perp_self = rrNN_sq_perp_self ** -0.5  # Due to the previous line, this is now never infinite
                rrSS_inv_perp_self = rrSS_sq_perp_self ** -0.5
                rrNS_inv_perp_self = rrNS_sq_perp_self ** -0.5
                rrSN_inv_perp_self = rrSN_sq_perp_self ** -0.5

                # Now take the correct slice, so this unit cell magnet is at its center (this is needed to convolve)
                rinv_perp_self[:, :, 0, magnet_n] = rrNN_inv_perp_self[y - self.mm.ny + 1:y + self.mm.ny, x - self.mm.nx + 1:x + self.mm.nx]
                rinv_perp_self[:, :, 1, magnet_n] = rrSS_inv_perp_self[y - self.mm.ny + 1:y + self.mm.ny, x - self.mm.nx + 1:x + self.mm.nx]
                rinv_perp_self[:, :, 2, magnet_n] = rrNS_inv_perp_self[y - self.mm.ny + 1:y + self.mm.ny, x - self.mm.nx + 1:x + self.mm.nx]
                rinv_perp_self[:, :, 3, magnet_n] = rrSN_inv_perp_self[y - self.mm.ny + 1:y + self.mm.ny, x - self.mm.nx + 1:x + self.mm.nx]

                # DETERMINE rinv WHEN ALL EXCEPT ONE UNITCELL MAGNET HAS FLIPPED
                rrxNN_perp_other = charges_xx_perp_other[:, :, 0, magnet_n] - charges_xx_perp_other[y, x, 0, magnet_n]
                rryNN_perp_other = charges_yy_perp_other[:, :, 0, magnet_n] - charges_yy_perp_other[y, x, 0, magnet_n]
                rrxSS_perp_other = charges_xx_perp_other[:, :, 1, magnet_n] - charges_xx_perp_other[y, x, 1, magnet_n]
                rrySS_perp_other = charges_yy_perp_other[:, :, 1, magnet_n] - charges_yy_perp_other[y, x, 1, magnet_n]
                rrxNS_perp_other = charges_xx_perp_other[:, :, 0, magnet_n] - charges_xx_perp_other[y, x, 1, magnet_n]  # Cross terms
                rryNS_perp_other = charges_yy_perp_other[:, :, 0, magnet_n] - charges_yy_perp_other[y, x, 1, magnet_n]
                rrxSN_perp_other = charges_xx_perp_other[:, :, 1, magnet_n] - charges_xx_perp_other[y, x, 0, magnet_n]
                rrySN_perp_other = charges_yy_perp_other[:, :, 1, magnet_n] - charges_yy_perp_other[y, x, 0, magnet_n]

                rrNN_sq_perp_other = rrxNN_perp_other ** 2 + rryNN_perp_other ** 2
                rrSS_sq_perp_other = rrxSS_perp_other ** 2 + rrySS_perp_other ** 2
                rrNS_sq_perp_other = rrxNS_perp_other ** 2 + rryNS_perp_other ** 2
                rrSN_sq_perp_other = rrxSN_perp_other ** 2 + rrySN_perp_other ** 2

                rrNN_sq_perp_other[y, x] = xp.inf
                rrSS_sq_perp_other[y, x] = xp.inf
                rrNS_sq_perp_other[y, x] = xp.inf
                rrSN_sq_perp_other[y, x] = xp.inf

                rrNN_inv_perp_other = rrNN_sq_perp_other ** -0.5  # Due to the previous line, this is now never infinite
                rrSS_inv_perp_other = rrSS_sq_perp_other ** -0.5
                rrNS_inv_perp_other = rrNS_sq_perp_other ** -0.5
                rrSN_inv_perp_other = rrSN_sq_perp_other ** -0.5

                # Now take the correct slice, so this unit cell magnet is at its center (this is needed to convolve)
                rinv_perp_other[:,:,0,magnet_n] = rrNN_inv_perp_other[y-self.mm.ny+1:y+self.mm.ny,x-self.mm.nx+1:x+self.mm.nx]
                rinv_perp_other[:, :, 1, magnet_n] = rrSS_inv_perp_other[y - self.mm.ny + 1:y + self.mm.ny,x - self.mm.nx + 1:x + self.mm.nx]
                rinv_perp_other[:, :, 2, magnet_n] = rrNS_inv_perp_other[y - self.mm.ny + 1:y + self.mm.ny,x - self.mm.nx + 1:x + self.mm.nx]
                rinv_perp_other[:, :, 3, magnet_n] = rrSN_inv_perp_other[y - self.mm.ny + 1:y + self.mm.ny,x - self.mm.nx + 1:x + self.mm.nx]

                magnet_n += 1

        # Now comes the part where we start splitting the different cells in the unit cells
        num_unitcells_x = 2 * math.ceil(self.mm.nx / self.unitcell.x) + 1
        num_unitcells_y = 2 * math.ceil(self.mm.ny / self.unitcell.y) + 1
        unitcell_occ = self.mm.occupation[:self.unitcell.y, :self.unitcell.x]
        toolargematrix_occ = xp.tile(unitcell_occ, (num_unitcells_y, num_unitcells_x))  # This is the maximum that we can ever need (this maximum occurs when the simulation does not cut off any unit cells)

        self.kernel_unitcell_indices = -xp.ones((self.unitcell.y, self.unitcell.x), dtype=int)  # unitcell (y,x) -> kernel (i)
        self.kernel_unitcell = []
        self.kernel_perpself_unitcell = []  # Initialize perp regardless of mm.USE_PERP_ENERGY, it only takes some unnecessary memory if it is not needed, initialization is very fast anyway
        self.kernel_perpother_unitcell = []
        magnet_n = 0
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                if unitcell_occ[y,x] == 0:
                    magnet_n += 1
                    continue  # Empty cell in the unit cell, so keep self.kernel_unitcell[y][x] equal to None

                # NORMAL KERNEL
                slice_startx = (self.unitcell.x - ((self.mm.nx - 1) % self.unitcell.x) + x) % self.unitcell.x  # Final % not strictly necessary because
                slice_starty = (self.unitcell.y - ((self.mm.ny - 1) % self.unitcell.y) + y) % self.unitcell.y  # toolargematrix_o{x,y} large enough anyway
                occ = toolargematrix_occ[slice_starty:slice_starty + 2 * self.mm.ny - 1, slice_startx:slice_startx + 2 * self.mm.nx - 1]
                kernelNN = rinv[:,:,0,magnet_n] * occ
                kernelSS = rinv[:,:,1,magnet_n] * occ
                kernelNS = -rinv[:,:,2,magnet_n] * occ
                kernelSN = -rinv[:,:,3,magnet_n] * occ
                kernel = kernelNN + kernelSS + kernelNS + kernelSN

                # PERPENDICULAR KERNEL SELF (so the magnet in the center of the kernel is 'perpendicular') (same formula as before)
                kernelNN_perp_self = rinv_perp_self[:, :, 0, magnet_n] * occ
                kernelSS_perp_self = rinv_perp_self[:, :, 1, magnet_n] * occ
                kernelNS_perp_self = -rinv_perp_self[:, :, 2, magnet_n] * occ
                kernelSN_perp_self = -rinv_perp_self[:, :, 3, magnet_n] * occ
                kernel_perpself = kernelNN_perp_self + kernelSS_perp_self + kernelNS_perp_self + kernelSN_perp_self

                # PERPENDICULAR KERNEL OTHER (so the magnet in the center is the only one that is still normal) (same formula asbefore)
                kernelNN_perp_other = rinv_perp_other[:, :, 0, magnet_n] * occ
                kernelSS_perp_other = rinv_perp_other[:, :, 1, magnet_n] * occ
                kernelNS_perp_other = -rinv_perp_other[:, :, 2, magnet_n] * occ
                kernelSN_perp_other = -rinv_perp_other[:, :, 3, magnet_n] * occ
                kernel_perpother = kernelNN_perp_other + kernelSS_perp_other + kernelNS_perp_other + kernelSN_perp_other

                magnet_n += 1
                ## PBC
                def apply_PBC(k):  # <k> is a DD kernel (2N+1,2N+1)-array
                    kopy = k.copy()
                    k[:, self.mm.nx:] += kopy[:, :self.mm.nx - 1]
                    k[self.mm.ny:, self.mm.nx:] += kopy[:self.mm.ny - 1, :self.mm.nx - 1]
                    k[self.mm.ny:, :] += kopy[:self.mm.ny - 1, :]
                    k[self.mm.ny:, :self.mm.nx - 1] += kopy[:self.mm.ny - 1, self.mm.nx:]
                    k[:, :self.mm.nx - 1] += kopy[:, self.mm.nx:]
                    k[:self.mm.ny - 1, :self.mm.nx - 1] += kopy[self.mm.ny:, self.mm.nx:]
                    k[:self.mm.ny - 1, :] += kopy[self.mm.ny:, :]
                    k[:self.mm.ny - 1, self.mm.nx:] += kopy[self.mm.ny:, :self.mm.nx - 1]

                if self.mm.PBC:  # Just copy the kernel 8 times, for the 8 'nearest simulations'
                    apply_PBC(kernel)
                    apply_PBC(kernel_perpself)
                    apply_PBC(kernel_perpother)

                kernel *= 1e-7  # [J/Am²], 1e-7 is mu_0/4Pi
                kernel_perpself *= 1e-7  # [J/Am²], 1e-7 is mu_0/4Pi
                kernel_perpother *= 1e-7  # [J/Am²], 1e-7 is mu_0/4Pi
                self.kernel_unitcell_indices[y, x] = len(self.kernel_unitcell)
                self.kernel_unitcell.append(kernel)
                self.kernel_perpself_unitcell.append(kernel_perpself)
                self.kernel_perpother_unitcell.append(kernel_perpother)
        self.kernel_unitcell = xp.asarray(self.kernel_unitcell)
        self.kernel_perpself_unitcell = xp.asarray(self.kernel_perpself_unitcell)
        self.kernel_perpother_unitcell = xp.asarray(self.kernel_perpother_unitcell)

    def update(self):
        total_energy = xp.zeros_like(self.mm.m)
        if self.mm.USE_PERP_ENERGY: total_energy_perp = xp.zeros_like(self.mm.m)
        for y in range(self.unitcell.y):
            for x in range(self.unitcell.x):
                if (n := self.kernel_unitcell_indices[y,x]) < 0:
                    continue
                else:
                    kernel = self.kernel_unitcell[n,::-1,::-1]
                    partial_m = xp.zeros_like(self.mm.m)
                    partial_m[y::self.unitcell.y, x::self.unitcell.x] = self.mm.m[y::self.unitcell.y, x::self.unitcell.x]
                    total_energy += partial_m*signal.convolve2d(kernel, self.mm.m, mode='valid')*self.charge_sq
                    if self.mm.USE_PERP_ENERGY:
                        kernel_perpself = self.kernel_perpself_unitcell[n,::-1,::-1]
                        total_energy_perp += partial_m * signal.convolve2d(kernel_perpself, self.mm.m, mode='valid')*self.charge_sq  # NOTE: partial_m is not strictly necessary if 'perependicular' does not necessarily mean '90° counterclockwise'
        self.E = self.prefactor * total_energy  # TODO: shouldn't we be multiplying with the moment of magnet 1 and moment of magnet 2 instead of (moment of magnet 1)²? (and fix this for all occurrences of _momentSq, I think _momentSq still stems from an era when _moment was a constant throughout the system.)
        if self.mm.USE_PERP_ENERGY: self.E_perp = self.prefactor * total_energy_perp

    def update_single(self, index2D):
        # First we get the x and y coordinates of magnet <i> in its unit cell
        y, x = index2D[0,0], index2D[1,0]
        x_unitcell = x.astype(int) % self.unitcell.x
        y_unitcell = y.astype(int) % self.unitcell.y
        # Test if there is actually a kernel
        n = self.kernel_unitcell_indices[y_unitcell, x_unitcell]
        if n < 0: return # Then there is no magnet there, so nothing happens
        # Multiply with the magnetization
        usefulkernel = self.kernel_unitcell[n,self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
        interaction = self.prefactor*self.mm.m[y,x]*xp.multiply(self.mm.m, usefulkernel)*self.charge_sq
        self.E += 2*interaction
        self.E[y, x] *= -1 # This magnet switched, so all its interactions are inverted
        if self.mm.USE_PERP_ENERGY:
            usefulkernel_perp = self.kernel_perpother_unitcell[n, self.mm.ny - 1 - y:2 * self.mm.ny - 1 - y, self.mm.nx - 1 - x:2 * self.mm.nx - 1 - x]
            interaction_perp = self.prefactor * self.mm.m[y, x] * xp.multiply(self.mm.m, usefulkernel_perp)*self.charge_sq
            self.E_perp += 2 * interaction_perp
            self.E_perp[y, x] *= -1  # NOTE: not strictly necessary if 'perpendicular' does not necessarily mean '90° counterclockwise'

    def update_multiple(self, indices2D):
        self.E[indices2D[0], indices2D[1]] *= -1
        indices2D_unitcell_raveled = (indices2D[1] % self.unitcell.x) + (indices2D[0] % self.unitcell.y)*self.unitcell.x
        binned_unitcell_raveled = xp.bincount(indices2D_unitcell_raveled)
        for i in binned_unitcell_raveled.nonzero()[0]: # Iterate over the unitcell indices present in indices2D
            y_unitcell, x_unitcell = divmod(int(i), self.unitcell.x)
            if (n := self.kernel_unitcell_indices[y_unitcell, x_unitcell]) < 0: continue # This should never happen, but check anyway in case indices2D includes empty cells
            kernel = self.kernel_unitcell[n,:,:]
            if self.mm.USE_PERP_ENERGY: kernel_perp = self.kernel_perpother_unitcell[n,:,:]
            indices2D_here = indices2D[:,indices2D_unitcell_raveled == i]
            if indices2D_here.shape[1] > self.mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF:
                ### EITHER WE DO THIS (CONVOLUTION) (starts to be better at approx. 40 simultaneous switches for 41x41 kernel, taking into account the need for complete recalculation every <something> steps, so especially for large T this is good)
                switched_field = xp.zeros_like(self.mm.m)
                switched_field[indices2D_here[0], indices2D_here[1]] = self.mm.m[indices2D_here[0], indices2D_here[1]]
                k = self.mm.params.REDUCED_KERNEL_SIZE
                kx, ky = min(k, self.mm.nx-1), min(k, self.mm.ny-1)
                usefulkernel = kernel[self.mm.ny-1-ky:self.mm.ny+ky, self.mm.nx-1-kx:self.mm.nx+kx] if k else kernel
                convolvedkernel = signal.convolve2d(switched_field, usefulkernel, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
                if self.mm.USE_PERP_ENERGY:
                    usefulkernel_perp = kernel_perp[self.mm.ny - 1 - ky:self.mm.ny + ky, self.mm.nx - 1 - kx:self.mm.nx + kx] if k else kernel_perp
                    convolvedkernel_perp = signal.convolve2d(switched_field, usefulkernel_perp, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
            else:
                ### OR WE DO THIS (BASICALLY self.update_single BUT SLIGHTLY PARALLEL AND SLIGHTLY NONPARALLEL)
                convolvedkernel = xp.zeros_like(self.mm.m) # Still is convolved, just not in parallel
                if self.mm.USE_PERP_ENERGY: convolvedkernel_perp = xp.zeros_like(self.mm.m)
                for j in range(indices2D_here.shape[1]): # Here goes the manual convolution
                    y, x = indices2D_here[0,j], indices2D_here[1,j]
                    convolvedkernel += self.mm.m[y,x]*kernel[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
                    if self.mm.USE_PERP_ENERGY: convolvedkernel_perp += self.mm.m[y,x]*kernel_perp[self.mm.ny-1-y:2*self.mm.ny-1-y,self.mm.nx-1-x:2*self.mm.nx-1-x]
            interaction = self.prefactor*xp.multiply(self.mm.m*self.charge_sq, convolvedkernel)
            self.E += 2*interaction
            if self.mm.USE_PERP_ENERGY:
                interaction_perp = self.prefactor * xp.multiply(self.mm.m*self.charge_sq, convolvedkernel_perp)
                self.E_perp += 2 * interaction_perp


class ExchangeEnergy(Energy):  # TODO: allow random variation in J, see https://stackoverflow.com/a/73398072 for this kind of convolution
    def __init__(self, J=1):
        self.J = J # [J]
    
    @property
    def J(self):
        return self._J
    
    @J.setter
    def J(self, value):
        self._J = value
        try: self.update()
        except AttributeError: pass # Not assigned to a Magnets object yet

    def _initialize(self):
        self.local_interaction = self.mm._get_nearest_neighbors()

    def update(self):
        if self.mm.in_plane: # Use the XY hamiltonian (but spin has fixed axis so model is still Ising-like)
            mx = self.mm.orientation[:,:,0]*self.mm.m
            my = self.mm.orientation[:,:,1]*self.mm.m
            sum_mx = signal.convolve2d(mx, self.local_interaction, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
            sum_my = signal.convolve2d(my, self.local_interaction, mode='same', boundary='wrap' if self.mm.PBC else 'fill')
            self.E = -self.J*(xp.multiply(sum_mx, mx) + xp.multiply(sum_my, my))
            if self.mm.USE_PERP_ENERGY: self.E_perp = -self.J*(xp.multiply(sum_my, mx) - xp.multiply(sum_mx, my)) # Just same but (mx,my)->(-my,mx) 90° rotation
        else: # Use Ising hamiltonian
            self.E = -self.J*xp.multiply(signal.convolve2d(self.mm.m, self.local_interaction, mode='same', boundary='wrap' if self.mm.PBC else 'fill'), self.mm.m)
            if self.mm.USE_PERP_ENERGY: self.E_perp = xp.zeros_like(self.mm.m)

    def update_single(self, index2D):
        self.update() # There are much faster ways of doing this, but this becomes difficult with PBC and in/out-of-plane

    def update_multiple(self, indices2D):
        self.update()

    @property
    def E_tot(self):
        return xp.sum(self.E)/2
