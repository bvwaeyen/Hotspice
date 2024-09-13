""" This file implements a clocking protocol in pinwheel ASI.
    Based on the paper
        Jensen, J. H., Strømberg, A., Breivik, I. et al. Clocked dynamics
        in artificial spin ice. Nature Communications, 15(1), 964, 2024.
"""
import numpy as np

import hotspice
import _example_plot_utils as epu
from _clocking_plot import plot_clocking


class BinaryListDatastream(hotspice.io.BinaryDatastream):
    """ Returns values from a given list. """
    def __init__(self, binary_list: list[int] = [0]):
        self.binary_list = binary_list
        self.len_list = len(binary_list)  # will be used often, no need to recalculate
        self.index = 0  # start at the beginning
        super().__init__()

    def get_next(self, n=1) -> np.ndarray:
        """ Returns next `n` values as xp.ndarray. Loops back to start at the end. """
        values = [self.binary_list[(self.index + i) % self.len_list] for i in range(n)]
        self.index += n
        return np.array(values)


class ClockingFieldInputter(hotspice.io.FieldInputter):
    def __init__(self, datastream: hotspice.io.BinaryDatastream, magnitude=1,
                 angle=0, spread=np.pi/8, n=2, frequency=1):
        """ Applies an external field at `angle+spread` rad for 0.5/frequency seconds,
            then at `angle-spread` rad for bit 0. It does the same +180deg for bit 1.
            This avoids avalanches by only affecting one sublattice at a time.
        """
        self.spread = spread
        super().__init__(datastream, magnitude=magnitude, angle=angle, n=n, frequency=frequency)

    def bit_to_angles(self, bit):
        if not bit:
            return (self.angle + self.spread, self.angle - self.spread)
        return (self.angle + self.spread + np.pi, self.angle - self.spread + np.pi)

    def input_single(self, mm: hotspice.ASI.IP_ASI, value: bool|int):
        if self.frequency and mm.params.UPDATE_SCHEME != hotspice.Scheme.NEEL:
            raise AttributeError("Can only use frequency if UPDATE_SCHEME is NEEL.")
        if not mm.in_plane:
            raise AttributeError("Can only use ClockingFieldInputter on in-plane ASI.")

        Zeeman: hotspice.ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None: mm.add_energy(Zeeman := hotspice.ZeemanEnergy(0, 0))
        angle1, angle2 = self.bit_to_angles(value)
        Zeeman.set_field(magnitude=self.magnitude, angle=angle1)
        mm.progress(t_max=0.5/self.frequency, MCsteps_max=self.n)
        Zeeman.set_field(magnitude=self.magnitude, angle=angle2)
        mm.progress(t_max=0.5/self.frequency, MCsteps_max=self.n)


def run(N=13, size: int = 101, magnitude=53e-3, E_B_std=0.05, m_perp_factor=0.4):
    ## Basic 101x101 system
    a = 248e-9 # Lattice spacing
    moment = 3e-16
    T = 1 # For deterministic switching
    E_B = hotspice.utils.eV_to_J(110)

    np.random.seed(2) # 2 gives a nice result.
    
    spread = np.pi/4 # We use 45deg instead of the 22.5deg used in the Clocking paper.
    bits = [None] + [1]*(N//2) + [0]*(N//2)
    datastream = BinaryListDatastream(bits)
    inputter = ClockingFieldInputter(datastream, magnitude=magnitude, spread=spread)
    lattice_angle = np.deg2rad(4) # Rotate all magnets by 4deg
    mm = hotspice.ASI.IP_Pinwheel(a, size, moment=moment, E_B=E_B, E_B_std=E_B_std,
                                  T=T, pattern="uniform", angle=lattice_angle, m_perp_factor=m_perp_factor)
    mm.params.UPDATE_SCHEME = hotspice.Scheme.NEEL # NEEL is best at low T and high E_B
    mm.add_energy(hotspice.ZeemanEnergy())

    states = []
    for bit in datastream.binary_list:
        if bit is not None: inputter.input(mm, values=[bit])
        states.append(np.copy(mm.m))
    
    ## Save
    hotspice.utils.save_results(parameters={"magnitude": magnitude, "E_B_std": E_B_std, "a": mm.a, "T": mm.T_avg, "PBC": mm.PBC, "m_perp_factor": m_perp_factor, "scheme": mm.params.UPDATE_SCHEME, "size": size, "N": N, "ASI_type": "IP_Pinwheel"},
                                data={"values": bits, "states": states})
    plot()


def plot(data_dir=None, show_domains=True):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = epu.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    
    ## Plot
    fig = plot_clocking(data, params, show_domains=show_domains)
    hotspice.utils.save_results(figures={'IP_Pinwheel_clocking': fig}, outdir=data_dir, copy_script=False)
    

if __name__ == "__main__":
    """ EXPECTED RUNTIME: ≈30s. """
    run(magnitude=53e-3, E_B_std=0.05, m_perp_factor=0.4)
    # epu.replot_all(plot)
