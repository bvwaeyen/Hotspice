""" This file implements a clocking scheme in OOP Square ASI. """
import numpy as np

import hotspice
import _example_plot_utils as epu
from _clocking_plot import plot_clocking


def run(N: int = 13, magnitude: float = 0.003*9.5, E_B_std: float = 0.05, E_B: float = hotspice.utils.eV_to_J(60), moment=2.37e-16, a=230e-9, d=0, vacancy_fraction: float = 0., size: int = 50):
    mm = hotspice.ASI.OOP_Square(a, size, E_B=E_B, E_B_std=E_B_std, moment=moment, major_axis=d, minor_axis=d,
                                 energies=(hotspice.energies.DipolarEnergy(), hotspice.energies.ZeemanEnergy()),
                                 params=hotspice.SimParams(UPDATE_SCHEME="Néel"))
    vacancies = int(mm.n*vacancy_fraction)
    mm.occupation[np.random.randint(mm.ny, size=vacancies), np.random.randint(mm.nx, size=vacancies)] = 0
    inputter = hotspice.io.OOPSquareChessStepsInputter(hotspice.io.RandomBinaryDatastream(), magnitude=magnitude)
    
    states, domains = [], []
    values = []
    mm.initialize_m('AFM', angle=np.pi)
    for i in range(N):
        if i == 0: values.append(None)
        else: values.append(inputter.input(mm, values=(i < (N//2 + 1)))[0])
        states.append(np.where(mm.occupation == 0, np.nan, mm.m))
        domains.append(np.where(mm.occupation == 0, np.nan, mm.get_domains()))
        
    ## Save
    hotspice.utils.save_results(parameters={"magnitude": magnitude, "E_B_std": E_B_std, "E_B": E_B, "vacancy_fraction": vacancy_fraction, "vacancies": vacancies, "a": mm.a, "d": d, "T": mm.T_avg, "PBC": mm.PBC, "scheme": mm.params.UPDATE_SCHEME, "size": size, "moment": mm.moment_avg, "N": N, "ASI_type": "OOP_Square"},
                                data={"values": values, "states": states, "domains": domains})
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
    hotspice.utils.save_results(figures={'OOP_Square_clocking': fig}, outdir=data_dir, copy_script=False)
    
if __name__ == "__main__":
    """ EXPECTED RUNTIME: ≈30s. """
    # run(E_B_std=0.1, E_B=hotspice.utils.eV_to_J(1), moment=1.6e-16, magnitude=0.003, vacancy_fraction=0.01) # Here, vacancies are important.
    # run(E_B_std=0.01, E_B=hotspice.utils.eV_to_J(60), moment=2.37e-16, magnitude=0.0615, vacancy_fraction=0.01)
    run(E_B_std=0.05, E_B=hotspice.utils.eV_to_J(60), moment=2.37e-16, magnitude=0.048, a=200e-9, d=170e-9)
    # epu.replot_all(plot)
    