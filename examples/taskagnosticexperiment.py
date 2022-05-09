import math
import re

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import examplefunctions as ef
from context import hotspin


def to_dataframe(experiment=None, u=None, y=None):
    """ The experiment contains <u> and <y>, which are CuPy arrays.
        Can also pass <u> and <y> directly, but <experiment> takes precedence.
        <u> is a 1D array of length N, and is stored as such.
        <y> is a <NxL> array, where L is the number of output nodes.
        The dataframe contains the columns "u", "y0", "y1", ... "y<L-1>".
    """
    if experiment is not None:
        u, y = experiment.u, experiment.y
    elif u is None or y is None: # experiment is None, so we need both u and y arguments
        raise ValueError('Must pass either an Experiment(), or both u and y arrays.')
    u = cp.asarray(u).get()
    y = cp.asarray(y).get()

    df = pd.DataFrame()
    df["u"] = u
    for i in range(y.shape[1]): df[f"y{i}"] = y[:,i]
    return df

def load_dataframe(df):
    """ Returns the CuPy arrays <u> and <y> present in the dataframe. """
    u = cp.asarray(df["u"])
    pattern = re.compile(r"\Ay[0-9]+\Z") # Match 'y0', 'y1', ... (\A and \Z represent end and start of string, respectively)
    y_cols = [colname for colname in df if pattern.match(colname)]
    y = cp.asarray(df[y_cols])
    return u, y


def main_taskagnostic(T=300, verbose=False, **kwargs):
    size = kwargs.get('size', 25) # Edge length of simulation as a number of cells
    a = kwargs.get('a', 420e-9*math.sqrt(2)) # [m] Lattice spacing
    T = kwargs.get('T', 300) # [K] Room temperature
    E_b = kwargs.get('E_b', hotspin.Energy.eV_to_J(71)) # [J] energy barrier between stable states (realistic for islands in flatspin paper)
    ext_field = kwargs.get('H_max', 0.04) # [T] extremal magnitude of external field
    V = kwargs.get('V', 470e-9*170e-9*10e-9) # [m³] volume of the islands, 

    mm = hotspin.ASI.PinwheelASI(size, a, T=T, V=V, E_b=E_b,
        energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()), PBC=False, pattern='vortex', params=hotspin.SimParams(UPDATE_SCHEME='Glauber'))
    datastream = hotspin.io.RandomUniformDatastream(low=-1, high=1)
    inputter = hotspin.io.FieldInputter(datastream, magnitude=ext_field, angle=math.pi/180*7, n=2)
    outputreader = hotspin.io.RegionalOutputReader(5, 5, mm)
    experiment = hotspin.experiments.TaskAgnosticExperiment(inputter, outputreader, mm)
    experiment.run(N=100, verbose=verbose)

    print(experiment.results)
    plt.ioff()
    hotspin.utils.shell()
    NL = experiment.NL(local=True)
    MC = experiment.MC(local=True)
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(outputreader.inflate_flat_array(NL)[:,:,0].get())
    ax1.set_title("NL x")
    plt.colorbar(im1)
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(outputreader.inflate_flat_array(NL)[:,:,1].get())
    ax2.set_title("NL y")
    plt.colorbar(im2)
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(outputreader.inflate_flat_array(MC)[:,:,0].get())
    ax3.set_title("MC x")
    plt.colorbar(im3)
    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(outputreader.inflate_flat_array(MC)[:,:,1].get())
    ax4.set_title("MC y")
    plt.colorbar(im4)
    # print('Nonlinearity nonlocal:', experiment.NL(local=False))
    # print('Memory capacity nonlocal:', experiment.MC(local=False))
    # print('Complexity averaged:', experiment.CP(transposed=False))
    # print('Complexity transposed:', experiment.CP(transposed=True))
    # print('Stability:', experiment.S())
    print(experiment.results)
    plt.show()

if __name__ == "__main__":
    main_taskagnostic(T=100, verbose=True)
