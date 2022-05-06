import math

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import examplefunctions as ef
from context import hotspin


def main_taskagnostic():
    mm = hotspin.ASI.PinwheelASI(25, 1e-6, T=100, energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy()), PBC=False, pattern='vortex')
    datastream = hotspin.io.RandomUniformDatastream(low=-1, high=1)
    inputter = hotspin.io.FieldInputter(datastream, magnitude=3e-5, angle=math.pi/180*7, n=2)
    outputreader = hotspin.io.RegionalOutputReader(5, 5, mm)
    experiment = hotspin.experiments.TaskAgnosticExperiment(inputter, outputreader, mm)
    experiment.run(N=100, verbose=True)
    print(experiment.results)
    hotspin.utils.shell()
    plt.ioff()
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
    main_taskagnostic()
