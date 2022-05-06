import math
import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import examplefunctions as ef
from context import hotspin


def main_kernelquality():
    mm = hotspin.ASI.PinwheelASI(25, 1e-6, T=300, V=3.5e-22, energies=(hotspin.DipolarEnergy(), hotspin.ZeemanEnergy())) # Same volume as in 'RC in ASI' paper
    datastream = hotspin.io.RandomBinaryDatastream()
    inputter = hotspin.io.PerpFieldInputter(datastream, magnitude=1e-4, angle=math.pi/180*7, n=2)
    outputreader = hotspin.io.RegionalOutputReader(2, 2, mm)
    experiment = hotspin.experiments.KernelQualityExperiment(inputter, outputreader, mm)
    values = 11

    filename = f'results/{type(experiment).__name__}/{type(inputter).__name__}/{type(outputreader).__name__}_{mm.nx}x{mm.ny}_out{outputreader.nx}x{outputreader.nx}_in{values}values.npy'
    
    experiment.run(values, save=filename, verbose=True)
    print(experiment.results['rank'])
    np.set_printoptions(threshold=np.inf)

    result = np.load(filename)
    plt.imshow(result, interpolation='nearest')
    plt.title(f'{mm.nx}x{mm.ny} {type(mm).__name__}\nField {inputter.magnitude*1e3} mT $\\rightarrow$ rank {np.linalg.matrix_rank(result)}')
    plt.xlabel('Output feature')
    plt.ylabel(f'Input # ({values} values each)')
    plt.savefig(f'{os.path.splitext(filename)[0]}.pdf')
    plt.show()

if __name__ == "__main__":
    main_kernelquality()
