import math
import os
# import warnings

import cupy as cp
import numpy as np

from abc import ABC, abstractmethod
# from cupyx.scipy import signal

from .core import DipolarEnergy, Magnets, ZeemanEnergy
from .io import DataStream, Inputter, OutputReader, RandomDataStream, PerpFieldInputter, RegionalOutputReader
from .plottools import show_m


class Experiment(ABC):
    def __init__(self, inputter: Inputter, outputreader: OutputReader, mm: Magnets):
        self.inputter = inputter
        self.outputreader = outputreader
        self.mm = mm
    
    @abstractmethod
    def run(self):
        ''' Runs the entire experiment and records all the useful data. '''


class KernelQualityExperiment(Experiment):
    def __init__(self, inputter, outputreader, mm):
        ''' Determines the output matrix rank for a given input signal and output readout.
            By using different inputters, one can determine kernel-quality K, generalization-capability G, ...
            By using this class on ASIs with different parameters, one can perform a sweep to determine the optimum.
        '''
        super().__init__(inputter, outputreader, mm)
        self.results = {}
    
    def run(self, input_bits_length, save=None, verbose=True):
        self.all_states = cp.zeros((self.outputreader.n, self.outputreader.n))
        for i in range(self.outputreader.n): # Input as many data-streams as there are output bits
            for j in range(input_bits_length):
                self.inputter.input_bit(self.mm)
                if verbose: print(f'Row {i}, bit {j}...')
            state = self.outputreader.read_state()
            self.all_states[i,:] = state.reshape(-1)
            self.results['all_states'] = self.all_states
            self.results['rank'] = np.linalg.matrix_rank(self.all_states)
            # print(self.mm.switches)
            # show_m(self.mm)
        
        if save:
            if not isinstance(save, str):
                save = f'results/{type(self).__name__}/{type(self.inputter).__name__}/{type(self.outputreader).__name__}_{mm.nx}x{mm.ny}_out{outputreader.nx}x{outputreader.nx}_in{bits}bits.npy'
            dirname = os.path.dirname(save)
            if not os.path.exists(dirname): os.makedirs(dirname)
            with open(save, 'wb') as f:
                np.save(f, self.all_states.get())
        
        return self.results['rank']


if __name__ == "__main__": # Run this file from the parent directory using cmd 'python -m hotspin.experiments'
    from .ASI import PinwheelASI
    mm = PinwheelASI(25, 5e-3, T=1, energies=(DipolarEnergy(), ZeemanEnergy()))
    datastream = RandomDataStream()
    inputter = PerpFieldInputter(datastream, magnitude=1, phi=math.pi/180*7, n=2)
    outputreader = RegionalOutputReader(5, 5, mm)
    experiment = KernelQualityExperiment(inputter, outputreader, mm)
    bits = 11

    filename = f'results/{type(experiment).__name__}/{type(inputter).__name__}/{type(outputreader).__name__}_{mm.nx}x{mm.ny}_out{outputreader.nx}x{outputreader.nx}_in{bits}bits.npy'
    
    experiment.run(bits, save=filename)
    print(experiment.results['rank'])
    np.set_printoptions(threshold=np.inf)

    import matplotlib.pyplot as plt
    result = np.load(filename)
    plt.imshow(result, interpolation='nearest')
    plt.title(f'{mm.nx}x{mm.ny} {type(mm).__name__}\nField {inputter.magnitude}T $\\rightarrow$ rank {np.linalg.matrix_rank(result)}')
    plt.xlabel('Output feature')
    plt.ylabel(f'Input # ({bits} bits each)')
    plt.savefig(f'{os.path.splitext(filename)[0]}.pdf')
    plt.show()
