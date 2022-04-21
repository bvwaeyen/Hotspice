import math
import os
import warnings

import cupy as cp
import numpy as np

from abc import ABC, abstractmethod
# from cupyx.scipy import signal

from .core import Magnets, DipolarEnergy, ZeemanEnergy
from .io import Inputter, OutputReader, RandomBinaryDatastream, FieldInputter, PerpFieldInputter, RandomUniformDatastream, RegionalOutputReader
from .utils import strided, shell


class Experiment(ABC):
    def __init__(self, inputter: Inputter, outputreader: OutputReader, mm: Magnets):
        self.inputter = inputter
        self.outputreader = outputreader
        self.mm = mm
        self.results = {} # General-purpose dict to store simulation results in
    
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
    
    def run(self, input_length: int, save=None, verbose=False):
        ''' @param input_length [int]: the number of times inputter.input_single() is called,
                before every recording of the output state.
        '''
        self.all_states = cp.zeros((self.outputreader.n, self.outputreader.n))
        for i in range(self.outputreader.n): # To get a square matrix, tecord the state as many times as there are output values by the outputreader
            for j in range(input_length):
                self.inputter.input_single(self.mm)
                if verbose: print(f'Row {i+1}/{self.outputreader.n}, value {j+1}/{input_length}...')
            state = self.outputreader.read_state()
            self.all_states[i,:] = state.reshape(-1)
            self.results['all_states'] = self.all_states
            self.results['rank'] = np.linalg.matrix_rank(self.all_states)
        
        if save:
            if not isinstance(save, str):
                save = f'results/{type(self).__name__}/{type(self.inputter).__name__}/{type(self.outputreader).__name__}_{self.mm.nx}x{self.mm.ny}_out{self.outputreader.nx}x{self.outputreader.nx}_in{input_length}values.npy'
            dirname = os.path.dirname(save)
            if not os.path.exists(dirname): os.makedirs(dirname)
            with open(save, 'wb') as f:
                np.save(f, self.all_states.get())
        
        return self.results['rank']


###############################################################################
def main_kernelquality():
    from .ASI import PinwheelASI
    mm = PinwheelASI(25, 1e-6, T=300, energies=(DipolarEnergy(), ZeemanEnergy()))
    datastream = RandomBinaryDatastream()
    inputter = PerpFieldInputter(datastream, magnitude=1, angle=math.pi/180*7, n=2)
    outputreader = RegionalOutputReader(5, 5, mm)
    experiment = KernelQualityExperiment(inputter, outputreader, mm)
    values = 11

    filename = f'results/{type(experiment).__name__}/{type(inputter).__name__}/{type(outputreader).__name__}_{mm.nx}x{mm.ny}_out{outputreader.nx}x{outputreader.nx}_in{values}values.npy'
    
    experiment.run(values, save=filename, verbose=True)
    print(experiment.results['rank'])
    np.set_printoptions(threshold=np.inf)

    import matplotlib.pyplot as plt
    result = np.load(filename)
    plt.imshow(result, interpolation='nearest')
    plt.title(f'{mm.nx}x{mm.ny} {type(mm).__name__}\nField {inputter.magnitude*1e3} mT $\\rightarrow$ rank {np.linalg.matrix_rank(result)}')
    plt.xlabel('Output feature')
    plt.ylabel(f'Input # ({values} values each)')
    plt.savefig(f'{os.path.splitext(filename)[0]}.pdf')
    plt.show()

if __name__ == "__main__": # Run this file from the parent directory using cmd 'python -m hotspin.experiments'
    main_kernelquality()
