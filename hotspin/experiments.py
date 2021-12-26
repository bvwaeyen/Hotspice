import math
import os
# import warnings

import cupy as cp
import numpy as np

from abc import ABC, abstractmethod
# from cupyx.scipy import signal

from .core import Magnets
from .io import DataStream, Inputter, OutputReader, RandomDataStream, PerpFieldInputter, RegionalOutputReader


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
            self.results['all_states'] = state
            self.results['all_states_flat'] = state.reshape(-1)
            self.results['rank'] = np.linalg.matrix_rank(experiment.all_states)
        
        if save is not None:
            dirname = os.path.dirname(save)
            if not os.path.exists(dirname): os.makedirs(dirname)
            with open(save, 'wb') as f:
                np.save(f, self.all_states.get())
        
        return self.results['rank']


if __name__ == "__main__": # Run this file from the parent directory using cmd 'python -m hotspin.experiments'
    from .ASI import PinwheelASI
    mm = PinwheelASI(25, 1)
    datastream = RandomDataStream()
    inputter = PerpFieldInputter(datastream, magnitude=1, phi=math.pi/180*7, n=2)
    outputreader = RegionalOutputReader(5, 5, mm)
    experiment = KernelQualityExperiment(inputter, outputreader, mm)
    experiment.run(10, save='results/KernelQualityExperiment/TEST.npy')
    print(experiment.results['rank'])
    np.set_printoptions(threshold=np.inf)
    # print(np.load('results/KernelQualityExperiment/TEST.npy'))
