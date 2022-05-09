import math
import os
import warnings

import cupy as cp
import numpy as np
import statsmodels.api as sm

from abc import ABC, abstractmethod

from .core import Magnets, DipolarEnergy, ZeemanEnergy
from .io import Inputter, OutputReader, RandomBinaryDatastream, FieldInputter, PerpFieldInputter, RandomUniformDatastream, RegionalOutputReader
from .plottools import init_interactive, init_fonts, show_m
from .utils import R_squared, strided


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


class TaskAgnosticExperiment(Experiment):
    def __init__(self, inputter, outputreader, mm):
        ''' Follows the paper
                J. Love, J. Mulkers, G. Bourianoff, J. Leliaert, and K. Everschor-Sitte. Task agnostic
                metrics for reservoir computing. arXiv preprint arXiv:2108.01512, 2021.
            to implement task-agnostic metrics for reservoir computing using a single random input signal.
        '''
        super().__init__(inputter, outputreader, mm)
        self.k = None

    def run(self, N=1000, k=10, verbose=False):
        ''' @param N [int]: The total number of iterations performed (each iteration consists of <self.inputter.n> MC steps).
            @param k [int]: The number of iterations back in time that are used to train the current time step.
        '''
        if N <= k: raise ValueError(f"Number of iterations N={N} must be larger than k={k}.")
        self.k = k
        self.mm.relax()
        if verbose:
            init_fonts()
            init_interactive()
            fig = None
        # First, we run the simulation for <N> steps where each step consists of <inputter.n> full Monte Carlo steps.
        self.u = cp.zeros(N) # Inputs
        self.y = cp.zeros((N, self.outputreader.n)) # Outputs
        if verbose: print(f'[0/{N}] Running TaskAgnosticExperiment...')
        for i in range(N):
            self.u[i] = self.inputter.input_single(self.mm)
            self.y[i,:] = self.outputreader.read_state().reshape(-1)
            if verbose:
                # print(self.u[i])
                if (i + 1) % 10**(math.floor(math.log10(N))-1) == 0 or i == 0: 
                    print(f'[{i+1}/{N}] {self.mm.switches}/{self.mm.attempted_switches} switching attempts successful ({self.mm.MCsteps:.2f} MC steps).')
                fig = show_m(self.mm, figure=fig)
        # Then, we use the recorded information to perform some funni calculations
        self.results['NL'] = self.NL(k=k)
        self.results['MC'] = self.MC(k=k)
        self.results['CP'] = self.CP()
        self.results['S'] = self.S()
    

    def NL(self, k=None, local=False, test_fraction=1/4, verbose=False): # Non-linearity
        ''' Returns a float (local=False) or a CuPy array (local=True) representing the nonlinearity,
            either globally or locally depending on <local>. For this, an estimator for the current readout state is
            trained which for any time instant has access to the <k> most recent inputs (including the present one).
        '''
        if verbose: print(f"Calculating NL (local={local})...")
        if k is None: k = 10 if self.k is None else self.k
        if self.u.size <= k: raise ValueError("NL: Number of iterations must be larger than k.")
        if self.u.size < k + 10: warnings.warn(f"NL: k={k} might be tiny touch too big to get a good train/test split.", stacklevel=2) # Just a wet finger estimate

        train_test_cutoff = math.ceil(self.u.size*(1-test_fraction))
        u_train, u_test = cp.split(self.u, [train_test_cutoff])
        y_train, y_test = cp.split(self.y, [train_test_cutoff]) # Need to put train_test_cutoff in iterable for correct behavior


        u_strided = strided(self.u, k) # List of last <k> entries in <u> for each time step (result: N x k array)
        u_train_strided, u_test_strided = np.split(u_strided, [train_test_cutoff])
        # u_train_strided, y_train = u_train_strided[k-1:], y_train[k-1:] # First <k-1> rows don't have <k> previous entries yet to use as 'samples'
        Rsq = cp.empty(self.outputreader.n) # R² correlation coefficient
        for j in range(self.outputreader.n): # Train an estimator for the j-th output feature
            model = sm.OLS(y_train[:,j].get(), u_train_strided.get(), missing='drop') # missing='drop' ignores samples with NaNs (i.e. the first k-1 samples)
            results = model.fit()
            y_hat_test = results.predict(u_test_strided.get())
            sigma_y_hat, sigma_y = np.var(y_hat_test), np.var(y_test[:,j])
            if sigma_y_hat == 0: # (highly unlikely) predicting constant value, so R² depends on whether this constant is the average or not
                Rsq[j] = float(cp.mean(y_hat_test) == cp.mean(y_test[:,j]))
            elif sigma_y == 0: # Constant readout, so NL should logically be 0, so Rsq = 1-NL = 1
                Rsq[j] = 1.
            else: # Rsq is 1 for very predictable 
                Rsq[j] = R_squared(y_hat_test, y_test[:,j]) # Same as np.corrcoef(y_hat_test, y_test[:,j])[0,1]**2

        # Handle <local> True or False and return appropriately
        if local:
            return 1 - Rsq
        else:
            return float(1 - cp.mean(Rsq))
    
    def MC(self, k=None, local=False, test_fraction=1/4, verbose=False): # Memory capacity
        ''' Returns a float (local=False) or a CuPy array (local=True) representing the memory capacity,
            either globally or locally depending on <local>. For this, an estimator is trained which
            for any time instant attempts to predict the previous <k> inputs based on the current readout state.
        '''
        if verbose: print(f"Calculating MC (local={local})...")
        if k is None: k = 10 if self.k is None else self.k
        if self.u.size <= k: raise ValueError("Number of iterations must be larger than k.")
        # if self.u.size < k + 10: warnings.warn(f"k={k} might be tiny touch too big to get a good train/test split.", stacklevel=2) # Just a wet finger estimate

        train_test_cutoff = math.ceil(self.u.size*(1-test_fraction))
        u_train, u_test = cp.split(self.u, [train_test_cutoff])
        y_train, y_test = cp.split(self.y, [train_test_cutoff]) # Need to put train_test_cutoff in iterable for correct behavior

        if u_test.size < k: warnings.warn(f"MC: k={k} is too large for the currently stored results (size {self.u.size}), possibly causing issues with train/test split.", stacklevel=2)
        Rsq = cp.empty(k) # R² correlation coefficient
        weights = cp.zeros((k, self.outputreader.n))
        for j in range(1, k+1): # Train an estimator for the input j iterations ago
            # This one is a mess of indices, but at least we don't need striding like for non-linearity
            results = sm.OLS(u_train[k-j:-j].get(), y_train[k:,:].get(), missing='drop').fit() # missing='drop' ignores samples with NaNs (i.e. the first k-1 samples)
            weights[j-1,:] = cp.asarray(results.params)
            u_hat_test = results.predict(y_test[j:,:].get()) # Start at y_test[j] to prevent leakage between train/test
            Rsq[j-1] = R_squared(u_hat_test, u_test[:-j])
        Rsq[cp.isnan(Rsq)] = 0 # If some std=0, then it is constant so R² actually gives 0

        # Handle <local> True or False and return appropriately
        if local:
            # TODO: I think some normalization is wrong here (result is all near 0.5, not full range between 0 and 1 as in paper)
            weights = weights - cp.min(weights, axis=1).reshape(-1, 1) # Minimum value becomes 0
            weights = weights / cp.max(weights, axis=1).reshape(-1, 1) # Maximum value becomes 1
            weights_avg = cp.mean(weights, axis=0) # Average across 'time'
            return weights_avg
        else:
            return float(cp.sum(Rsq))

    def CP(self, transposed=False): # Complexity
        ''' Calculates the complexity: i.e. the effective rank of the kernel matrix as used in KernelQualityExperiment.
            The default method used here is to simply average the complexity of recent observations over time.
            Usually, transposed=True gives (slightly) higher values than for transposed=False.
            NOTE: never compare results from transposed=True with those for transposed=False.
                @param transposed [bool] (False): If True, CP is the rank of <self.y> multiplied with <self.y.T>.
                    If False, it is the average of all consecutive square matrices in <self.y> along the time axis.
        '''
        if self.y.shape[0] < self.y.shape[1]: # Less rows than columns, so rank can at most be the number of rows
            warnings.warn(f"Not enough recorded iterations to reliably calculate complexity (shape {self.y.shape} has less rows than columns)." , stacklevel=2)
        
        if transposed:
            # Multiply with transposed to get a square matrix of size <self.outputreader.n> (suggested by J. Leliaert)
            squarematrix = cp.matmul(self.y.T, self.y)
            rank = cp.linalg.matrix_rank(squarematrix)
        else:
            ranks = cp.asarray([cp.linalg.matrix_rank(self.y[i:i+self.y.shape[1],:]) for i in range(max(1, self.y.shape[0]-self.y.shape[1]+1))])
            rank = cp.mean(ranks)
        return int(rank)/self.outputreader.n
    
    def S(self): # Stability
        ''' Calculates the stability: i.e. how similar the initial and final states are, after relaxation.
            NOTE: This function is not normalized. #TODO
            NOTE: It can be advantageous to define a different metric for stability if the entire magnetization profile
                  is known, since this function only has access to the readout nodes, not the whole self.m array.
        '''
        self.mm.relax()
        final_readout = self.outputreader.read_state().reshape(-1)
        initial_readout = self.y[0,:] # Assuming this was relaxed first
        return float(cp.sum((initial_readout - final_readout)**2))
