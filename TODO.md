# TODO

## Sweeps and metrics

At this moment, running a sweep (on multiple GPUs/CPUs) requires several commands to be run with(in) the correct directories. I will resolve these things in the following manner:

- Sweep:
  - A specific sweep with specific parameter ranges etc. is defined in a sweep file (like `SweepKQ_RC_ASI.py` etc.), but these files are at the moment a bit messy and contain some things which do not change when making a different sweep file. Would it be possible to move as much of this unchanging code over to `ParallelJobs.py` or other places (e.g. creating the summary and (trying to) plot it)? Probably not (due to the multiple CPUs etc.), so those parts which remain unchanged should be re-organized to appear at the end of the file in an `if __name__ == "__main__":` block.
- Plots:
  - Add a staticmethod to `Sweep()`, which plots (either in 1D `plt.plot` or 2D `plt.imshow`) some metrics (i.e. columns from a summarized dataframe, or combinations thereof).
  - Put a call to plot with default arguments at the end of `ParallelJobs.py`, or we could put it in the sweep definition script to run when it is called without arguments (but with an 'outdir' argument specifying the location of the summarized file)
  - Note that we do not care about spatially resolved metrics, as these should not be plotted in a sweep but rather for a specific parameter combination.
- Single Dynamical Node (SDN) and Rotating Neurons Reservoir (RNR): create a new `Datastream` which correctly applies the mask etc. as in the SDN procedure, a similar RNR protocol can be created as well.

## Core functionality

1. High priority
    - [ ] Improve the `RegionalOutputReader` etc.
        - [ ] Make the state a 1D array, because the 2D-ness is quite restrictive and for some ASIs causes unnecessary constant zeroes which can simply be removed in a 1D array.
        - [x] Make a custom `OutputReader` for the OOP system which applies an AFM mask to distinguish between the two degenerate ground states
    - [ ] Implement parallel Poisson-disc sampling
    - [ ] Improve accuracy of $E_b$ calculation by using a four-state approach. For OOP ASI this might not be relevant yet, but can become so if we would use input methods with an in-plane component on OOP ASIs.

2. Medium priority
    - [ ] Make unit tests
    - [ ] If kernel is cut off, recalculate it after a certain amount of steps
    - [ ] Implement commonly used metrics to compare with theory/experiment (average magnetization and dimensionless amplitude ratio  $\langle m^2 \rangle^2/\langle m^4 \rangle$, correlation e.g. by looking at nearest neighbors minimizing/maximizing dipolar interaction or by looking at the dot/cross(?) product between vectors, susceptibility, width of domains (related to correlation probably)...)
    - [ ] Sort out the AFM-ness and its normalization etc., or even better find new ways to analyze systems with AFM ground state (e.g. Néel vector?)
    - [ ] Can we come up with some sort of time measure for multiswitching Glauber dynamics?
    - [ ] Make a `get_summary()` method that returns a dict with important parameters of the simulation for easier creation of the "constants" in JSON data files (e.g. average `E_B`, `T`, `moment`...).

3. Low priority
    - [ ] Could sparse matrices cause some speedup anywhere, e.g. in `DipolarEnergy`?
    - [ ] Create a frustrated OOP ASI (e.g. hexagonal close packed, equilateral Cairo...)
    - [ ] Randomly missing magnets. The only foreseeable issue that could occur with missing spins, is that some `Magnets().select()` samples remain unused, but I could live with that.
        - [ ] An extension of this randomness could be to generate an 'ensemble' of systems, which could be calculated efficiently in a parallel manner on GPU by extending arrays into a third dimension, and then using advanced indexing or some `stride_tricks` to manipulate each 2D slice differently while still being parallel. The issue with this is that this will be a lot of work where not a single slice may be wrong, and that a lot of code will have to be refactored to work with this extra dimension. Currently it seems that only `E_B`, `moment` and `T` should be moved into the 3rd dimension. But selecting magnets will become problematic, as for Glauber this will result in different number of samples in each element of the ensemble (so the indices can no longer be a rectangular array unless padded with e.g. -1), and for Néel this will require an argmin along only two of the three axes. The question is whether this will use more of the GPU at once, or whether this excessive indexing would slow things down more than we gain. There is only one way to find out, I guess, since the proof is in the pudding...

## Analysis and tests

1. High priority
    - [ ] Compare with other models
        - [ ] "Reservoir Computing in ASI": Pinwheel ASI with squinting
    - [ ] Calculate kernel-quality and task agnostic metrics as function of T, for different lattices with different interactions etc. First try easiest models and then go closer to experiment. Also I first have to determine suitable ranges for these metrics before I can put it on HPC ;)
        - [ ] To prevent using way too much calculation time, plot (NL, MC, CP) as function of the number of samples `len(experiment.u)` to determine when the metrics stabilize, and possibly also `k` if that is not too much work. This behavior might be situation-dependent.

2. Medium priority
    - [ ] How many MC steps do we need to reach a statistically stable state?
    - Test physical accuracy of Hotspin
        - [ ] Domain size of square and pinwheel? As a function of time? (is there a maximum size or do they keep growing)
        - [ ] 2D ferromagnet (Arrott-Belov? Curie-Weiss? Susceptibility/magnetization?)
        - [x] 2D Ising with long-range dipolar interactions ("Aging in a Two-Dimensional Ising Model with Dipolar Interactions")
            - [ ] Can we also check the 'aging' part of that paper?
    - [ ] Multi-switch: analyze influence of Q on physical behavior to see which values of Q are acceptable (0.01? 0.1? 1? 100???), but how to do this? In `OOP_Square` perhaps?
    - [x] Simulation speed (samples/sec) for different grid sizes?
        - [ ] Are certain nicely factorable grid sizes preferable? (all I notice now is that there is a dependence on whether or not the unitcells fit perfectly in the simulation area, but this is likely due to PBC)

3. Low priority
    - Test physical accuracy of Hotspin
        - [ ] Angle between 0° and 45° between IP_Square and IP_Pinwheel where AFM alignment changes to uniform and vice versa? Can be useful to compare to experiment

## Various smaller things that I will probably never come around to doing

- [ ] Improve relaxation algorithm for `Magnets.relax()`
- [ ] Improve `plottools.py`
  - [ ] Organize plotting functions better
  - [ ] Improve the Average class to make it more versatile and consistent between different ASIs, for example by actually calculating the field of the neighbors at each cell (but this might be quite computationally expensive, which is not an issue for normal plots but for animations this might not be ideal).
  - [ ] Function to plot field direction at every cell by taking into account Zeeman and dipolar field from all magnets
- [ ] Linear transformations (e.g. skewing or squeezing) can be implemented by acting on `xx` and `yy`, but this might not be so easy with the unit cells etc. (for plots: [plt imshow](https://matplotlib.org/stable/gallery/images_contours_and_fields/affine_image.html "Affine transform of an image for skewed geometries"))
