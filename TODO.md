# TODO

## Core functionality

1. High priority
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
