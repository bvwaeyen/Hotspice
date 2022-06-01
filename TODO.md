# TODO

## Core functionality

1. High priority
    - [ ] If kernel is cut off, recalculate it after every *something* steps (requires a parameter to specify this *something*, and further investigation of the error made when cutting off the kernel, because the recent estimated calculation did not seem to correspond to reality very well)
    - [ ] Implement parallel Poisson-disc sampling
    - [ ] Improve accuracy of $E_b$ calculation by using a four-state approach. For OOP ASI this might not be relevant yet, but can become so if we would use input methods with an in-plane component on OOP ASIs.

2. Medium priority
    - [ ] Make unit tests
    - [ ] Implement commonly used metrics to compare with theory/experiment (average magnetization and dimensionless amplitude ratio  $\langle m^2 \rangle^2/\langle m^4 \rangle$, correlation e.g. by looking at nearest neighbors minimizing/maximizing dipolar interaction or by looking at the dot/cross(?) product between vectors, susceptibility, width of domains (related to correlation probably)...)
    - [ ] Sort out the AFM-ness and its normalization etc., or even better find new ways to analyze systems with AFM ground state (e.g. Néel vector?)
    - [ ] Can we come up with some sort of time measure for multiswitching Glauber dynamics?
    - [ ] Make a `get_summary()` method that returns a dict with important parameters of the simulation for easier creation of the "constants" in JSON data files (e.g. average `E_B`, `T`, `moment`...).

3. Low priority
    - [ ] Randomness: missing magnets and variation in `E_B` and `moment`. Probably no other randomness is possible since this would interfere with the unit cells (the variation in `E_B` and `moment` might already be tricky to take into account, idk). The only issue that can occur with missing spins, is probably that some multi-switching samples remain unused, but I could live with that.

## Analysis and tests

1. High priority
    - [ ] Calculate kernel-quality and task agnostic metrics as function of T, for different lattices with different interactions etc. First try easiest models and then go closer to experiment. Also I first have to determine suitable ranges for these metrics before I can put it on HPC ;)
        - [ ] To prevent using way too much calculation time, plot (NL, MC, CP) as function of the number of samples `len(experiment.u)` to determine when the metrics stabilize, and possibly also `k` if that is not too much work. This behavior might be situation-dependent.

2. Medium priority
    - [ ] How many MC steps do we need to reach a statistically stable state?
    - Test physical accuracy of Hotspin
        - [ ] Domain size of square and pinwheel? As a function of time? (is there a maximum size or do they keep growing)
        - [ ] 2D ferromagnet (Arrott-Belov? Curie-Weiss? Susceptibility/magnetization?)
        - [x] 2D Ising with long-range dipolar interactions ("Aging in a Two-Dimensional Ising Model with Dipolar Interactions")
            - [ ] Can we also check the 'aging' part of that paper?
    - [ ] Multi-switch: analyze influence of Q on physical behavior to see which values of Q are acceptable (0.01? 0.1? 1? 100???), but how to do this? In `FullASI` perhaps?
    - [x] Simulation speed (samples/sec) for different grid sizes?
        - [ ] Are certain nicely factorable grid sizes preferable? (all I notice now is that there is a dependence on whether or not the unitcells fit perfectly in the simulation area, but this is likely due to PBC)

3. Low priority
    - Test physical accuracy of Hotspin
        - [ ] Angle between 0° and 45° between SquareASI and PinwheelASI where AFM alignment changes to uniform and vice versa? Can be useful to compare to experiment

## Various smaller things that I will probably never come around to doing

- [ ] Improve relaxation algorithm for `Magnets.relax()`
- [ ] Improve `plottools.py`
  - [ ] Organize plotting functions better
  - [ ] Improve the Average class to make it more versatile and consistent between different ASIs
  - [ ] Function to plot field direction at every cell by taking into account Zeeman and dipolar field from all magnets
- [ ] Linear transformations (e.g. skewing or squeezing) can be implemented by acting on `xx` and `yy`, but this might not be so easy with the unit cells etc. (for plots: [plt imshow](https://matplotlib.org/stable/gallery/images_contours_and_fields/affine_image.html "Affine transform of an image for skewed geometries"))
