# TODO

## Sweeps and metrics

- Sweep:
  - Put the summarizing and plotting code in `ParallelJobs.py` instead of the sweep file (perhaps rename to `ParallelSweep.py` to better reflect that it is specifically for sweeps, or make a new file that is indeed purely for sweeps)
- Plots:
  - Plot spatially resolved metrics using a separate `TaskAgnosticExperiment().plot()` method.
- Single Dynamical Node (SDN) and Rotating Neurons Reservoir (RNR): create a new `Datastream` which correctly applies the mask etc. as in the SDN procedure, a similar RNR protocol can be created as well.

## Core functionality

1. High priority
    - [ ] Take a look at the `E_B` calculation for Néel, as the current method might not be entirely correct after all.

2. Medium priority
    - [ ] Create some sort of GUI that takes an ASI, inputter and datastream and shows what happens, allows applying individual bits etc.
    - [ ] In-plane systems could benefit from a better, angle-dependent, `E_barrier` calculation. Currently, the effective energy barrier in the presence of a dipolar or Zeeman interaction is crudely approximated as something like `E_B - switch_energy/2`. There are two ways this can be improved, both of whose effect should be investigated to see if they are worth the trouble.
        - [ ] The first way is to improve this formula, by using the actual minima and maxima and calculating the effective barrier from them. If I recall correctly, this would not significantly affect the effective barrier as compared to how it is calculated now.
        - [ ] The second way is to make the `DipolarEnergy` angular dependent, and combining this with the `ZeemanEnergy`'s inherent angular dependence, to create a better effective barrier by taking into account the offset angles of all the sines making up the energy landscape. This should not be too hard to do, but will require a completely different calculation which will certainly affect performance (because we need a separate dipolar kernel for both the x- and y-axes, so everything will take twice as long to calculate).
    - [ ] Improve the output saving with e.g. a `FullOutputReader` etc.
        - [ ] For `TaskAgnosticExperiment`: use the full `mm.m` in `initial_state` and `final_state` for a better `S` calculation (instead of `outputreader.read_state()`).
        - [ ] For `FullOutputReader`, perhaps we could save the full state as bits (`(mm.m + 1)/2`) (with encoding `np.packbits()` and `np.unpackbits()` to `uint8` to save space), but further compression might not be easy because we use JSON.
        - [ ] To 'squint', we could save the full state by default in the files, and perhaps only when processing them (`calculate_all` etc.) we choose an `OutputReader` (and thus the squinting level). This will make us save more data, but with the compression discussed above this might not be too catastrophic (now we need `float` as we are not sure if the readout will be a whole number, which takes a lot of space).
    - [ ] Make unit tests
    - [ ] Implement commonly used metrics to compare with theory/experiment (average magnetization and dimensionless amplitude ratio  $\langle m^2 \rangle^2/\langle m^4 \rangle$, correlation e.g. by looking at nearest neighbors minimizing/maximizing dipolar interaction or by looking at the dot/cross(?) product between vectors, susceptibility, width of domains (related to correlation probably)...)
    - [ ] Sort out the AFM-ness and its normalization etc., or even better find new ways to analyze systems with AFM ground state (e.g. Néel vector?)
    - [ ] Make a `get_summary()` method that returns a dict with important parameters of the simulation for easier creation of the "constants" in JSON data files (e.g. average `E_B`, `T`, `moment`...).

3. Low priority
    - [ ] Allow `T=0` for Néel (and Wolff?), though this will require a different `E_B` calculation which has previously led to problems. One option would be to use the alternative calculation only if `T=0`.
    - [ ] Could sparse matrices cause some speedup anywhere, e.g. in `DipolarEnergy`?
    - [ ] Rename the `OutputReader`s to something else, e.g. `Readout`.
    - [ ] Create more OOP ASI geometries (Kagome, equilateral Cairo...)
    - [ ] Randomly missing magnets. The only foreseeable issue that could occur with missing spins, is that some `Magnets().select()` samples remain unused, but I could live with that.
        - [ ] An extension of this randomness could be to generate an 'ensemble' of systems, which could be calculated efficiently in a parallel manner on GPU by extending arrays into a third dimension, and then using advanced indexing or some `stride_tricks` to manipulate each 2D slice differently while still being parallel. The issue with this is that this will be a lot of work where not a single slice may be wrong, and that a lot of code will have to be refactored to work with this extra dimension. Currently it seems that only `E_B`, `moment` and `T` should be moved into the 3rd dimension. But selecting magnets will become problematic, as for Glauber this will result in different number of samples in each element of the ensemble (so the indices can no longer be a rectangular array unless padded with e.g. -1), and for Néel this will require an argmin along only two of the three axes. The question is whether this will use more of the GPU at once, or whether this excessive indexing would slow things down more than we gain. There is only one way to find out, I guess, since the proof is in the pudding...

## Analysis and tests

1. High priority
    - [ ] Compare with other models
        - [ ] "Reservoir Computing in ASI": Pinwheel ASI with squinting
    - [ ] Calculate kernel-quality and task agnostic metrics as function of T, for different lattices with different interactions etc. First try easiest models and then go closer to experiment. Also I first have to determine suitable ranges for these metrics before I can put it on HPC ;)
        - [ ] To prevent using way too much calculation time, plot (NL, MC, S) as function of the number of samples `len(experiment.u)` to determine when the metrics stabilize, and possibly also `k` if that is not too much work. This behavior might be situation-dependent.

2. Medium priority
    - [ ] How many MC steps do we need to reach a statistically stable state?
    - Test physical accuracy of Hotspice
        - [ ] Domain size of square and pinwheel? As a function of time? (is there a maximum size or do they keep growing)
        - [ ] 2D ferromagnet (Arrott-Belov? Curie-Weiss? Susceptibility/magnetization?)
        - [x] 2D Ising with long-range dipolar interactions ("Aging in a Two-Dimensional Ising Model with Dipolar Interactions")
            - [ ] Can we also check the 'aging' part of that paper?
    - [ ] Multi-switch: analyze influence of Q on physical behavior to see which values of Q are acceptable (0.01? 0.1? 1? 100???), but how to do this? In `OOP_Square` perhaps?
    - [x] Simulation speed (samples/sec) for different grid sizes?
        - [ ] Are certain nicely factorable grid sizes preferable? (with PBC off, otherwise the unitcells will have an overwhelming effect depending on whether or not they fit perfectly together along the edges)

3. Low priority
    - Test physical accuracy of Hotspice
        - [ ] Angle between 0° and 45° between IP_Square and IP_Pinwheel where AFM alignment changes to uniform and vice versa? Can be useful to compare to experiment

## Various smaller things that I will probably never come around to doing

- [ ] Improve relaxation algorithm for `Magnets.relax()`
- [ ] Improve `plottools.py`
  - [ ] Organize plotting functions better
  - [ ] Improve the Average class to make it more versatile and consistent between different ASIs, for example by actually calculating the field $\vec{B} = \mu_0 \Big[\frac{3\vec{u_r}(\vec{u_r}\cdot\vec{m}) - \vec{m}}{4\pi r^3} + \frac{2\vec{m}}{3}\delta(\vec{r})\Big]$ of the neighbors at each cell (but this might be quite computationally expensive for large neighborhoods, which is not an issue for normal plots but for animations this might not be ideal).
  - [ ] Function to plot field direction at every cell by taking into account Zeeman and dipolar field from all magnets
- [ ] Implement parallel Poisson-disc sampling
- [ ] Linear transformations (e.g. skewing or squeezing) can be implemented by acting on `xx` and `yy`, but this might not be so easy with the unit cells etc. (for plots: [plt imshow](https://matplotlib.org/stable/gallery/images_contours_and_fields/affine_image.html "Affine transform of an image for skewed geometries"))
