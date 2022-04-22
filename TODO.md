# TODO

## Core functionality

1. High priority
    - [ ] Determine if $E_B$ needs to be taken into account in Glauber model
    - Multi-switching
        - [ ] Fix PBC issue in Grid multi-switching
        - [ ] If kernel is cut off, recalculate it after every *something* steps (requires a parameter to specify this *something*, and further investigation of the error made when cutting off the kernel, because the recent estimated calculation did not seem to correspond to reality very well)

    - [ ] Develop .io and .experiments modules
        - [ ] Task Agnostic Metrics for RC

    - [ ] Reconsider distribution of responsibilities between ASI() and Magnets() classes

2. Medium priority
    - [ ] Make unit tests
    - [ ] Organize plotting functions better
        - [ ] Improve the Average class to make it more versatile and consistent between different ASIs
    - [ ] Sort out the AFM-ness and its normalization etc., or even better find new ways to analyze systems with AFM ground state because the current method is behaving similarly as when a cheese grater is used as ship
    - [ ] Implement metrics (average magnetization, correlation e.g. by looking at nearest neighbors minimizing/maximizing dipolar interaction, susceptibility, width of domains (related to correlation probably)...) to compare with theory/experiment
    - [ ] Improve relaxation algorithm
    - [ ] Can implement autocorrelation length by taking into account cross product between vectors?
    - [ ] Possibly allow specifying angle also for pattern='AFM', if it makes any sense

3. Low priority
    - [ ] Compress $M_{sat}$ and $V$ into one parameter (unit Am² = Nm/T) (or is there some reason why they need to be separate?)
    - [ ] Linear transformations (e.g. skewing or squeezing) should be relatively easy to implement by acting on xx, yy, but unit cells and more advanced calculations might become an issue ([plt imshow](https://matplotlib.org/stable/gallery/images_contours_and_fields/affine_image.html "Affine transform of an image for skewed geometries"))
    - [ ] Random defects (i.e. missing magnets, because other randomness will interfere with the unit cells)
    - [ ] Turn Vec2D into a Unitcell class that includes more information than just nx and ny, and detect the unitcell dimensions etc. automatically
    - [ ] Plot that shows field direction at every cell taking into account Zeeman and dipolar field from all other magnets

## Analysis and tests

1. High priority
    - Test physical accuracy of Hotspin
        - [ ] 2D ferromagnet (Arrott-Belov? Curie-Weiss? Susceptibility/magnetization? "Aging in a Two-Dimensional Ising Model with Dipolar Interactions"?)
        - [x] Pinwheel reversal in external field (cfr. flatspin paper) (try with Néel update equation)
    - [ ] Calculate kernel-quality and task agnostic metrics as function of T, for different lattices with different interactions etc. First try easiest models and then go closer to experiment

2. Medium priority
    - [ ] How many MC steps do we need to reach a statistically stable state?
    - Test physical accuracy of Hotspin
        - [ ] Domain size of square and pinwheel? As a function of time? (is there a maximum size or do they keep growing)
    - [ ] Multi-switch: analyze influence of Q on physical behavior to see which values of Q are acceptable (0.01? 0.1? 1? 100???)
    - [ ] Simulation speed (samples/sec) for different grid sizes? Are certain nicely factorable lengths preferable?
    - [ ] Use date & time to save results

3. Low priority
    - Test physical accuracy of Hotspin
        - [ ] Angle between 0° and 45° between SquareASI and PinwheelASI where AFM alignment changes to uniform and vice versa? Can be useful to compare to experiment
