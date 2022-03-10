# TODO

## Core functionality

1. High priority
    - [ ] Determine if $E_B$ needs to be taken into account in Glauber model
    - Multi-switching
        - [ ] Fix PBC issue in Grid multi-switching
        - [x] Make r in Grid multi-switching different for x and y
        - [ ] If kernel is cut off, recalculate it after every *something* steps (requires a parameter to specify this *something*, and further investigation of the error made when cutting off the kernel since the last calculation I did, did not seem to correspond to reality very well)
    - [x] Create parameter system which user can set, maybe module-wide but better might be `Magnets()`-wide, which controls implementation-related things like whether to use Glauber or Néel-Arrhenius, or how many to multi-switch at once, or how many steps between full kernel recalculations, or how hard to truncate the kernel, or... (probably a dataclass or something)

    - [ ] Develop .io and .experiments modules
        - [x] Kernel rank experiment
        - [ ] Task Agnostic Metrics for RC
        - [ ] Allow analog inputs, not just binary

2. Medium priority
    - [ ] Make unit tests
    - [ ] Organize plotting functions better (standardization of fonts, saving, ...)
    - [ ] Sort out the AFM-ness and its normalization etc., or even better find new ways to analyze systems with AFM ground state because the current method is behaving similarly as when a cheese grater is used as ship
    - [ ] Update autocorrelation function to use SI units etc.
    - [x] Make a plottools function which clearly shows the basic lattice of an ASI (not full, just a couple of unit cells), with grid points and such indicated, to include in $\LaTeX$.

3. Low priority
    - [ ] Compress $M_{sat}$ and $V$ into one parameter (unit Am² = Nm/T) (or is there some reason why they need to be separate?)
    - [ ] Linear transformations (e.g. skewing or squeezing) should be relatively easy to implement by acting on xx, yy, but unit cells and more advanced calculations might become an issue ([plt imshow](https://matplotlib.org/stable/gallery/images_contours_and_fields/affine_image.html "Affine transform of an image for skewed geometries"))
    - [ ] Random defects (i.e. missing magnets, because other randomness will interfere with the unit cells)
    - [ ] Turn Vec2D into a Unitcell class that includes more information than just nx and ny

## Analysis and tests

1. High priority
    - Test physical accuracy of Hotspin
        - [x] 2D square Ising model
        - [ ] 2D ferromagnet (Arrott-Belov? Curie-Weiss? Susceptibility/magnetization?)
        - [ ] Pinwheel reversal in external field (cfr. flatspin paper)
    - [ ] Learn to use pandas (or something similar) to save results in a structured manner

2. Medium priority
    - Test physical accuracy of Hotspin
        - [ ] Domain size of square and pinwheel? As a function of time? (is there a maximum size or do they keep growing)
    - [ ] Multi-switch: analyze influence of Q on physical behavior to see which values of Q are acceptable (0.01? 0.1? 1? 100???)
    - [ ] Simulation speed (samples/sec) for different grid sizes? Are certain nicely factorable lengths preferable?

3. Low priority
    - Test physical accuracy of Hotspin
        - [ ] Angle between 0° and 45° between SquareASI and PinwheelASI where AFM alignment changes to uniform and vice versa? Can be useful to compare to experiment
    - [x] Distribution of sampling method
