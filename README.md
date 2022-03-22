# Hotspin

Hotspin is a tool for simulating thermally active artificial spin ices, using an Ising-like approximation with Glauber dynamics to model their statistical behavior. The axis and position of each spin is fixed, only their binary state can switch.

## Dependencies

Hotspin makes heavy use of the `CuPy` library, which allows GPU-accelerated array computing with CUDA for NVIDIA GPUs. Installing it can be nontrivial, but the easiest method might be using conda and running `conda install -c conda-forge cupy`.

## Getting started

Hotspin functions as a Python module, and can therefore be used through `import hotspin`.

To create a simulation, instantiate any of the ASI subclasses in the `hotspin.ASI` submodule, for example `mm = hotspin.ASI.PinwheelASI(100, 1e-6)` (we often use the variable `mm` to store a spin ice). For more information on which parameters are required to instantiate a given ASI lattice, refer to the docstrings.

Examples of usage for each of the ASI lattices, as well as examples of functions operating on these ASI objects, are provided in the 'examples' directory.

To perform a single simulation step, call `mm.update()`.

More complex input-output simulations can be performed using the `hotspin.io` and `hotspin.experiments` modules, but these are still under construction.
