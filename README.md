# Hotspin

Hotspin is a tool for simulating thermally active artificial spin ices, using an Ising-like approximation: the axis and position of each spin is fixed, only their binary state can switch.
The time evolution can either follow the Néel-Arrhenius law of switching over an energy barrier in a chronological manner, or use Glauber dynamics to model the statistical behavior while making abstraction of the time variable.

## Dependencies

Hotspin relies on the `CuPy` library to provide GPU-accelerated array computing with CUDA for NVIDIA GPUs.

There are several ways to [install CuPy](https://docs.cupy.dev/en/stable/install.html), but likely the easiest method is to use the following `conda` command, since this automatically installs the appropriate version of the CUDA toolkit:

```shell
conda install -c conda-forge cupy
```

## Getting started

Hotspin is designed as a Python module, and can therefore simply be imported through `import hotspin`.
Several submodules provide various components, most of which are optional.
All of the strictly crucial classes and functions are provided in the default `hotspin` namespace, but other modules like `hotspin.ASI` provide wrappers and examples for ease of use.

### Creating a simple spin ice

To create a simulation, the first thing one has to do is to create a spin ice. This can be done by instantiating any of the ASI subclasses from the `hotspin.ASI` submodule, for example a 'diamond' pinwheel geometry:

```python
import hotspin
mm = hotspin.ASI.IP_Pinwheel(1e-6, 100)
```

The meaning of the values `1e-6` and `100` requires a small introduction on how the spin ices are stored in memory.
Most data and parameters of a spin ice are stored as 2D CuPy arrays, since this improves efficiency for certain steps in the calculation.
However, this also implies that the possible spin ice geometries are restricted to those that can be represented as a periodic structure on a square grid.
The geometry is then defined by leaving some spots on this grid empty, while filling others with a magnet and its relevant parameters like magnetic moment $M_{sat} V$ (`mm.moment`), energy barrier $E_B$ (`mm.E_B`), temperature $T$ (`mm.T`), orientation...

- The first parameter (`a=1e-6`) specifies a typical distance between two magnets, but the exact meaning of this depends on the spin ice geometry being used, and how it is implemented on a grid.

- The second parameter in the initialization call (`n=100`) specifies the length of each axis of these underlying 2D arrays.
Hence, the spin ice in the example above will have 10000 available grid-points to put a magnet.
However, this 'diamond' pinwheel geometry can only be represented on a grid by leaving half of the available cells empty, so the simulation above actually contains 5000 magnets (see `mm.n`).

- Additional parameters can be used, for which we currently refer to the docstring of `hotspin.Magnets()`. In particular, the `params` parameter accepts a `hotspin.SimParams` instance which can set technical details for the spin ice, e.g. the update/sampling scheme to use, whether to use a reduced kernel...

For details on the grid representation and exact meaning of the 'typical distance' for each of the standard spin ices, see [Available spin ices](#available-spin-ices) or refer to the docstrings.

Examples of usage for each of the ASI lattices, as well as examples of functions operating on these ASI objects, are provided in the 'examples' directory.

### Stepping in time

To perform a single simulation step, call `mm.update()`. The scheme used to perform this single step is determined by `mm.params.UPDATE_SCHEME` (possible schemes: `'Néel'`, `'Glauber'`, `'Wolff'`).

To relax the magnetization to a (meta)stable state, call `mm.relax()` or `mm.minimize()` (the former is faster for large simulations but less accurate, the latter is faster for small simulations and follows the real relaxation order more closely).

### Applying input and reading output

More complex input-output simulations can be performed using the `hotspin.io` and `hotspin.experiments` modules, but these are still under construction.

The `hotspin.io` module contains classes that apply external stimuli to the spin ice, or read the state of the spin ice in some manner.

The `hotspin.experiments` module contains classes to bundle many input/output runs and calculate relevant metrics from them, as well as classes to perform parameter sweeps.

### Performing a parameter sweep on multiple GPUs

The `GPUparallel.py` script can be used to run a `hotspin.experiments.Sweep` on multiple GPUs.

### Choosing between GPU or CPU

By default, hotspin runs on the GPU. One can also choose to run hotspin on the CPU instead, which can be useful for small simulations with only several tens of magnets, where the parallelism of GPU computing is not quite beneficial.

When hotspin is imported through `import hotspin`, it checks the environment variable `HOTSPIN_USE_CPU` (default: `False`) to determine whether to compute on CPU or GPU. *Hence, this variable must be set **BEFORE** importing the hotspin module*, and thus the choice can only be made once in a given script (unless hotspin is somehow re-loaded completely, e.g. using `importlib.reload`).

One can set this environment variable in a python script as follows:

```python
import os
os.environ["HOTSPIN_USE_CPU"] = True
import hotspin # Only import AFTER setting HOTSPIN_USE_CPU!
```

## Available spin ices

Several geometries are available out-of-the-box in hotspin.
They are listed below with a small description of their peculiarities.
They all follow the pattern `hotspin.ASI.<name>(a, n, nx=None, ny=None, **kwargs)`, where `n` is only required if either `nx` or `ny` is not specified.

### Out-of-plane

Since all magnets of an out-of-plane spin ice have a magnetic moment with no in-plane component, they are simply represented as circles.

#### `OOP_Square`

<img src="./figures/ASI_lattices/OOP_Square_11x11.png" alt="OOP_Square_11x11" width="200"/>

`a` is the distance between nearest neighbors. The occupation is full.

#### `OOP_Triangle`

<img src="./figures/ASI_lattices/OOP_Triangle_7x4.png" alt="OOP_Triangle_7x4" width="200"/>

`a` is the distance between nearest neighbors. The occupation is 1/2.

### In-plane

Since all magnets of an in-plane spin ice have an anisotropy causing their magnetic moment to lay along a certain axis, called the 'easy axis', they are represented as ellipses whose long axis corresponds to this easy axis.

#### `IP_Ising`

<img src="./figures/ASI_lattices/IP_Ising_8x8.png" alt="IP_Ising_8x8" width="200"/>

`a` is the distance between nearest neighbors. The occupation is full.

#### `IP_Square`

<img src="./figures/ASI_lattices/IP_Square_5x5.png" alt="IP_Square_5x5" width="200"/>

`a` is the side length of a square, i.e. the side length of a unit cell. The occupation is 1/2.

#### `IP_Pinwheel`

<img src="./figures/ASI_lattices/IP_Pinwheel_5x5.png" alt="IP_Pinwheel_5x5" width="200"/>

This is the same as [`IP_Square`](#ipsquare), but with all spins rotated 45°.

#### `IP_Kagome`

<img src="./figures/ASI_lattices/IP_Kagome_5x3.png" alt="IP_Kagome_5x3" width="200"/>

`a` is the distance between opposing edges of a hexagon. The occupation is 3/8.

#### `IP_Triangle`

<img src="./figures/ASI_lattices/IP_Triangle_5x3.png" alt="IP_Triangle_5x3" width="200"/>

This is the same as [`IP_Kagome`](#ipkagome), but with all spins rotated 90°.
