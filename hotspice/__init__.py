"""
This is the Hotspice package.
"""
__all__ = ['ASI', 'config', 'energies', 'experiments', 'gui', 'io', 'plottools', 'utils']
__author__ = "Jonathan Maes"
__version__ = None

# Import config before anything else, just to be sure that xp is either CuPy or NumPy everywhere as desired
from . import config

# The following explicit import allows accessing the things in 'core.py' through hotspice.<thing_in_core.py>
from .core import *
__all__.extend(core.__all__) # the name 'core' is defined in the namespace after 'from .core import *'

# The following imports allow accessing everything in <name>.py through hotspice.<name>.<thing_in_<name>.py>
from . import utils
from . import ASI
from . import plottools
from . import io
from . import experiments
from . import gui
from . import energies
# from . import poisson #! unfinished module

# For backwards compatibility (and ease of use), allow Energy components to be accessed as 'hotspice.ZeemanEnergy' etc.
__all__.extend(['ZeemanEnergy', 'DipolarEnergy', 'ExchangeEnergy'])
from .energies import ZeemanEnergy, DipolarEnergy, ExchangeEnergy