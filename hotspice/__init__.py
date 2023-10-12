"""
This is the Hotspice package.
"""
__all__ = ['config', 'utils', 'ASI', 'plottools', 'io', 'experiments', 'gui']
__author__ = "Jonathan Maes"
__version__ = None

# Import config before anything else, just to be sure
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
# from . import poisson #! unfinished module