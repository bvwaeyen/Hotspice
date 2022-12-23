"""
This is the Hotspice package.
"""


__author__ = "Jonathan Maes"
__version__ = None


# The following explicit import allows accessing the things in 'core.py' through hotspice.<thing_in_core.py>
from .core import *

# The following imports allow accessing everything in <name>.py through hotspice.<name>.<thing_in_<name>.py>
from . import config
from . import utils
from . import ASI
from . import plottools
from . import io
from . import experiments
# from . import poisson #! unfinished