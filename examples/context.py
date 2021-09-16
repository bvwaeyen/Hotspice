""" This file (context.py) allows easy access to the hotspin.py module from 
    any python file inside this subdirectory, but only if hotspin.py is
    located in the parent directory exactly one level higher. To import, use:
    >>> from context import hotspin

    To access hotspin.py from deeper subdirectories, modify the dots (..) in
    the sys.path.insert() call in this file.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add parent directory to python path

import hotspin # This is the line of code we need: importing the hotspin.py module which is located in the parent directory

if __name__ == "__main__":
    print(hotspin)