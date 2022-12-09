""" This file (context.py) allows easy access to the hotspice module from 
    any python file inside this subdirectory, but only if the hotspice directory is
    located in the parent directory exactly one level higher. To import, use:
    >>> from context import hotspice

    To access Hotspice from deeper subdirectories, modify the dots (..) in
    the sys.path.insert() call in this file.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add parent directory to python path

import hotspice

if __name__ == "__main__":
    print(f"sys.path = {sys.path}")
    print(f"Found: {hotspice}")