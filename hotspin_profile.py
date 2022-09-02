''' There are two available modes for debugging:
    1) python hotspin_profile.py <script>.py
       OR
       python hotspin_profile.py -m <package.module>
        Uses cProfile to profile the python file <script>.py, or the module <package.module> if
        the script is a non-executable part of a package. This follows the same naming convention
        as used when importing said package through 'import <package.module>'.
    2) python hotspin_profile.py -l <script>.py
        Line-by-line profiling of one or multiple functions used by <script>.py.
        Can also use 'kernprof -l -v <script>.py, but this does not save the output to a file.
        NOTE: the function(s) to be profiled has to be decorated with @profile! <script>.py does not
            need to import this decorator explicitly; kernprof does this behind-the-scenes.
            Thus, ignore any warnings about 'profile' being undefined.
        NOTE: requires kernprof to be installed. Use 'pip install line_profiler' to do so.

    Examples for each of the 3 cases:
        python profile.py hotspin/core.py
        python profile.py -m hotspin.plottools
        python profile.py -l examples/ASI_IP_Pinwheel.py
    Note that each command can be followed by any command-line arguments passed to the script <name>.py.
    
    The output is saved in the file
        "./profiling/<cProfile|kernprof>_<script.py|package.module>_<timestamp>.txt"
'''

import colorama
colorama.init()
import os
import subprocess
import sys
import warnings

from datetime import datetime


if not os.path.exists(outdir := 'profiling'): os.mkdir(outdir)
timestamp = datetime.utcnow().strftime(r"%Y%m%d%H%M%S")

def run(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError: # This error type is expected due to check=True
        warnings.warn(f"{colorama.Fore.RED}The command '{' '.join(cmd)}' could not be run successfully. See a possible error message above for more info.{colorama.Style.RESET_ALL}", stacklevel=2)


args = sys.argv[1:] # Don't need our own filename
if args[0] == "-m":
    outfile = os.path.abspath(f"{outdir}/cProfile_{args[1]}_{timestamp}.txt")
    run(["python", "-m", "cProfile", "-s", "tottime", "-m"] + args[1:] + [">", outfile])
elif args[0] == "-l":
    print(f"{colorama.Fore.LIGHTBLUE_EX}!!! Do not forget to add the @profile decorator to the funtion(s) of interest, otherwise the output of this profiling will be empty.{colorama.Style.RESET_ALL}")
    tempfile = os.path.abspath(f"{outdir}/kernprof_{os.path.basename(args[1])}_{timestamp}.lprof")
    outfile = os.path.abspath(f"{outdir}/kernprof_{os.path.basename(args[1])}_{timestamp}.txt")
    run(["kernprof", "-l", "-o", tempfile, "-v"] + args[1:])
    run(["python", "-m", "line_profiler", tempfile, ">", outfile])
    run(["del", tempfile])
else:
    outfile = os.path.abspath(f"{outdir}/cProfile_{os.path.basename(args[0])}_{timestamp}.txt")
    print(["python", "-m", "cProfile", "-s", "tottime"] + args + [">", outfile])
    run(["python", "-m", "cProfile", "-s", "tottime"] + args + [">", outfile])
