""" Run this file using the command
        python <this_file.py> <script_path.py>
    to run the python script located at path <script_path.py> on all available GPUs.
    It is assumed that <script_path.py> contains a global variable named 'sweep', which should support __len__().
    In the normal use case, 'sweep' is an instance of a hotspice.experiments.Sweep() subclass.
    Also, <script_path.py> should be callable from shell as
        python <script_path.py> [-h] [-o [OUTDIR]] [N]
    where N specifies the index of the iteration of the sweep that the script should actually execute when called, and
    where OUTDIR is the directory path where the output file of that iteration (or, if N is not specified, all iterations) should be stored.
"""

import argparse
import importlib.util
import multiprocessing
import os
import psutil
import shutil
import subprocess
import sys

from joblib import Parallel, delayed
from cupy.cuda.runtime import getDeviceCount

try: import hotspice
except ModuleNotFoundError: from context import hotspice


if __name__ != "__main__": raise RuntimeError("ParallelJobs.py should only be run as a script, never as a module.")


## Define, parse and clean command-line arguments
# Usage: python <this_file.py> [script_path] [-h] [-o [OUTDIR]] [-i [ITERATIONS_LIST]]
# Example: python ParallelJobs.py some_script_file.py -i 13 21 34 55 -o "results/some_script_file.out"
# Note that outdir and script_path are both relative to the current working directory, not to ParallelJobs.py!
parser = argparse.ArgumentParser(description="Runs the sweep defined in another script on all available GPUs.")
parser.add_argument('script_path', type=str,
                    help="The path of the script file containing the parameter sweep to be performed.")
parser.add_argument('-o', '--outdir', nargs='?', 
                    dest='outdir', type=str, default=None,
                    help="The output directory, relative to the current working directory.")
parser.add_argument('-i', '--iterations', nargs='*', # '*' creates a list of all 'arguments' that follow after -i, until the next keyword
                    dest='iterations', type=int, default=None,
                    help="If specified, only these iterations of the sweep are ran. A space-separated list of integers.")
args, _ = parser.parse_known_args()
args.script_path = os.path.abspath(args.script_path) # Make sure the path is ok
if args.outdir is None: args.outdir = os.path.splitext(args.script_path)[0] + '.out/Sweep'
if not os.path.exists(args.script_path):
    raise ValueError(f"Path '{args.script_path}' provided as cmd argument does not exist.")


## Load script_path as module to access sweep
spec = importlib.util.spec_from_file_location('sweepscript', args.script_path)
foo = importlib.util.module_from_spec(spec)
sys.modules['sweepscript'] = foo
spec.loader.exec_module(foo)
try:
    sweep = foo.sweep
except AttributeError:
    raise AttributeError("There is no global variable named 'sweep' in the provided script.")


## Create queue
multiprocessing.set_start_method('spawn')
N_PARALLEL_JOBS = getDeviceCount() if hotspice.config.USE_GPU else psutil.cpu_count(logical=False)
q = multiprocessing.Queue(maxsize=N_PARALLEL_JOBS)
for i in range(N_PARALLEL_JOBS): q.put(i)

## Copy input file to prevent changes during runtime
outdir = os.path.abspath(args.outdir) + (timestamp := hotspice.utils.timestamp())
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
    print(f"Created output directory '{outdir}'")
copied_script_path = os.path.join(outdir, f"{os.path.splitext(os.path.basename(args.script_path))[0]}_{timestamp}.py")
shutil.copy(args.script_path, copied_script_path)

## Create some global variables/functions for the entire sweep
n_funcs = len(sweep)
failed = []
def runner(i):
    device_id = q.get() # ID of GPU or CPU core to be used
    text_core_num = f"GPU{device_id}" if hotspice.config.USE_GPU else f"CPU{device_id}"
    # Run a shell command that runs the relevant python script
    hotspice.utils.log(f"Attempting to run job #{i} of {n_funcs} on {text_core_num}...", style='header')
    cmd = ["python", copied_script_path, '-o', outdir, str(i)]
    try:
        env = os.environ.copy()
        #! These os.environ statements must be done before running any CuPy statement (importing is fine),
        #! but since CuPy is imported separately in the subprocess python script this is not an issue.
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env['CUDA_VISIBLE_DEVICES'] = f'{device_id:d}'
        env['HOTSPICE_DEVICE_ID'] = f'{device_id:d}' # To allow detection of used CPU core without advanced sorcery
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError: # This error type is expected due to check=True
        failed.append(i)
        hotspice.utils.log(f"The command '{' '.join(cmd)}' could not be run successfully. See a possible error message above for more info.", style='issue')

    # Return gpu id to queue
    q.put(device_id)


## Run the jobs
iterations = range(n_funcs) if args.iterations is None else args.iterations # Which iterations of the sweep are ran (default: all)
num_cores = min(N_PARALLEL_JOBS, len(iterations))
cores_text = f"{num_cores} {'G' if hotspice.config.USE_GPU else 'C'}PU{'s' if num_cores > 1 else ''}"
hotspice.utils.log(f"Running {len(iterations)} job{'s' if len(iterations) > 1 else ''} on {cores_text}...", style='header', show_device=False)
Parallel(n_jobs=N_PARALLEL_JOBS, backend='threading')(delayed(runner)(i) for i in iterations)

with open(os.path.abspath(os.path.join(outdir, "README.txt")), 'w') as readme:
    readme.write(f"Failed iterations (flat-index): {failed}")

if len(failed):
    hotspice.utils.log(f"Failed sweep iterations (zero-indexed): {failed}", style='issue', show_device=False)
elif len(iterations) == len(sweep):
    hotspice.utils.log(f"Sweep successfully finished. Summarizing results...", style='success', show_device=False)
    try:
        subprocess.run(cmd := ["python", copied_script_path, '-o', outdir], check=True) # To summarize the sweep if all went well
    except subprocess.CalledProcessError: # This error type is expected due to check=True
        hotspice.utils.log(f"Failed to summarize sweep with '{' '.join(cmd)}'", style='issue', show_device=False)
