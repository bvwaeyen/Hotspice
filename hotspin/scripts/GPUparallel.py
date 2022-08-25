''' Run this file using the command
        python <this_file.py> <script_path.py>
    to run the python script located at path <script_path.py> on all available GPUs.
    It is assumed that <script_path.py> contains a global hotspin.experiments.Sweep() subclass instance named 'sweep'.
    Also, <script_path.py> should be callable from shell as
        python <script_path.py> [-h] [-o [OUTDIR]] [N]
    where N specifies the index of the iteration of the sweep that the script should actually execute when called.
'''

import argparse
import importlib.util
import multiprocessing
import os
import subprocess
import sys
import warnings

import cupy as cp

from joblib import Parallel, delayed

import hotspin


if __name__ != "__main__": raise RuntimeError("GPUparallel.py should only be run as a script, never as a module.")


## Define, parse and clean command-line arguments
# Usage: python <this_file.py> [-h] [-o [OUTDIR]] [script_path]
# Note that outdir and script_path are both relative to the current working directory, not to this file!
parser = argparse.ArgumentParser(description="Runs the sweep defined in another script on all available GPUs.")
parser.add_argument('script_path', type=str, nargs='?',
                    default=os.path.join(os.path.dirname(__file__), "../../examples/SweepTA_RC_ASI.py"), #! hardcoded paths :(
                    help="The path of the script file containing the parameter sweep to be performed.")
parser.add_argument('-o', '--outdir', dest='outdir', type=str, nargs='?',
                    default='results/Sweeps/Sweep',
                    help="The output directory, relative to the current working directory.")
args = parser.parse_args()
args.script_path = os.path.abspath(args.script_path) # Make sure the path is ok
if not os.path.exists(args.script_path):
    raise ValueError(f"Path '{args.script_path}' provided as cmd argument does not exist.")


## Load script_path as module to access sweep
spec = importlib.util.spec_from_file_location("sweepscript", args.script_path)
foo = importlib.util.module_from_spec(spec)
sys.modules["sweepscript"] = foo
spec.loader.exec_module(foo)
try:
    sweep = foo.sweep
except:
    raise AttributeError("There is no global variable named 'sweep' in the provided script.")


## Create queue
multiprocessing.set_start_method('spawn')
N_GPU = cp.cuda.runtime.getDeviceCount()
q = multiprocessing.Queue(maxsize=N_GPU)
for device in range(N_GPU): q.put(device)


## Create some global variables for the entire sweep
num_jobs = len(sweep)
failed = []
outdir = args.outdir + (timestamp := hotspin.utils.timestamp())

def runner(i):
    # Select an available gpu
    gpu = q.get()
    # Run a shell command that runs the relevant python script
    hotspin.utils.log(f"Attempting to run job #{i} of {num_jobs} on GPU{i}...")
    cmd = f"python {args.script_path} -o {outdir} {i}"
    try:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #! These os.environ statements must be done before running any cupy statement (importing is fine),
        env["CUDA_VISIBLE_DEVICES"] = f"{gpu:d}" #! but since cupy is imported separately in the subprocess python script this is not an issue.
        subprocess.run(["python", args.script_path, '-o', outdir, str(i)], check=True, env=env)
    except subprocess.CalledProcessError: # This error type is expected due to check=True
        failed.append(i)
        warnings.warn(f"The command '{cmd}' could not be run successfully. See a possible error message above for more info.", stacklevel=2)

    # Return gpu id to queue
    q.put(gpu)


## Run the jobs
print(f"Running {num_jobs} jobs on {N_GPU} GPU{'s' if N_GPU > 1 else ''}...")
Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(i) for i in range(num_jobs))

if len(failed):
    print(f"Failed sweep iterations: {failed}")
else:
    print(f"Sweep successfully finished.")