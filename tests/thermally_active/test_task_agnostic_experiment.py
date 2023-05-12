""" Created 10/05/2023
Trying to set up task agnostic experiment with given system.
"""
from thermally_active_system import *
import os
# import pandas as pd
# import matplotlib.pyplot as plt

# TODO vary lattice size a as well

# things to do
# TODO this is very limited, because it takes ages
samples = 10  # not enough
H_list =  H * np.array([0.5, 1., 2., 5., 8.])  # Manually chosen maybe interesting points
dt_list = dt * np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])  # from 10% to 200% in steps of 10%

# handy to control zeeman directly
zeeman = hotspice.ZeemanEnergy(magnitude=H, angle=input_angle)
mm.add_energy(zeeman)

# experimental variables
N = 200  # I would like to have more, but takes ages
experiment_pattern = "random"  # after this mm.relax() is called, which makes it close to thermally relaxed system
directory = "TAE/"
def filename(H, dt, sample):
    return f"TAE H {H} dt {dt} sample {sample}.json"
os.makedirs(directory, exist_ok=True)
verbose = True

# experimental setup
datastream = hotspice.io.RandomBinaryDatastream()  # input will always be random
inputter = hotspice.io.PerpFieldInputter(datastream=datastream, magnitude=H, angle=input_angle, n=np.inf,
                                         relax=False, frequency=1. / dt, sine=False, half_period=True)
outputreader = hotspice.io.FullOutputReader(mm) # Full resolution, TODO change output for actual metrics
experiment = hotspice.experiments.TaskAgnosticExperiment(inputter, outputreader, mm)

make_new_data = True
if make_new_data:
    for H_num, H in enumerate(H_list):
        inputter.magnitude = H
        for dt_num, dt in enumerate(dt_list):
            inputter.frequency = 1./dt
            for sample in range(samples):
                print(f"Calculating H = {H * 1000 :.2f} mT [{H_num+1}/{len(H_list)}], dt = {dt * 1e9} ns [{dt_num+1}/{len(dt_list)}]: sample {sample+1}/{samples}")

                if os.path.exists(directory + filename(H, dt, H)):
                    print(f"File already exists! Skipping this one!")
                    break

                mm.E_B = E_B * np.ones((n, n))*np.random.normal(1, randomness, size=(n, n))  # new sample new E_B
                zeeman.set_field(magnitude=0)  # turn off magnetic field for proper initialization inside run()
                experiment.run(N=N, pattern=experiment_pattern, verbose=verbose)
                experiment.to_dataframe().to_json(directory + filename(H, dt, sample))  # save

    print("Done creating data!")

# --------------------------------------------------
# TODO Data analysation

# TODO decide k (default 10)