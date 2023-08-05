""" Created 10/05/2023, heavily changed 05/08/2023
Trying to set up task agnostic experiment with given system.
"""
import hotspice
import numpy as np
import os


# PARAMETERS
n = 21
E_B = hotspice.utils.eV_to_J(25e-3)  # same as E_T
randomness = 0.05  # Random distribution still recommended. Reroll for new samples!

mm = hotspice.ASI.IP_Pinwheel(a=21e-9, n=n, E_B=E_B, T=300, moment=hotspice.utils.eV_to_J(2.5))
mm.params.UPDATE_SCHEME = "Néel"

# handy to control zeeman directly
input_angle = 7. * np.pi/180.  # angle is probably unnecessary, but can't hurt? Normally used to break symmetry ties
zeeman = hotspice.ZeemanEnergy(magnitude=0, angle=input_angle)
mm.add_energy(zeeman)


# things to do
samples = 20
B_array = np.arange(0.50, 5.01, 0.25) * 1e-3
dt_array = np.arange(0.15, 2.71, 0.15) * 1e-9  # can easily add more small dt if necessary

# experimental variables
N = 500
experiment_pattern = "vortex"  # after this mm.relax() is called, which makes it close to thermally relaxed system
directory = "TAE/"
os.makedirs(directory, exist_ok=True)
def filename(B, dt, sample):
    return f"TAE B {B * 1e3:.2f} mT, dt {dt * 1e9:.2f} ns, sample {sample}.json"
verbose = False

# experimental setup
datastream = hotspice.io.RandomBinaryDatastream()  # input will always be random
inputter = hotspice.io.PerpFieldInputter(datastream=datastream, magnitude=B_array[0], angle=input_angle, n=np.inf,
                                         relax=False, frequency=1./dt_array[0], sine=False, half_period=True)
fulloutputreader = hotspice.io.FullOutputReader(mm)  # Full resolution to save
experiment = hotspice.experiments.TaskAgnosticExperiment(inputter, fulloutputreader, mm)

make_new_data = True
if make_new_data:
    print("Creating new data")

    for sample in range(0, samples):
        np.random.seed(sample)  # sample number is seed
        E_B_distr = E_B * np.random.normal(1, randomness, size=(n, n))
        mm.E_B = E_B_distr

        for B_i, B in enumerate(B_array):
            inputter.magnitude = B
            for dt_i, dt in enumerate(dt_array):
                inputter.frequency = 1./dt

                print(f"Calculating B = {B * 1e3 :.2f} mT [{B_i}/{B_array.size}], dt = {dt * 1e9} ns [{dt_i}/{dt_array.size}]: sample {sample}/{samples}")

                if os.path.exists(directory + filename(B, dt, sample)):
                    print(f"File already exists! Skipping this one!")
                    continue

                zeeman.set_field(magnitude=0)  # turn off magnetic field for proper initialization inside run()
                experiment.run(N=N, pattern=experiment_pattern, verbose=verbose)
                experiment.to_dataframe().to_json(directory + filename(B, dt, sample))  # save

    print("Done creating data!")

