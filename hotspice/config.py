import os
import sys


if '--hotspice-use-cpu' in sys.argv:
    os.environ['HOTSPICE_USE_GPU'] = 'False'
elif '--hotspice-use-gpu' in sys.argv:
    os.environ['HOTSPICE_USE_GPU'] = 'True'

# USE_GPU: True or False, determines whether to use CuPy or NumPy to store and manipulate arrays
USE_GPU = os.environ.get('HOTSPICE_USE_GPU', 'True').lower() in ('true', 't', '1', 'y', 'yes', 'on')

# DEVICE_ID: Int or None, used when running multiple instances on several CPUs/GPUs to know which one
#            If HOTSPICE_DEVICE_ID is set, this hotspice instance can know the ID of CPU core or GPU it is running on.
DEVICE_ID = os.environ.get('HOTSPICE_DEVICE_ID', None)
DEVICE_ID = int(DEVICE_ID.split(',')[0]) if isinstance(DEVICE_ID, str) else None

def get_dict():
    """ Returns a dictionary containing all the configuration parameters and their values at the moment. """
    return {
            'USE_GPU': USE_GPU,
            'DEVICE_ID': DEVICE_ID
            }
