import os
import sys

if '--hotspice-use-cpu' in sys.argv: # Perhaps a bit weird to have 'use-cpu' here, and 'USE_GPU' in the env?
    os.environ['HOTSPICE_USE_GPU'] = 'False'

USE_GPU = os.environ.get('HOTSPICE_USE_GPU', 'True').lower() in ('true', 't', '1', 'y', 'yes', 'on')


def get_dict():
    ''' Returns a dictionary containing all the configuration parameters and their values at the moment. '''
    return {'USE_GPU': USE_GPU}
