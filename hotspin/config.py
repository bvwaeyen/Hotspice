import os


USE_GPU = os.environ.get('HOTSPIN_USE_GPU', 'True').lower() in ('true', 't', '1')
# TODO: change 'cp' to 'xp' and keep the things that have to REALLY be CuPy (can't be numpy in any way) named 'cp'

def get_dict():
    ''' Returns a dictionary containing all the configuration parameters and their values at the moment. '''
    return {'USE_GPU': USE_GPU}
