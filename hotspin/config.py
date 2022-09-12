import os

USE_GPU = os.environ.get('HOTSPIN_USE_CPU', False)
# TODO: change 'cp' to 'xp' and keep the things that have to REALLY be CuPy (can't be numpy in any way) named 'cp'