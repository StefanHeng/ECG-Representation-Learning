import os

DIR_DSET = 'datasets'  # Dataset root folder name

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]
# Absolute system path for root directory; e.g.: '/Users/stefanh/Documents/UMich/Research/ECG Classification'
PATH_BASE = os.sep.join(paths[:-1])
DIR_PROJ = paths[-1]  # Repo root folder name; e.g.: 'ECG-Representation-Learning'
