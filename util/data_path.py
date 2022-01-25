import os

# PATH_BASE = '/Users/stefanh/Documents/UMich/Research/ECG Classification'  # Absolute system path for root directory
# DIR_PROJ = 'ECG-Representation-Learning'  # Repo root folder name
DIR_DSET = 'datasets'  # Dataset root folder name

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]
PATH_BASE = os.sep.join(paths[:-1])
DIR_PROJ = paths[-1]

# if __name__ == '__main__':
#     import os
#     from icecream import ic
#
#     paths = __file__.split(os.sep)
#     paths = paths[:paths.index('util')]
#     PATH_BASE = os.sep.join(paths[:-1])
#     DIR_PROJ = paths[-1]
#     ic(PATH_BASE, DIR_PROJ)
