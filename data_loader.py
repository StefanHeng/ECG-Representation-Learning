"""
Load heartbeats samples from each dataset, specified by a sample frequency and fixed distances relative to R-peak
"""

from data_path import *


class DataLoader:
    def __init__(self, dnm, fqs=250):
        """
        :param dnm: Encoded dataset name
        :param fqs: (Potentially re-sampling) frequency
        """
        self.path = f'{PATH_BASE}/{DIR_DSET}/{dnm}'
