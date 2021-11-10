import json
import math
from functools import reduce

import numpy as np
from wfdb import processing
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

from data_path import *

sns.set_style('darkgrid')


def get(dic, ks):
    """
    :param dic: Potentially multi-level dictionary
    :param ks: Potentially `.`-separated keys
    """
    ks = ks.split('.')
    return reduce(lambda acc, elm: acc[elm], ks, dic)


def keys(dic, prefix=''):
    """
    :return: Generator for all potentially-nested keys
    """
    def _full(k_):
        return k_ if prefix == '' else f'{prefix}.{k_}'
    for k, v in dic.items():
        if isinstance(v, dict):
            for k__ in keys(v, prefix=_full(k)):
                yield k__
        else:
            yield _full(k)


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(f'{PATH_BASE}/{DIR_PROJ}/config.json') as f:
            config.config = json.load(f)

    # node = config.config
    # for part in attr.split('.'):
    #     node = node[part]
    # return node
    return get(config.config, attr)


def plot_single(arr, label=None):
    """ Plot 1D signal """
    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(arr.size), arr, label=f'Signal {label}', marker='o', ms=0.3, lw=0.25)
    plt.legend()
    plt.show()


def plot_resampling(x, y, x_, y_, title=None):
    """
    Plots the original signal pair and it's resampled version
    """
    plt.figure(figsize=(16, 9))
    plt.plot(x, y, marker='o', ms=4, lw=5, label='Original', alpha=0.5)
    plt.plot(x_, y_, marker='x', ms=4, lw=1, label='Resampled')  # ls=(0, (2, 5)),
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_rpeak(sig, idx_rpeak, title=None):
    x = np.arange(sig.size)

    plt.figure(figsize=(16, 9))
    plt.plot(x, sig, marker='o', ms=0.3, lw=0.25, label='Original', alpha=0.5)

    for i in idx_rpeak:
        plt.axvline(x=i, c='r', lw=0.5, label='R peak')

    t = 'ECG R-peaks'
    if title:
        t = f'{t}, {title}'
    plt.title(t)
    handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def refine_rpeak(sig, idxs_peak, fqs, r_wd=100):
    """
    :param sig: 1D ECG signal
    :param idxs_peak: Indices of tentative R peaks
    :param fqs: Sample frequency
    :param r_wd: Half range in ms to look for optimal R peak
    :return: Refined R peak indices
    """
    return processing.correct_peaks(
        sig, idxs_peak,
        search_radius=math.ceil(fqs * r_wd / 1e3),
        smooth_window_size=2,  # TODO: what's this?
        peak_dir='up'
    )


if __name__ == '__main__':
    from icecream import ic
    ic(config('datasets.BIH_MVED'))
