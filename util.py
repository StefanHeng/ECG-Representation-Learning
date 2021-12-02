import json
import math
import os
import glob
from functools import reduce
import pathlib
import concurrent.futures

import h5py
import numpy as np
import pandas as pd
import wfdb
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


def set_(dic, ks, val):
    ks = ks.split('.')
    node = reduce(lambda acc, elm: acc[elm], ks[:-1], dic)
    node[ks[-1]] = val


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


def conc_map(fn, it):
    """
    Wrapper for `concurrent.futures.map`

    :param fn: A function
    :param it: A list of elements
    :return: Iterator of `lst` elements mapped by `fn` with concurrency
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.map(fn, it)


def stem(path, ext=False):
    """
    :param path: A potentially full path to a file
    :param ext: If True, file extensions is preserved
    :return: The file name, without parent directories
    """
    return os.path.basename(path) if ext else pathlib.Path(path).stem


def remove_1st_occurrence(str_, str_sub):
    idx = str_.find(str_sub)
    return str_[:idx] + str_[idx+len(str_sub):]


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(f'{PATH_BASE}/{DIR_PROJ}/config.json') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def save_fig(save, title):
    if save:
        fnm = f'{title}.png'
        plt.savefig(os.path.join(PATH_BASE, DIR_PROJ, 'plot', fnm), dpi=300)


def plot_1d(arr, label=None, title=None, save=False, s=None, e=None, **kwargs):
    """ Plot potentially multiple 1D signals """
    # kwargs = dict(
    #     label=label
    # )

    def _plot(a, lb):
        a = a[s:e]
        plt.plot(np.arange(a.size), a, marker='o', ms=0.3, lw=0.25, label=lb, **kwargs)
    plt.figure(figsize=(18, 6), constrained_layout=True)
    if not isinstance(arr, list):
        arr = [arr]
    lbl = [None for _ in arr] if label is None else label
    _ = [_plot(a, lb) for a, lb in zip(arr, lbl)]  # Execute
    # else:
    #     _plot(arr, label)
    if label:
        plt.legend()
    if title:
        plt.title(title)
    save_fig(save, title)
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


def get_my_rec_labels():
    d_my = config(f'{DIR_DSET}.my')
    recs_csv_fnm = os.path.join(PATH_BASE, DIR_DSET, d_my['dir_nm'], d_my['fnm_labels'])
    df = pd.read_csv(recs_csv_fnm)
    return df.apply(lambda x: x.astype('category'))


def get_rec_paths(dnm):
    d_dset = config(f'{DIR_DSET}.{dnm}')
    dir_nm = d_dset['dir_nm']
    path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
    return sorted(glob.iglob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))


def get_record_eg(dnm, n=0, ln=None):
    """
    Get an arbitrary record

    :param dnm: Dataset name
    :param n: Entry in the dataset
    :param ln: Number of samples in the record
        if None, full record returned

    .. note:: Works only if a wfdb record file exists
    """
    # d_dset = config(f'{DIR_DSET}.{dnm}')
    # dir_nm = d_dset['dir_nm']
    # path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
    # rec_path = next(glob.iglob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))
    rec_path = get_rec_paths(dnm)[n]
    kwargs = dict(
        sampto=ln
    )
    for k, v in kwargs.items():
        if k is None:
            del kwargs[v]
    return wfdb.rdrecord(rec_path[:rec_path.index('.')], **kwargs)


def get_signal_eg(dnm=None, n=None):
    """
    :param dnm: Dataset name, sampled at random if not given
    :param n: Entry in the dataset, sampled at random if not given
    :return: A 12*`l` array of raw signal samples
    """
    if dnm is None:
        dsets = config('datasets_export.total')
        idx = np.random.randint(len(dsets))
        dnm = dsets[idx]
    if n is None:
        n = np.random.randint(config(f'{DIR_DSET}.{dnm}.n_rec'))

    if dnm == 'CHAP_SHAO':
        fnm = get_rec_paths(dnm)[n]
        df = pd.read_csv(fnm)
        return df.to_numpy()
    elif dnm == 'CODE_TEST':
        fnm = get_rec_paths(dnm)[0]  # 1 hdf5 file
        rec = h5py.File(fnm, 'r')
        return rec['tracings'][n]
    else:
        return get_record_eg(dnm, n=n).p_signal


def get_nlm_denoise_truth(verbose=False):
    dnm = 'CHAP_SHAO'
    fnm = get_rec_paths(dnm)[77]
    fnm_stem = stem(fnm)
    dbg_path = os.path.join(PATH_BASE, DIR_DSET, config(f'{DIR_DSET}.{dnm}.dir_nm'), 'my_denoise_debugging')
    if verbose:
        ic(fnm, fnm_stem)
        ic(dbg_path)

    df = pd.read_csv(fnm)
    df_de = pd.read_csv(fnm.replace('ECGData', 'ECGDataDenoised'), header=None)
    if verbose:
        ic(len(df))
        ic(df_de.head(6))
        ic(df_de.iloc[:6, 0])

    fnm_lowpass = os.path.join(dbg_path, f'{fnm_stem}, lowpass.csv')
    fnm_rloess = os.path.join(dbg_path, f'{fnm_stem}, rloess.csv')
    fnm_localres = os.path.join(dbg_path, f'{fnm_stem}, localres.csv')
    fnm_after2nd = os.path.join(dbg_path, f'{fnm_stem}, after2nd.csv')

    return (
        df.iloc[:]['I'].to_numpy(),
        df_de.iloc[:][0].to_numpy(),
        pd.read_csv(fnm_lowpass, header=None).iloc[:, 0].to_numpy(),
        pd.read_csv(fnm_rloess, header=None).iloc[:, 0].to_numpy(),
        pd.read_csv(fnm_localres, header=None).iloc[:, 0].to_numpy(),
        pd.read_csv(fnm_after2nd, header=None).iloc[:, 0].to_numpy()
    )


if __name__ == '__main__':
    from icecream import ic
    np.random.seed(77)

    # ic(config('datasets.BIH_MVED'))

    # ic(remove_1st_occurrence('E00002.mat 12 500 5000 05-May-2020 14:50:55', '.mat'))

    ic(get_signal_eg(dnm='G12EC', n=0).shape)
    ic(get_signal_eg(dnm='CHAP_SHAO', n=0))
    ic(get_signal_eg(dnm='CODE_TEST', n=0).shape)
