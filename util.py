import json
import math
import os
import glob
from functools import reduce
import pathlib
import concurrent.futures
from datetime import datetime
from typing import Union

import colorama
import h5py
import numpy as np
import pandas as pd
import scipy.optimize
import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from data_path import *

rcParams['figure.constrained_layout.use'] = True
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


def now(as_str=True):
    d = datetime.now()
    return d.strftime('%Y-%m-%d %H:%M:%S') if as_str else d


def sizeof_fmt(num, suffix='B'):
    """ Converts byte size to human-readable format """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def log(s, c: str = '', as_str=False):
    """
    Prints `s` to console with color `c`
    """
    if not hasattr(log, 'reset'):
        log.reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
    if not hasattr(log, 'd'):
        log.d = dict(
            log='',
            warn=colorama.Fore.YELLOW,
            error=colorama.Fore.RED,
            err=colorama.Fore.RED,
            success=colorama.Fore.GREEN,
            suc=colorama.Fore.GREEN,
            info=colorama.Fore.BLUE,
            i=colorama.Fore.BLUE,

            y=colorama.Fore.YELLOW,
            yellow=colorama.Fore.YELLOW,
            red=colorama.Fore.RED,
            r=colorama.Fore.RED,
            green=colorama.Fore.GREEN,
            g=colorama.Fore.GREEN,
            blue=colorama.Fore.BLUE,
            b=colorama.Fore.BLUE,
        )
    if c in log.d:
        c = log.d[c]
    if as_str:
        return f'{c}{s}{log.reset}'
    else:
        print(f'{c}{now()}| {s}{log.reset}')


def logs(s, c):
    return log(s, c=c, as_str=True)


def clipper(low, high):
    """
    :return: A clipping function for range [low, high]
    """
    return lambda x: max(min(x, high), low)


def stem(path_, ext=False):
    """
    :param path_: A potentially full path to a file
    :param ext: If True, file extensions is preserved
    :return: The file name, without parent directories
    """
    return os.path.basename(path_) if ext else pathlib.Path(path_).stem


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


def plot_1d(arr, label=None, title=None, save=False, s=None, e=None, new_fig=True, show=True, plot_kwargs=None):
    """ Plot potentially multiple 1D signals """
    kwargs = dict(marker='o', ms=0.3, lw=0.25) | plot_kwargs

    def _plot(a, lb):
        a = a[s:e]
        plt.plot(np.arange(a.size), a, label=lb, **kwargs)
    if new_fig:
        plt.figure(figsize=(18, 6))
    if not isinstance(arr, list):
        arr = [arr]
    if not isinstance(label, list):
        label = [label]
    lbl = [None for _ in arr] if label is None else label
    _ = [_plot(a, lb) for a, lb in zip(arr, lbl)]  # Execute

    if label:
        plt.legend()
    if title:
        plt.title(title)
    if new_fig:
        save_fig(save, title)
    if show:
        plt.show()


def r2(y, y_fit):
    return 1 - (np.square(y - y_fit).sum() / np.square(y - np.mean(y)).sum())


def fit_power_law(x: np.ndarray, y: np.ndarray, return_fit: Union[int, bool] = False):
    """
    :return: 2-tuple of (coefficient, exponent) for power law
        If `return_fit` is True, return additionally 2-tuple of (fitted x, fitted y)
            If integer given, the fitted curve is returned by scale
    """
    def pow_law(x_, a, b):
        return a * np.power(x_, b)
    x, y = np.asarray(x).astype(float), np.asarray(y)
    (a_, b_), p_cov = scipy.optimize.curve_fit(f=pow_law, xdata=x, ydata=y, p0=(x[0], -1))
    ret = (a_, b_)
    if return_fit:
        scale = 1 if return_fit is True else return_fit
        x_plot = np.linspace(x.min(), x.max(), num=x.size * scale)
        print(x.min(), x_plot)
        y_fit = pow_law(x_plot, a_, b_)
        ret = ret, (x_plot, y_fit)
        # plt.plot(x_plot, y_fit, label='Fitted power law', lw=2)
    return ret


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
    path_ = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
    return sorted(glob.iglob(f'{path_}/{d_dset["rec_fmt"]}', recursive=True))


def get_record_eg(dnm, n=0, ln=None):
    """
    Get an arbitrary record

    :param dnm: Dataset name
    :param n: Entry in the dataset
    :param ln: Number of samples in the record
        if None, full record returned

    .. note:: Works only if a wfdb record file exists
    """
    rec_path = get_rec_paths(dnm)[n]
    kwargs = dict(
        sampto=ln
    )
    # for k, v in kwargs.items():
    #     if k is None:
    #         del kwargs[v]
    kwargs = {k: v for k, v in kwargs.items() if k is not None}
    return wfdb.rdrecord(rec_path[:rec_path.index('.')], **kwargs)


def fnm2sigs(fnm, dnm):
    if not hasattr(config, 'd_d_dset'):
        fnm2sigs.d_d_dset = config(DIR_DSET)

    if dnm == 'CHAP_SHAO':
        return pd.read_csv(fnm).to_numpy().T
    elif dnm == 'CODE_TEST':
        assert isinstance(fnm, int)  # Single file with all recordings
        if not hasattr(config, 'ct_tracings'):
            fnms = get_rec_paths(dnm)
            assert len(fnms) == 1
            fnm2sigs.ct_tracings = h5py.File(fnm, 'r')

        return fnm2sigs.ct_tracings['tracings'][fnm]
    else:
        return wfdb.rdrecord(fnm.removesuffix(fnm2sigs.d_d_dset[dnm]['rec_ext'])).p_signal.T


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
        # fnm = get_rec_paths(dnm)[n]
        # df = pd.read_csv(fnm)
        # return df.to_numpy()
        return fnm2sigs(get_rec_paths(dnm)[n], dnm)
    elif dnm == 'CODE_TEST':
        # fnm = get_rec_paths(dnm)[0]  # 1 hdf5 file
        # rec = h5py.File(fnm, 'r')
        # return rec['tracings'][n]
        return fnm2sigs(n, dnm)
    else:
        return get_record_eg(dnm, n=n).p_signal


def get_nlm_denoise_truth(verbose=False):
    dnm = 'CHAP_SHAO'
    fnm = get_rec_paths(dnm)[77]  # Arbitrary
    fnm_stem = stem(fnm)
    dbg_path = os.path.join(PATH_BASE, DIR_DSET, config(f'{DIR_DSET}.{dnm}.dir_nm'), 'my_denoise_debugging')
    if verbose:
        ic(fnm, fnm_stem)
        ic(dbg_path)

    df = pd.read_csv(fnm)
    df_de = pd.read_csv(fnm.replace('ECGData', 'ECGDataDenoised'), header=None)
    if verbose:
        ic(fnm)
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
    np.random.seed(config('random_seed'))

    # ic(config('datasets.BIH_MVED'))

    # ic(remove_1st_occurrence('E00002.mat 12 500 5000 05-May-2020 14:50:55', '.mat'))

    ic(get_signal_eg(dnm='G12EC', n=0).shape)
    ic(get_signal_eg(dnm='CHAP_SHAO', n=0))
    ic(get_signal_eg(dnm='CODE_TEST', n=0).shape)

    for dnm_ in config(f'datasets_export.total'):
        path = get_rec_paths(dnm_)[0]
        ic(dnm_, stem(path, ext=True), sizeof_fmt(os.path.getsize(path)))
