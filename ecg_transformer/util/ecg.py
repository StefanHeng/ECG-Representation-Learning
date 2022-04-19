import os
import math
import glob
from typing import Union

import numpy as np
import pandas as pd
import h5py
import scipy.optimize
import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
import seaborn as sns

from .util import *
from .data_path import PATH_BASE, DIR_DSET
from .check_args import ca


def plot_1d(arr, label=None, title=None, save=False, s=None, e=None, new_fig=True, plot_kwargs=None, show=True):
    """ Plot potentially multiple 1D signals """
    kwargs = LN_KWARGS
    if plot_kwargs is not None:
        kwargs |= plot_kwargs

    if new_fig:
        plt.figure(figsize=(18, 6))
    if not isinstance(arr, list):
        arr = list(arr) if isinstance(arr, np.ndarray) else arr[arr]
    if not isinstance(label, list):
        label = [label] * len(arr)
    lbl = [None for _ in arr] if label is None else label
    cs = sns.color_palette('husl', n_colors=len(arr))

    def _plot(a_, lb_, c_):
        a_ = a_[s:e]
        args = dict(c=c_) | kwargs
        plt.gca().plot(np.arange(a_.size), a_, label=lb_, **args)
    for a, lb, c in zip(arr, lbl, cs):
        _plot(a, lb, c)

    if label:
        handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    if title:
        plt.title(title)
    if new_fig:
        save_fig(title, save)
    if show:
        plt.show()


def plot_ecg(
        arr: np.ndarray, title: str = None, ax=None, legend: bool = True, gap_factor: float = 1.0,
        xlabel: str = 'Timestep (potentially resampled)', ylabel: str = 'Amplitude, normalized (mV)'
):
    n_lead = arr.shape[0]
    height = (abs(np.max(arr)) + abs(np.min(arr))) / 4 * gap_factor  # Empirical

    if not ax:
        plt.figure(figsize=(16, 13))
        ax = plt.gca()

    ylb_ori = ((np.arange(n_lead) - n_lead + 1) * height)[::-1]
    ylb_new = ['I', 'II', 'III', 'avR', 'avL', 'avF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # TODO; verify order
    cs = sns.color_palette('husl', n_colors=n_lead)
    for i, row in enumerate(arr):
        offset = height * i
        x = np.arange(row.size)
        y = row - offset
        ax.plot(x, y, label=ylb_new[i], marker='o', ms=0.3, lw=0.25, c=cs[i])
        ax.axhline(y=-offset, lw=0.2)

    title = title or 'ECG 12-lead plot'
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(ylb_ori, ylb_new)
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1))
    if not ax:
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
    (a_, b_), p_cov = scipy.optimize.curve_fit(f=pow_law, xdata=x, ydata=y, p0=(x[0]*2, -1))

    ret = (a_, b_)
    if return_fit:
        scale = 1 if return_fit is True else return_fit
        x_plot = np.linspace(x.min(), x.max(), num=x.size * scale)
        y_fit = pow_law(x_plot, a_, b_)
        ret = ret, (x_plot, y_fit)
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


def get_processed_path():
    """
    Path where the processed records are stored
    """
    return os.path.join(PATH_BASE, DIR_DSET, config('datasets.my.dir_nm'))


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
    kwargs = dict(sampto=ln)
    kwargs = {k: v for k, v in kwargs.items() if k is not None}
    return wfdb.rdrecord(rec_path[:rec_path.index('.')], **kwargs)


def fnm2sigs(fnm, dnm, to_fp32: bool = True):
    if dnm == 'CHAP-SHAO':
        arr = pd.read_csv(fnm).to_numpy().T
    elif dnm == 'CODE-TEST':  # one hdf5 file with all recordings
        assert isinstance(fnm, int)
        if not hasattr(config, 'ct_tracings'):
            fnms = get_rec_paths(dnm)
            assert len(fnms) == 1
            fnm2sigs.ct_tracings = h5py.File(fnm, 'r')

        arr = fnm2sigs.ct_tracings['tracings'][fnm]
    else:
        arr = wfdb.rdsamp(fnm.removesuffix(config(f'datasets.{dnm}.rec_ext')))[0].T  # (signal, meta)
    if to_fp32:
        arr = arr.astype(np.float32)  # for faster processing, & for ML anyway
    return arr


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
        return fnm2sigs(get_rec_paths(dnm)[n], dnm)
    elif dnm == 'CODE_TEST':
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


def get_processed_record_path(dataset_name, type: str = 'denoised'):
    ca(type=type, dataset_name=dataset_name)
    fmt = 'rec_fmt_denoised' if type == 'denoised' else 'rec_fmt'
    return os.path.join(get_processed_path(), config(f'datasets.my.{fmt}') % dataset_name)


if __name__ == '__main__':
    from icecream import ic

    ic(get_signal_eg(dnm='G12EC', n=0).shape)
    ic(get_signal_eg(dnm='CHAP_SHAO', n=0))
    ic(get_signal_eg(dnm='CODE_TEST', n=0).shape)

    for dnm_ in config(f'datasets_export.total'):
        path = get_rec_paths(dnm_)[0]
        ic(dnm_, stem(path, ext=True), sizeof_fmt(os.path.getsize(path)))
