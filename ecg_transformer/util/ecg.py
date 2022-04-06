import math
import glob

import h5py
import scipy.optimize
import wfdb
from wfdb import processing

from .util import *


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

    def _plot(a_, lb_, c):
        a_ = a_[s:e]
        plt.gca().plot(np.arange(a_.size), a_, label=lb_, c=c, **kwargs)
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


def r2(y, y_fit):
    return 1 - (np.square(y - y_fit).sum() / np.square(y - np.mean(y)).sum())


def fit_power_law(x: np.ndarray, y: np.ndarray, return_fit: Union[int, bool] = False):
    """
    :return: 2-tuple of (coefficient, exponent) for power law
        If `return_fit` is True, return additionally 2-tuple of (fitted x, fitted y)
            If integer given, the fitted curve is returned by scale
    """
    # from icecream import ic
    def pow_law(x_, a, b):
        # ic(x_, a, b)
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

    if dnm == 'CHAP-SHAO':
        return pd.read_csv(fnm).to_numpy().T
    elif dnm == 'CODE-TEST':
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


def get_denoised_h5_path(dnm):
    if not hasattr(get_denoised_h5_path, 'd_dset'):
        get_denoised_h5_path.d_dset = config(f'datasets.my')
    d_dset = get_denoised_h5_path.d_dset
    if not hasattr(get_denoised_h5_path, 'path_exp'):
        # Path where the processed records are stored
        get_denoised_h5_path.path_exp = os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'])
    return os.path.join(get_denoised_h5_path.path_exp, d_dset['rec_fmt_denoised'] % dnm)


if __name__ == '__main__':
    from icecream import ic

    ic(get_signal_eg(dnm='G12EC', n=0).shape)
    ic(get_signal_eg(dnm='CHAP_SHAO', n=0))
    ic(get_signal_eg(dnm='CODE_TEST', n=0).shape)

    for dnm_ in config(f'datasets_export.total'):
        path = get_rec_paths(dnm_)[0]
        ic(dnm_, stem(path, ext=True), sizeof_fmt(os.path.getsize(path)))
