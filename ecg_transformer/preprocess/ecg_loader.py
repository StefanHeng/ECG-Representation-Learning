"""
ECG signal loading

Intended for self-supervised ECG representation pretraining
"""
import h5py

import torch
from scipy.stats import norm
from torch.utils.data import Dataset

from ecg_transformer.util import *
import ecg_transformer.util.ecg as ecg_util


_NormArg = Union[str, Tuple[str, Union[float, int]]]
NormArg = Union[_NormArg, List[_NormArg]]


class NormTransform:
    def __init__(self, arr, scheme: str = 'std', arg: Union[float, int] = None):
        assert scheme in ['global', 'std', 'norm', 'none']
        self.norm_meta = None
        self.scheme = scheme
        # TODO: not sure why many functions return nan, even when no nan elements are present
        if scheme in ['std', 'norm']:
            if arg is None:
                arg = 3
            else:
                assert isinstance(arg, (float, int))
            self.arg = arg
        else:
            self.arg = None

        # shapes are (1, 12, 1)
        if scheme == 'global':
            self.norm_meta = (
                np.nanmin(arr, axis=(0, -1), keepdims=True),
                np.nanmax(arr, axis=(0, -1), keepdims=True)
            )
        elif scheme == 'std':
            self.norm_meta = (
                np.nanmean(arr, axis=(0, -1), keepdims=True),
                np.nanstd(arr, axis=(0, -1), keepdims=True) * arg
            )
        elif scheme == 'norm':
            p = norm().cdf(arg) * 100
            self.norm_meta = (
                np.nanpercentile(arr, 100-p, axis=(0, -1), keepdims=True),
                np.nanpercentile(arr, p, axis=(0, -1), keepdims=True)
            )

    def __call__(self, arr, squeeze_1st=False):
        if self.scheme == 'none':
            return arr
        else:
            if self.scheme in ['global', 'norm']:
                mi, ma = self.norm_meta
                sub, div = mi, (ma - mi)
            else:  # normalize == 'std'
                sub, div = self.norm_meta
            if squeeze_1st:
                sub, div = sub[0], div[0]  # (1, 12, 1) -> (12, 1)
            return (arr - sub) / div

    @staticmethod
    def _a2s(arr):
        return np.array2string(arr.flatten(), precision=2, separator=',', suppress_small=True, max_line_width=256)

    def __repr__(self):
        # if self.scheme != 'none':
        #     a, b = self.norm_meta
        #     a, b = NormTransform._a2s(a), NormTransform._a2s(b)
        #     str_meta = f'meta=({a}, {b})'
        # else:
        #     str_meta = ''
        str_arg = f'arg={self.arg}' if self.arg is not None else ''
        return f'<{self.__class__.__qualname__} {self.scheme} {str_arg}>'


class EcgDataset(Dataset):
    """
    Load pre-processed (de-noised) heartbeat records given a single dataset

    See `data_export.py`, `DataExport.m`

    The HDF5 data files are of shape N x C x L where
        N: # signal records
        C: # channel in a signal
        L: # sample per channel

    Children should define an index-able `dset` field
    """
    def __init__(self, patch_size: int = 64, pad: bool = True):
        self.patch_size, self.pad = patch_size, pad

    def _post_init(self, arr: np.ndarray, normalize: NormArg = (('norm', 3), ('std', 1)), return_type: str = 'pt'):
        """
        :param normalize: Normalization or a sequence of normalizations, as 2-tuples of
            normalize scheme: one of [`global`, `std`, `norm`, `none`]
                Normalization is done per channel/lead
                If `global`, normalize by global minimum and maximum
                If `std`, normalize by subtracting mean & dividing 1 standard deviation
                If `norm`, normalize by subtracting mean & dividing range based on standard deviation percentile
            normalize argument: Intended for `norm` or `std` scheme
                FYI:
                    pnorm(1) ~= 0.841
                    pnorm(2) ~= 0.977
                    pnorm(3) ~= 0.999
                    pnorm(4) ~= 0.99997
        """
        if isinstance(normalize, str) or isinstance(normalize, (tuple, list)) and not isinstance(normalize[0], tuple):
            norm_args = [normalize]
        else:
            norm_args = normalize
        norm_args = [(pr if isinstance(pr, tuple) else (pr,)) for pr in norm_args]
        assert all((isinstance(pr, tuple) and len(pr) in [1, 2]) for pr in norm_args)
        self.normalizers = []
        for pr in norm_args:
            normzer = NormTransform(arr, *pr)
            self.normalizers.append(normzer)
            arr = normzer(arr, squeeze_1st=False)  # the normalizations are done sequentially

        assert return_type in ['pt', 'np']
        self.return_type = return_type

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dset.shape[0] if self.is_full else self.idxs_processed.size

    def __getitem__(self, idx) -> torch.FloatTensor:
        lst_squeeze_1st = [isinstance(idx, int)] * len(self.normalizers)  # if `idx` is slice
        arr = self.dset[idx]
        for normalizer, squeeze_1st in zip(self.normalizers, lst_squeeze_1st):
            arr = normalizer(arr, squeeze_1st)
        if self.pad:  # pad along the last dimension, in the end
            l = arr.shape[-1]
            n_pad = self.patch_size - (l % self.patch_size)
            if n_pad != 0:
                pad_width = [(0, 0)] * arr.ndim
                pad_width[-1] = (0, n_pad)
                arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
        if self.return_type == 'pt':  # TODO: debugging
            return torch.from_numpy(arr).float()[:128]  # cos the h5py stores float64
        else:
            return arr


class NamedDataset(EcgDataset):
    """
    Data samples are from a single H5 file
    """
    def __init__(self, dataset_name, fqs=250, init_kwargs=None, post_init_kwargs=None):
        if init_kwargs is None:
            init_kwargs = dict()
        super().__init__(**init_kwargs)
        self.rec = h5py.File(ecg_util.get_denoised_h5_path(dataset_name))
        self.dset = self.rec['data']
        self.attrs = json.loads(self.rec.attrs['meta'])
        assert self.attrs['fqs'] == fqs  # Sanity check

        # TODO: debugging for now, as not all records are processed
        self.is_full = all(np.any(d != 0) for d in self.dset)  # cos potentially costly to load entire data
        if not self.is_full:
            self.idxs_processed = np.array([idx for idx, d in enumerate(self.dset) if np.any(d != 0)])
            arr = self.dset[self.idxs_processed]
        else:
            arr = self.dset[:]
        assert not np.all(np.isnan(arr))

        super()._post_init(arr, **post_init_kwargs)


if __name__ == '__main__':
    from icecream import ic

    dnm = 'CHAP_SHAO'

    def sanity_check():
        nd = NamedDataset(dnm, normalize='global')
        ic(len(nd), nd[0].shape)
        ic(nd.normalizers)
        for i, rec in enumerate(nd[:8]):
            ic(rec.shape, rec[0, :4])
    # sanity_check()

    def check_norm_meta():
        ic(NamedDataset(dnm, normalize='global').normalizers)
        ic(NamedDataset(dnm, normalize=('std', 3)).normalizers)
        ic(NamedDataset(dnm, normalize=('norm', 3)).normalizers)
    # check_norm_meta()

    def check_normalize(n: int = 128):
        # el = EcgLoader(dnm, normalize='global')
        # nd = NamedDataset(dnm, normalize='std', return_type='np')
        # el = EcgLoader(dnm, normalize='norm')
        # increase the 1st stage norm, to clip as many signals to [0, 1]
        # nd = NamedDataset(dnm, return_type='np', normalize=[('norm', 3)])
        nd = NamedDataset(dnm, return_type='np', normalize=[('norm', 3), ('std', 1)])  # default 2-stage normalization
        ic(nd.normalizers)
        n_sig, (n_ch, l) = len(nd), nd.dset.shape[1:]
        idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))
        idxs_ch = np.random.randint(n_ch, size=n)
        sigs = nd[idxs_sig][range(n), idxs_ch]

        # Fraction of signals, points that go beyond range 0, 1
        oor = (0 > sigs) | (sigs > 1)
        oor = dict(signal=np.any(oor, axis=-1).sum() / n, sample=oor.sum() / oor.size)
        ic(oor)

        plt.figure(figsize=(18, 6))
        plt.hlines([-1, 0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
        ecg_util.plot_1d(sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11))
        plt.show()
    check_normalize()

    def check_normalize_channel(n: int = 128):
        nd = NamedDataset(dnm, return_type='np', normalize=[('norm', 3), ('std', 1)])
        nd_n = NamedDataset(dnm, return_type='np', normalize='none')
        n_sig, (n_ch, l) = len(nd), nd.dset.shape[1:]
        for i in range(n_ch):
            idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))
            sigs = nd[idxs_sig][:, i, :]
            sigs_ori = nd_n[idxs_sig][:, i, :]

            ratio_oor = np.any((0 > sigs) | (sigs > 1), axis=-1).sum() / n
            ic(ratio_oor)

            plt.figure(figsize=(18, 6))
            plt.hlines([-1, 0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
            ecg_util.plot_1d(
                sigs_ori, label='Signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11, c='r'), show=False
            )
            ecg_util.plot_1d(
                sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11, c='m'), show=False
            )
            plt.title(f'Normalize sanity check for channel {i+1}')
            plt.show()
    # check_normalize_channel(n=32)

    def check_extracted_dataset():
        for dnm in config('datasets_export.total'):
            el_ = NamedDataset(dnm)
            ic(el_.dset.shape)
    # check_extracted_dataset()
