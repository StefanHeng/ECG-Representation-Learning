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
    def _post_init(self, arr, normalize: str = 'std', norm_arg: Union[float, int] = 4, return_type='pt'):
        """
        :param normalize: Normalization scheme, one of [`global`, `std`, `norm`, `none`]
            Normalization is done per channel/lead
            If `global`, normalize by global minimum and maximum
            If `std`, normalize by subtracting mean & dividing 1 standard deviation
            If `norm`, normalize by subtracting mean & dividing range based on standard deviation percentile
        :param norm_arg: Intended for `norm` or `std` scheme
            FYI:
                pnorm(1) ~= 0.841
                pnorm(2) ~= 0.977
                pnorm(3) ~= 0.999
                pnorm(4) ~= 0.99997
        """
        assert normalize in ['global', 'std', 'norm', 'none']
        self.norm_meta = None
        self.normalize = normalize
        # TODO: not sure why many functions return nan, even when no nan elements are present
        # shapes are (1, 12, 1)
        if normalize == 'global':
            self.norm_meta = (
                np.nanmin(arr, axis=(0, -1), keepdims=True),
                np.nanmax(arr, axis=(0, -1), keepdims=True)
            )
        elif normalize == 'std':
            assert isinstance(norm_arg, (float, int))
            self.norm_meta = (
                np.nanmean(arr, axis=(0, -1), keepdims=True),
                np.nanstd(arr, axis=(0, -1), keepdims=True) * norm_arg
            )
        elif normalize == 'norm':
            assert isinstance(norm_arg, (float, int))
            p = norm().cdf(norm_arg) * 100
            self.norm_meta = (
                np.nanpercentile(arr, 100-p, axis=(0, -1), keepdims=True),
                np.nanpercentile(arr, p, axis=(0, -1), keepdims=True)
            )
        assert return_type in ['pt', 'np']
        self.return_type = return_type

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dset.shape[0] if self.is_full else self.idxs_processed.size

    def __getitem__(self, idx, normalize: str = None) -> torch.FloatTensor:
        normalize = self.normalize if normalize is None else normalize
        if normalize == 'none':
            arr = self.dset[idx]
        else:
            if normalize in ['global', 'norm']:
                mi, ma = self.norm_meta
                sub, div = mi, (ma - mi)
            else:   # normalize == 'std'
                sub, div = self.norm_meta
                # from icecream import ic
                # ic(mu.shape, sigma.shape, self.dset[idx].shape)
            assert sub.shape[0] == 1 and div.shape[0] == 1
            # from icecream import ic
            # ic(idx)
            if isinstance(idx, int):  # not slice
                sub, div = sub[0], div[0]  # (1, 12, 1) -> (12, 1)
            arr = (self.dset[idx] - sub) / div
        from icecream import ic
        ic(idx, arr.shape, self.dset[idx].shape)
        if self.return_type == 'pt':
            return torch.from_numpy(arr).float()  # cos the h5py stores float64
        else:
            return arr


class NamedDataset(EcgDataset):
    """
    Data samples are from a single H5 file
    """
    def __init__(self, dataset_name, fqs=250, **kwargs):
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

        super()._post_init(arr, **kwargs)


if __name__ == '__main__':
    from icecream import ic

    dnm = 'CHAP_SHAO'

    def sanity_check():
        nd = NamedDataset(dnm, normalize='global')
        ic(len(nd), nd[0].shape)
        ic(nd.norm_meta)
        for i, rec in enumerate(nd[:8]):
            ic(rec.shape, rec[0, :4])
    # sanity_check()

    def check_normalize(n: int = 128):
        # el = EcgLoader(dnm, normalize='global')
        nd = NamedDataset(dnm, normalize='std', norm_arg=3, return_type='np')
        # el = EcgLoader(dnm, normalize='norm', norm_arg=3)
        n_sig, (n_ch, l) = len(nd), nd.dset.shape[1:]
        idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))
        idxs_ch = np.random.randint(n_ch, size=n)
        sigs = nd[idxs_sig][range(n), idxs_ch]

        ratio_oor = np.any((0 > sigs) | (sigs > 1), axis=-1).sum() / n  # Fraction of points that go beyond range 0, 1
        ic(ratio_oor)

        plt.figure(figsize=(18, 6))
        plt.hlines([-1, 0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
        ecg_util.plot_1d(sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11))
        plt.show()
    # check_normalize()

    def check_normalize_channel(n: int = 128):
        nd = NamedDataset(dnm, normalize='std', norm_arg=3, return_type='np')
        n_sig, (n_ch, l) = len(nd), nd.dset.shape[1:]
        for i in range(n_ch):
            idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))
            sigs = nd[idxs_sig][:, i, :]
            sigs_ori = nd.__getitem__(idxs_sig, normalize='none')[:, i, :]

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
            plt.title(f'Normalize sanity check for channel {i}')
            plt.show()
    check_normalize_channel(n=32)

    def check_extracted_dataset():
        for dnm in config('datasets_export.total'):
            el_ = NamedDataset(dnm)
            ic(el_.dset.shape)
    # check_extracted_dataset()
