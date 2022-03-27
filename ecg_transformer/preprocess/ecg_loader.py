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
    def _post_init(self, arr, normalize: str = 'std', norm_arg: Union[float, int] = 4):
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

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dset.shape[0] if self.is_full else self.idxs_processed.sum()

    def __getitem__(self, idx):
        if self.normalize in ['global', 'norm']:
            mi, ma = self.norm_meta
            arr = (self.dset[idx] - mi) / (ma - mi)
        elif self.normalize == 'std':
            mu, sigma = self.norm_meta
            arr = (self.dset[idx] - mu) / sigma
        else:
            arr = self.dset[idx]
        return torch.from_numpy(arr).float()  # cos the h5py stores float64


class NamedDataset(EcgDataset):
    """
    Data samples are from a single H5 file
    """
    def __init__(self, dataset_name, fqs=250, normalize='std', norm_arg=3):
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

        super()._post_init(arr, normalize, norm_arg)


if __name__ == '__main__':
    from icecream import ic

    def sanity_check():
        dnm = 'CHAP_SHAO'
        nd = NamedDataset(dnm, normalize='global')
        ic(len(nd), nd[0].shape)
        ic(nd.norm_meta)
        for i, rec in enumerate(nd[:8]):
            ic(rec.shape, rec[0, :4])
    sanity_check()

    def check_normalize():
        dnm = 'CHAP_SHAO'
        # el = EcgLoader(dnm, normalize='global')
        nd = NamedDataset(dnm, normalize='std', norm_arg=3)
        ic(nd.norm_meta)
        # el = EcgLoader(dnm, normalize='norm', norm_arg=3)
        n_sig, n_ch, l = nd.dset.shape
        n = 512
        idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))  # Per hdf5 array indexing
        idxs_ch = np.random.randint(n_ch, size=n)
        sigs = nd[idxs_sig][range(n), idxs_ch]

        ic(sigs.shape)
        ic(np.any((0 > sigs) | (sigs > 1), axis=-1).sum() / n)  # Fraction of points that go beyond range 0, 1

        plt.figure(figsize=(18, 6))
        plt.hlines([-1, 0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
        ecg_util.plot_1d(sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11))
        plt.show()
    check_normalize()

    def check_extracted_dataset():
        for dnm in config('datasets_export.total'):
            el_ = NamedDataset(dnm)
            ic(el_.dset.shape)
    # check_extracted_dataset()
