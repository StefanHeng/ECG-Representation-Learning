"""
ECG signal loading

Intended for self-supervised ECG representation pretraining
"""


from scipy.stats import norm

from ecg_transformer.util.util import *


class EcgLoader:
    """
    Load pre-processed (de-noised) heartbeat records from each dataset

    See `data_export.py`, `DataExport.m`

    The HDF5 data files are of shape N x C x L where
        N: # signal records
        C: # channel in a signal
        L: # sample per channel
    """
    def __init__(self, dataset_name, fqs=250, normalize: str = 'std', norm_arg: Union[float, int] = 4):
        """
        :param dataset_name: Encoded dataset name
        :param fqs: Frequency for loaded signals, potentially re-sampled
        :param normalize: Normalization scheme, one of [`global`, `std`, `norm`, `none`]
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
        self.rec = h5py.File(get_denoised_h5_path(dataset_name))
        self.dset = self.rec['data']
        self.attrs = json.loads(self.rec.attrs['meta'])
        assert self.attrs['fqs'] == fqs  # Sanity check

        # TODO: debugging for now, as not all records are processed
        self.is_full = all(np.any(d != 0) for d in self.dset)  # cos potentially costly to load entire data
        if not self.is_full:
            self.idxs_processed = np.array([idx for idx, d in enumerate(self.dset) if np.any(d != 0)])
            self.set_processed = set(self.idxs_processed)
            arr = self.dset[self.idxs_processed]
        else:
            arr = self.dset[:]
        # arr = arr.astype(np.float32)
        # ic(arr.shape, type(arr), arr.dtype)
        assert not np.all(np.isnan(arr))
        # ic(arr.max(axis=(-2, -1)))
        # ic(arr.max(axis=0))
        # ic(arr.max(axis=(-2, -1)).max())

        assert normalize in ['global', 'std', 'norm', 'none']
        self.norm_meta = None
        self.normalize = normalize
        # TODO: not sure why many functions return nan, even when no nan elements are present
        if normalize == 'global':
            self.norm_meta = np.nanmin(arr), np.nanmax(arr)
        elif normalize == 'std':
            assert isinstance(norm_arg, (float, int))
            self.norm_meta = np.nanmean(arr), np.nanstd(arr) * norm_arg
            ic(self.norm_meta)
        elif normalize == 'norm':
            assert isinstance(norm_arg, (float, int))
            p = norm().cdf(norm_arg) * 100
            self.norm_meta = np.nanpercentile(arr, 100-p), np.nanpercentile(arr, p)

    @property
    def shape(self):
        return self.dset.shape

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dset.shape[0] if self.is_full else len(self.set_processed)

    def __getitem__(self, idx):
        if self.normalize in ['global', 'norm']:
            mi, ma = self.norm_meta
            return (self.dset[idx] - mi) / (ma - mi)
        elif self.normalize == 'std':
            mu, sigma = self.norm_meta
            return (self.dset[idx] - mu) / sigma
        else:
            return self.dset[idx]


if __name__ == '__main__':
    from icecream import ic

    def sanity_check():
        dnm = 'CHAP_SHAO'
        el = EcgLoader(dnm, normalize='global')
        ic(len(el), el.shape, el[0].shape)
        ic(el.norm_meta)
        for i, rec in enumerate(el[:8]):
            np.testing.assert_array_equal(rec[0, :8], el[i, 0, :8])
            ic(rec.shape, rec[0, :4])
    # sanity_check()

    def check_normalize():
        dnm = 'CHAP_SHAO'
        # el = EcgLoader(dnm, normalize='global')
        el = EcgLoader(dnm, normalize='std', norm_arg=3)
        # el = EcgLoader(dnm, normalize='norm', norm_arg=3)
        n_sig, n_ch, l = el.shape
        n = 512
        idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))  # Per hdf5 array indexing
        idxs_ch = np.random.randint(n_ch, size=n)
        sigs = el[idxs_sig][range(n), idxs_ch]

        ic(sigs.shape)
        ic(np.any((0 > sigs) | (sigs > 1), axis=-1).sum() / n)  # Fraction of points that go beyond range 0, 1

        plt.figure(figsize=(18, 6))
        plt.hlines([-1, 0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
        plot_1d(sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11))
        plt.show()
    check_normalize()

    def check_extracted_dataset():
        for dnm in config('datasets_export.total'):
            el_ = EcgLoader(dnm)
            ic(el_.dset.shape)
    # check_extracted_dataset()
