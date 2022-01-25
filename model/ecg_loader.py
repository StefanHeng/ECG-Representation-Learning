from scipy.stats import norm

from util.util import *


class EcgLoader:
    """
    Load pre-processed (de-noised) heartbeat records from each dataset

    See `data_export.py`, `DataExport.m`

    The HDF5 data files are of shape N x C x L where
        N: # signal records
        C: # channel in a signal
        L: # sample per channel
    """

    D_DSET = config(f'datasets.my')
    PATH_EXP = os.path.join(PATH_BASE, DIR_DSET, D_DSET['dir_nm'])  # Path where the processed records are stored

    def __init__(self, dnm, fqs=250, normalize: Union[bool, float, int] = 4):
        """
        :param dnm: Encoded dataset name
        :param fqs: Frequency for loaded signals, potentially re-sampled
        :param normalize: Normalization scheme
            If True, normalize by global minimum and maximum
            If number given, normalize by global mean and multiplied standard deviation as a percentile
            FYI:
                pnorm(1) ~= 0.841
                pnorm(2) ~= 0.977
                pnorm(3) ~= 0.999
                pnorm(4) ~= 0.99997
        """
        self.rec = h5py.File(self.get_h5_path(dnm))
        self.dset = self.rec['data']
        self.attrs = json.loads(self.rec.attrs['meta'])
        assert self.attrs['fqs'] == fqs  # Sanity check

        # TODO: debugging for now, as not all records are processed
        self.is_full = all(np.any(d != 0) for d in self.dset)
        if not self.is_full:
            self.idxs_processed = np.array([idx for idx, d in enumerate(self.dset) if np.any(d != 0)])
            self.set_processed = set(self.idxs_processed)

        self.normalize = bool(normalize)
        self.normalize_norm = isinstance(normalize, (float, int))

        arr = self.dset[self.idxs_processed]
        self.range = arr.min(), arr.max()
        scale = normalize if self.normalize_norm else 4  # For computing `norm_range` anyway
        p = norm().cdf(scale) * 100
        self.norm_range = np.percentile(arr, 100-p), np.percentile(arr, p)

    @property
    def shape(self):
        return self.dset.shape

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dset.shape[0] if self.is_full else len(self.set_processed)

    def __getitem__(self, idx):
        if self.normalize:
            mi, ma = self.norm_range if self.normalize_norm else self.range
            return (self.dset[idx] - mi) / (ma - mi)
        else:
            return self.dset[idx]

    @staticmethod
    def get_h5_path(dnm):
        return os.path.join(EcgLoader.PATH_EXP, EcgLoader.D_DSET['rec_fmt_denoised'] % dnm)


if __name__ == '__main__':
    from icecream import ic

    dnm_ = 'CHAP_SHAO'
    el = EcgLoader(dnm_)

    def sanity_check():
        ic(len(el), el.shape, el[0].shape)
        ic(el.range)
        for i, rec in enumerate(el[:8]):
            np.testing.assert_array_equal(rec[0, :8], el[i, 0, :8])
            ic(rec.shape, rec[0, :4])
    # sanity_check()

    def check_normalize():
        n_sig, n_ch, l = el.shape
        n = 512
        idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))  # Per hdf5 array indexing
        idxs_ch = np.random.randint(n_ch, size=n)
        sigs = el[idxs_sig][range(n), idxs_ch]

        ic(np.any((0 > sigs) | (sigs > 1), axis=-1).sum() / n)  # Fraction of points that go beyond range 0, 1

        plt.figure(figsize=(18, 6))
        plt.hlines([0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
        plot_1d(sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11))
        plt.show()
    check_normalize()
