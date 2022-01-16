from util import *


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

    def __init__(self, dnm, fqs=250, normalize=True):
        """
        :param dnm: Encoded dataset name
        :param fqs: Frequency for loaded signals, potentially re-sampled
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

        self.normalize = normalize

    @property
    def range(self) -> tuple[float, float]:
        """
        :return: (min, max) of the signal records
        """
        arr = self.dset[self.idxs_processed]
        # ic(arr, arr.shape, type(arr))
        return arr.min(), arr.max()

    @property
    def shape(self):
        return self.dset.shape

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dset.shape[0] if self.is_full else len(self.set_processed)

    def __getitem__(self, idx):
        # ic(idx)
        # assert self.is_full or idx in self.set_processed
        if self.normalize:
            mi, ma = self.range
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
    ic(len(el), el.shape, el[0].shape)
    ic(el.range)
    for i, rec in enumerate(el[:8]):
        np.testing.assert_array_equal(rec[0, :8], el[i, 0, :8])
        ic(rec.shape, rec[0, :4])
