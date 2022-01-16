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

    def __init__(self, dnm, fqs=250):
        """
        :param dnm: Encoded dataset name
        :param fqs: Frequency for loaded signals, potentially re-sampled
        """
        self.rec = h5py.File(self.get_h5_path(dnm))
        self.dset = self.rec['data']
        ic(self.dset, self.dset.dtype)
        self.attrs = json.loads(self.rec.attrs['meta'])
        assert self.attrs['fqs'] == fqs  # Sanity check

        self.is_full = all(np.any(d != 0) for d in self.dset)
        if not self.is_full:  # TODO: debugging for now, as not all records are processed
            # self.idxs_processed = np.array([np.any(d != 0) for d in self.rec['data']])
            self.idxs_processed = set(idx for idx, d in enumerate(self.dset) if np.any(d != 0))
            # ic(self.idxs_processed, len(self.idxs_processed))
        # ic(self.rec.keys())
        ic(self.dset.shape)
        # ic(self.attrs)
        # ic(list(self.rec.attrs))

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dset.shape[0] if self.is_full else len(self.idxs_processed)

    def __getitem__(self, idx):
        return self.dset[idx]

    @staticmethod
    def get_h5_path(dnm):
        return os.path.join(EcgLoader.PATH_EXP, EcgLoader.D_DSET['rec_fmt_denoised'] % dnm)


if __name__ == '__main__':
    from icecream import ic

    dnm_ = 'CHAP_SHAO'
    el = EcgLoader(dnm_)
    ic(len(el))
    for rec in el[:8]:
        ic(rec)
