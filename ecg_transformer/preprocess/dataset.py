"""
ECG signal dataset

Intended for self-supervised ECG representation pretraining
"""
import h5py

import torch
from torchvision.transforms import Compose  # just a nice wrapper, and not 2D image specific
from torch.utils.data import Dataset

from ecg_transformer.util import *
import ecg_transformer.util.ecg as ecg_util
from ecg_transformer.preprocess.transform import NormArg, Normalize, DynamicNormalize, Transform


class EcgDataset(Dataset):
    """
    Load pre-processed (de-noised) heartbeat records given a single dataset

    See `data_export.py`, `DataExport.m`

    The HDF5 data files are from a single H5 file, of shape N x C x L where
        N: # signal records
        C: # channel in a signal
        L: # sample per channel
    """
    def __init__(
            self, dataset: str = 'PTB-XL', subset: Union[bool, Any] = None,
            fqs=250, return_type: str = 'np',
            normalize: NormArg = 'std', transform: Union[Transform, List[Transform]] = None
    ):
        self.rec = h5py.File(ecg_util.get_denoised_h5_path(dataset))
        self.dataset: h5py.Dataset = self.rec['data']
        self.attrs = json.loads(self.rec.attrs['meta'])
        assert self.attrs['fqs'] == fqs  # Sanity check

        if subset is not None and subset is not False:
            # all data stored in memory; TODO: optimization?
            self.dataset: np.ndarray = self.dataset[subset]
            self.is_full = True
            arr = self.dataset[:]
        else:
            # TODO: debugging for now, as not all records are processed
            self.is_full = all(np.any(d != 0) for d in self.dataset)  # cos potentially costly to load entire data
            if not self.is_full:
                self.idxs_processed = np.array([idx for idx, d in enumerate(self.dataset) if np.any(d != 0)])
                arr = self.dataset[self.idxs_processed]
            else:
                arr = self.dataset[:]
            assert not np.all(np.isnan(arr))

        return_types = ['pt', 'np']
        assert return_type in return_types, \
            f'Unexpected return_type: expect one of {logi(return_type)}, got {logi(return_type)}'

        self.return_type = return_type

        if isinstance(normalize, dict):
            tsf = [Normalize(**normalize)]
        elif normalize:
            tsf = [DynamicNormalize(arr, normalize)]
        else:
            tsf = []
        if transform:
            tsf.extend(transform if isinstance(transform, list) else [transform])

        self.meta = OrderedDict([
            ('frequency', fqs),
            ('transform', [tsf.__repr__()] if tsf else None),
        ])
        self.transform = Compose(tsf) if tsf else None

    def __len__(self):
        """
        :return: Number of records
        """
        return self.dataset.shape[0] if self.is_full else self.idxs_processed.size

    def __getitem__(self, idx) -> Union[np.ndarray, torch.Tensor]:
        arr = self.dataset[idx].astype(np.float32)  # cos the h5py stores float64
        if self.transform:
            arr = self.transform(arr)
        if self.return_type == 'pt':
            return torch.from_numpy(arr)  # somehow without
        else:
            return arr


if __name__ == '__main__':
    from icecream import ic

    dnm = 'CHAP_SHAO'

    def sanity_check():
        nd = EcgDataset(dnm, normalize='global')
        ic(len(nd), nd[0].shape)
        ic(nd.transform)
        for i, rec in enumerate(nd[:8]):
            ic(rec.shape, rec[0, :4])
    sanity_check()

    def check_extracted_dataset():
        for dnm_ in config('datasets_export.total'):
            el_ = EcgDataset(dnm_)
            ic(el_.dataset.shape)
    # check_extracted_dataset()
