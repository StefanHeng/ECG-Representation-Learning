"""
Dataloader for downstream supervised fine-tuning on the PTB-XL dataset

Frame the problem as multi-label classification with total of 71 classes spanning 3 aspects,
    each class as a binary probability
"""

from torch.utils.data import Dataset

from ecg_transformer.util import *


class PtbxlDataset(Dataset):
    DNM = 'PTB_XL'

    def __init__(self, idxs: List[int]):
        """
        :param idxs: Indices into the original PTB-XL csv rows
            Intended for selecting rows in the original dataset to create splits
        """
        self.rec = h5py.File(get_denoised_h5_path(PtbxlDataset.DNM))
        self.dset = self.rec['data']
        self.attrs = json.loads(self.rec.attrs['meta'])
        assert self.attrs['fqs'] == 250  # TODO: generalize?

