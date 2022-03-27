"""
Dataloader for downstream supervised fine-tuning on the PTB-XL dataset

Frame the problem as multi-label classification with total of 71 classes spanning 3 aspects,
    each class as a binary probability
"""

from ecg_transformer.util import *
from ecg_transformer.preprocess import EcgDataset


class PtbxlDataset(EcgDataset):
    DNM = 'PTB_XL'

    def __init__(self, idxs: List[int], normalize='std', norm_arg=3):
        """
        :param idxs: Indices into the original PTB-XL csv rows
            Intended for selecting rows in the original dataset to create splits
        """
        rec = h5py.File(get_denoised_h5_path(PtbxlDataset.DNM))
        dset = rec['data']
        attrs = json.loads(rec.attrs['meta'])
        assert attrs['fqs'] == 250  # TODO: generalize?
        self.dset = dset[idxs]  # All data stored in memory; TODO: optimization?

        super()._post_init(self.dset, normalize, norm_arg)


if __name__ == '__main__':
    from icecream import ic

    def check_ptb_denoise_progress():
        from ecg_transformer.preprocess import NamedDataset

        dnm = 'PTB_XL'
        nd = NamedDataset(dnm, normalize='std')
        ic(len(nd), nd.dset.shape, nd[0].shape)
        ic(nd.norm_meta)
        for i, rec in enumerate(nd[:8]):
            ic(rec.shape, rec[0, :4])
    # check_ptb_denoise_progress()

    pdset = PtbxlDataset(list(range(2304)))  # TODO: the amount of denoised data
    ic(pdset[0], pdset.dset.shape)
