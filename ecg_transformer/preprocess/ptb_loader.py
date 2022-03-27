"""
Dataloader for downstream supervised fine-tuning on the PTB-XL dataset

Frame the problem as multi-label classification with total of 71 classes spanning 3 aspects,
    each class as a binary probability
"""

from ast import literal_eval
from typing import Sequence
import h5py

import torch

from ecg_transformer.util import *
import ecg_transformer.util.ecg as ecg_util
from ecg_transformer.preprocess import EcgDataset


class PtbxlDataset(EcgDataset):
    DNM = 'PTB_XL'
    N_CLASS = 71

    def __init__(self, idxs: List[int], labels: Sequence[List[int]], normalize='std', norm_arg=3):
        """
        :param idxs: Indices into the original PTB-XL csv rows
            Intended for selecting rows in the original dataset to create splits
        """
        rec = h5py.File(ecg_util.get_denoised_h5_path(PtbxlDataset.DNM))
        dset = rec['data']
        fqs = json.loads(rec.attrs['meta'])['fqs']
        assert fqs == 250  # TODO: generalize?
        self.dset = dset[idxs]  # All data stored in memory; TODO: optimization?
        self.labels = labels

        self.is_full = True  # Assume user passed in processed data; Fit with `EcgDataset.__len__` API
        self.meta = OrderedDict([
            ('frequency', fqs),
            ('normalization-scheme', normalize),
            ('normalization-arg', norm_arg)
        ])

        super()._post_init(self.dset, normalize, norm_arg)

    @staticmethod
    def lbs2multi_hot(lbs: List[int]) -> torch.LongTensor:
        multi_hot: torch.LongTensor = torch.zeros(PtbxlDataset.N_CLASS, dtype=torch.long, device='cpu')
        multi_hot[lbs] = 1
        return multi_hot

    def __getitem__(self, idx):
        return dict(
            sample_values=super().__getitem__(idx),
            labels=PtbxlDataset.lbs2multi_hot(self.labels[idx])
        )


def get_ptbxl_splits() -> Tuple[PtbxlDataset, ...]:
    logger = get_logger('Get PTB-XL splits')
    idxs_processed = list(range(4096))  # TODO: the amount of denoised data
    logger.info(f'Getting PTB-XL splits with n={logi(len(idxs_processed))}... ')
    # pdset = PtbxlDataset()
    # ic(pdset[0], pdset.dset.shape)

    # Use 0-indexed rows, not 1-indexed `ecg_id`s
    df = pd.read_csv(os.path.join(config('path-export'), 'ptb-xl-labels.csv'), usecols=['strat_fold', 'labels'])
    df = df.iloc[idxs_processed]
    df.labels = df.labels.apply(literal_eval)
    # `strat_fold`s are in [1, 10]
    df_tr, df_vl, df_ts = df[df.strat_fold < 9], df[df.strat_fold == 9], df[df.strat_fold == 10]
    n_tr, n_vl, n_ts = len(df_tr), len(df_vl), len(df_ts)
    assert n_tr + n_vl + n_ts == len(df)
    # ic(df, df_tr, df_vl, df_ts)
    logger.info(f'Splits created with sizes {log_dict(dict(train=n_tr, eval=n_vl, test=n_ts))}... ')
    ic(df_tr.labels[0])
    ic(type(df_tr.labels[0]), type(df_tr.labels))
    ic(df_tr.index)
    # dset_tr = PtbxlDataset(df_tr.index.to_numpy(), df_tr.labels)
    # ic(dset_tr)
    # ic(dset_tr[0])
    return tuple(
        PtbxlDataset(dset.index.to_numpy(), dset.labels) for dset in (df_tr, df_vl, df_ts)
    )


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

    dest_tr, dset_vl, dset_ts = get_ptbxl_splits()
    ic(dest_tr[0])
