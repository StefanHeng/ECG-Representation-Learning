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
from ecg_transformer.preprocess import NormArg, EcgDataset


def export_ptbxl_labels():
    """
    Export PTB-XL dataset labels

    All keys in the `scp_code` are considered ground truth binary labels
        Downside: the likelihood are ignored, effectively treating each key as 100% confidence
    """
    from icecream import ic

    path = os.path.join(PATH_BASE, DIR_DSET, config('datasets.PTB_XL.dir_nm'), 'ptbxl_database.csv')
    df = pd.read_csv(path, usecols=['ecg_id', 'patient_id', 'scp_codes', 'strat_fold'], index_col=0)
    df.patient_id = df.patient_id.astype(int)
    df.scp_codes = df.scp_codes.apply(literal_eval)
    _d_codes = config('datasets.PTB_XL.code')
    d_codes, code2id = _d_codes['codes'], _d_codes['code2id']

    def map_row(row: pd.Series):
        codes = list(row.scp_codes.keys())
        assert all(c in d_codes for c in codes)
        return sorted(code2id[c] for c in codes)

    df['labels'] = df.apply(map_row, axis=1)
    ic(df)

    df.to_csv(os.path.join(config('path-export'), 'ptb-xl-labels.csv'))


class PtbxlDataset(EcgDataset):
    DNM = 'PTB_XL'
    N_CLASS = 71

    def __init__(self, idxs: List[int], labels: Sequence[List[int]], init_kwargs=None, post_init_kwargs=None):
        """
        :param idxs: Indices into the original PTB-XL csv rows
            Intended for selecting rows in the original dataset to create splits
        """
        if init_kwargs is None:
            init_kwargs = dict()
        if post_init_kwargs is None:
            post_init_kwargs = dict()
        super().__init__(**init_kwargs)
        rec = h5py.File(ecg_util.get_denoised_h5_path(PtbxlDataset.DNM))
        dset = rec['data']
        fqs = json.loads(rec.attrs['meta'])['fqs']
        assert fqs == 250  # TODO: generalize?
        self.dset: np.ndarray = dset[idxs]  # All data stored in memory; TODO: optimization?
        self.labels = labels

        self.is_full = True  # Assume user passed in processed data; Fit with `EcgDataset.__len__` API
        super()._post_init(self.dset, **post_init_kwargs)
        self.meta = OrderedDict([
            ('frequency', fqs),
            ('normalization', self.normalizers.__repr__()),
        ])

    @staticmethod
    def lbs2multi_hot(lbs: List[int], return_float=False) -> torch.Tensor:
        multi_hot = torch.zeros(PtbxlDataset.N_CLASS, dtype=torch.float32 if return_float else torch.long, device='cpu')
        multi_hot[lbs] = 1
        return multi_hot

    def __getitem__(self, idx) -> Dict[str, torch.FloatTensor]:
        return dict(
            sample_values=super().__getitem__(idx),
            labels=PtbxlDataset.lbs2multi_hot(self.labels[idx], return_float=True)
        )


def get_ptbxl_splits(
        n_sample: int = None, dataset_args: Dict = None
) -> Tuple[PtbxlDataset, PtbxlDataset, PtbxlDataset]:
    from icecream import ic
    ic(n_sample)
    logger = get_logger('Get PTB-XL splits')
    idxs_processed = list(range(4224))  # TODO: the amount of denoised data
    logger.info(f'Getting PTB-XL splits with n={logi(len(idxs_processed))}... ')

    # Use 0-indexed rows, not 1-indexed `ecg_id`s
    df = pd.read_csv(os.path.join(get_processed_path(), 'ptb-xl-labels.csv'), usecols=['strat_fold', 'labels'])
    df = df.iloc[idxs_processed]
    df.labels = df.labels.apply(literal_eval)
    # `strat_fold`s are in [1, 10]
    df_tr, df_vl, df_ts = df[df.strat_fold < 9], df[df.strat_fold == 9], df[df.strat_fold == 10]
    if n_sample is not None:
        df_tr, df_vl, df_ts = df_tr.iloc[:n_sample], df_vl.iloc[:n_sample], df_ts.iloc[:n_sample]
    n_tr, n_vl, n_ts = len(df_tr), len(df_vl), len(df_ts)
    if n_sample is None:
        assert n_tr + n_vl + n_ts == len(df)
    logger.info(f'Splits created with sizes {log_dict(dict(train=n_tr, eval=n_vl, test=n_ts))}... ')

    if dataset_args is None:
        dataset_args = dict()

    def get_dset(df_) -> PtbxlDataset:
        # so that indexing into `labels` is 0-indexed
        return PtbxlDataset(df_.index.to_numpy(), df_.reset_index(drop=True).labels, **dataset_args)
    return get_dset(df_tr), get_dset(df_vl), get_dset(df_ts)


if __name__ == '__main__':
    from icecream import ic

    def check_ptb_denoise_progress():
        from ecg_transformer.preprocess import NamedDataset

        dnm = 'PTB_XL'
        nd = NamedDataset(dnm, post_init_kwargs=dict(normalize='std'))
        ic(len(nd), nd.dset.shape, nd[0].shape)
        ic(nd.normalizers)
        for i, rec in enumerate(nd[:8]):
            ic(rec.shape, rec[0, :4])
    check_ptb_denoise_progress()

    def check_split_dataset():
        dest_tr, dset_vl, dset_ts = get_ptbxl_splits()
        ic(dest_tr, dset_vl, dset_ts)
        batch = dest_tr[0]
        sv, lbs = batch['sample_values'], batch['labels']
        ic(sv, lbs, sv.shape, lbs.shape)
    check_split_dataset()
