"""
Export PTB-XL dataset labels

All keys in the `scp_code` are considered ground truth binary labels
    Downside: the likelihood are ignored, effectively treating each key as 100% confidence
"""

from ast import literal_eval

from ecg_transformer.util import *


if __name__ == '__main__':
    from icecream import ic

    path = os.path.join(PATH_BASE, DIR_DSET, config('datasets.PTB_XL.dir_nm'))
    df = pd.read_csv(
        os.path.join(path, 'ptbxl_database.csv'), usecols=['ecg_id', 'patient_id', 'scp_codes'], index_col=0
    )
    # df = df[:128]  # TODO: debugging
    df.patient_id = df.patient_id.astype(int)
    df.scp_codes = df.scp_codes.apply(literal_eval)
    _d_codes = config('datasets.PTB_XL.code')
    d_codes, code2id = _d_codes['codes'], _d_codes['code2id']

    def map_row(row: pd.Series):
        codes = list(row.scp_codes.keys())
        assert all(c in d_codes for c in codes)
        # return dict(labels=sorted(code2id[c] for c in codes))
        return sorted(code2id[c] for c in codes)

    df['labels'] = df.apply(map_row, axis=1)
    ic(df)

    df.to_csv(os.path.join(config('path-export'), 'ptb-xl-labels.csv'))
