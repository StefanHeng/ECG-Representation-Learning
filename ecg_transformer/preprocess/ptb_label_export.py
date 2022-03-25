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
    df = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), usecols=['ecg_id', 'patient_id', 'scp_codes'])
    df = df[:128]  # TODO: debugging
    df.patient_id = df.patient_id.astype(int)
    df.scp_codes = df.scp_codes.apply(literal_eval)
    ic(df)
