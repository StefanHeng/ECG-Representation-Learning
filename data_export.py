import glob
from pathlib import Path

import pandas as pd
from icecream import ic
import wfdb

from util import *

# pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 7)


class DataExport:
    """
    Integrate & export collected 12-lead ECG datasets
    """

    def __init__(self, fqs=250):
        """
        :param fqs: (Potentially re-sampling) frequency
        """
        cols = ['dataset', 'patient_name', 'record_name', 'record_path']

        d_dsets = config('datasets')
        dnms_selected = ['INCART', 'PTB_XL']
        dfs = []
        for dnm in dnms_selected:
            d_dset = d_dsets[dnm]
            dir_nm = d_dset['dir_nm']
            path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'

            def get_get_pat_num():
                def ptb_xl(rec_):
                    if not hasattr(ptb_xl, 'csv'):
                        ptb_xl.csv = pd.read_csv(f'{path}/ptbxl_database.csv', usecols=['patient_id'])
                        ptb_xl.pat_nms = ptb_xl.csv.to_numpy().astype(int).squeeze()
                        ptb_xl.count = 0

                    pat = ptb_xl.pat_nms[ptb_xl.count]
                    ptb_xl.count += 1
                    return pat

                d_f = dict(
                    INCART=lambda rec_: rec_.comments[1],
                    PTB_XL=ptb_xl
                )
                return d_f[dnm]

            get_pat_num = get_get_pat_num()

            def get_get_row(fnm_):
                path_r = fnm_.split('/')
                path_r = '/'.join(path_r[path_r.index(dir_nm):-1])  # From `datasets` to record file name, exclusive
                rec_nm = Path(fnm_).stem
                path_no_ext = fnm_[:fnm_.index('.')]
                rec = wfdb.rdrecord(path_no_ext, sampto=1)  # Don't need to see the signal
                pat_nm = get_pat_num(rec)

                return [dnm, pat_nm, rec_nm, path_r]

            recs = sorted(glob.iglob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))
            df_ = pd.concat([pd.DataFrame([get_get_row(fnm)], columns=cols) for fnm in recs], ignore_index=True)
            ic(df_)
            dfs.append(df_)
        df = pd.concat(dfs, ignore_index=True)

        ic(df)
        dir_my = config('datasets.my.dir_nm')
        ic(dir_my)
        df.to_csv(f'{PATH_BASE}/{DIR_DSET}/{dir_my}, records.csv', sep='\t')


if __name__ == '__main__':
    de = DataExport()
