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
        cols = ['dataset', 'patient_name', 'record_name', 'record_path', 'comments']
        df = pd.DataFrame(columns=cols)

        dnms = config('datasets')
        for dnm, d_dset in dnms.items():
            if dnm == 'INCART':  # TODO: refine later
                def incart(fnm_):
                    path_r = fnm_.split('/')
                    path_r = '/'.join(path_r[path_r.index(dir_nm):-1])  # From `datasets` to record file name, exclusive
                    rec_nm = Path(fnm_).stem
                    # ic(path_r, rec_nm)
                    rec = wfdb.rdrecord(fnm_[:fnm_.index('.')], sampto=1)  # Don't need to see the signal
                    # ic(rec.comments)
                    comments = f'{rec.comments[0]}, {rec.comments[2]}'
                    pat_nm = rec.comments[1]
                    return [dnm, pat_nm, rec_nm, path_r, comments]

                dir_nm = d_dset['dir_nm']
                self.path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
                recs = sorted(glob.iglob(f'{self.path}/*.dat', recursive=True))
                # ic(recs)
                df_ = pd.concat([pd.DataFrame([incart(fnm)], columns=cols) for fnm in recs], ignore_index=True)
                ic(df_)
        ic(df)


if __name__ == '__main__':
    de = DataExport()
