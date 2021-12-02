import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from icecream import ic
import wfdb

from util import *

# pd.set_option('display.width', 200)
# pd.set_option('display.max_columns', 7)


def fix_g12ec_headers():
    """
    The 1st row of header files in G12EC datasets has an extra `.mat` in the record name
    """
    recs = get_rec_paths('G12EC')
    ic(recs[:5], len(recs))
    for r in recs:
        r = r.removesuffix('.mat') + '.hea'
        # r += '.hea'
        with open(r, 'r') as f:
            lns = f.readlines()
            # ic(lns)
            lns[0] = remove_1st_occurrence(lns[0], '.mat')
        with open(r, 'w') as f:
            f.write(''.join(lns))


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
        exp = config('datasets_export')
        supports_wfdb = exp['support_wfdb']
        dfs = []
        for dnm in exp['total']:
            d_dset = d_dsets[dnm]
            dir_nm = d_dset['dir_nm']
            path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'

            def get_get_pat_num():
                def incart(fnm_):
                    path_no_ext = fnm_[:fnm_.index('.')]
                    rec = wfdb.rdrecord(path_no_ext, sampto=1)  # Don't need to see the signal
                    return rec.comments[1]

                def ptb_xl(path_r_no_dset):
                    if not hasattr(ptb_xl, 'df__'):
                        ptb_xl.df__ = pd.read_csv(f'{path}/ptbxl_database.csv', usecols=['patient_id', 'filename_hr'])
                    return int(ptb_xl.df__[ptb_xl.df__.filename_hr == path_r_no_dset].iloc[0]['patient_id'])

                def ptb_diagnostic(rec_nm):
                    if not hasattr(ptb_diagnostic, 'df__'):
                        fnm = config(f'{DIR_DSET}.{dnm}.path_label')
                        with open(f'{PATH_BASE}/{DIR_DSET}/{dir_nm}/{fnm}') as f:
                            ptb_diagnostic.df__ = pd.DataFrame(
                                [ln.split('/') for ln in map(str.strip, f.readlines())],
                                columns=['patient_nm', 'rec_nm']
                            )
                    return ptb_diagnostic.df__[ptb_diagnostic.df__.rec_nm == rec_nm].iloc[0]['patient_nm']

                def one2one():
                    """ From dataset description, we have one-to-one mapping of patient to record """
                    if not hasattr(one2one, 'n'):
                        one2one.n = 0
                    n = one2one.n
                    one2one.n += 1
                    return n

                def na():
                    if not hasattr(na, 'nan'):
                        na.nan = float('nan')
                    return na.nan

                d_f = dict(
                    INCART=incart,
                    PTB_XL=ptb_xl,
                    PTB_Diagnostic=ptb_diagnostic,
                    CSPC_CinC=one2one,
                    CSPC_Extra_CinC=na,  # Unknown, suspect multiple records for a single patient
                    G12EC=na,  # Patient info not available & multiple records for a single patient
                    CHAP_SHAO=one2one,
                    CODE_TEST=one2one
                )
                return d_f[dnm]

            get_pat_num = get_get_pat_num()

            def get_relative_path_n_name(fnm):
                """
                :return: 2-tuple of from (`datasets` to record file name], and file name without extension
                """
                path_r = fnm.split('/')
                return '/'.join(path_r[path_r.index(dir_nm):-1]), Path(fnm).stem

            def get_row(fnm_):
                path_r, rec_nm = get_relative_path_n_name(fnm_)
                d_args = dict(
                    INCART=[fnm_],
                    PTB_XL=[f'{path_r}/{rec_nm}'[len(dnm)+1:]],
                    PTB_Diagnostic=[rec_nm],
                    CSPC_CinC=[],
                    CSPC_Extra_CinC=[],
                    G12EC=[],
                    CHAP_SHAO=[]
                )
                pat_nm = get_pat_num(*d_args[dnm])
                return [dnm, pat_nm, rec_nm, path_r]

            def get_row_code_test(recs_):
                fnm = recs_[0]  # Only 1 hdf5 file
                path_r, rec_nm = get_relative_path_n_name(fnm)
                n_pat = h5py.File(fnm, 'r')['tracings'].shape[0]
                return [[dnm, i, rec_nm, path_r] for i in range(n_pat)]

            recs = sorted(glob.iglob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))[:3]
            if dnm == 'CODE_TEST':
                rows = get_row_code_test(recs)
            else:
                rows = [get_row(fnm) for fnm in recs]
            df_ = pd.concat([pd.DataFrame([r], columns=cols) for r in rows], ignore_index=True)
            ic(df_)
            dfs.append(df_)
        df = pd.concat(dfs, ignore_index=True)
        df = df.apply(lambda x: x.astype('category'))
        ic(df)

        d_my = config('datasets.my')
        fnm = f'{PATH_BASE}/{DIR_DSET}/{d_my["dir_nm"]}/{d_my["rec_labels"]}'
        df.to_csv(fnm)
        print(f'Exported to {fnm}')


if __name__ == '__main__':
    de = DataExport()

    # fix_g12ec_headers()

