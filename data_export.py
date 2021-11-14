import glob
from pathlib import Path

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
        dnms_selected = ['INCART', 'PTB_XL', 'PTB_Diagnostic', 'CSPC_CinC', 'CSPC_Extra_CinC', 'G12EC']
        dfs = []
        for dnm in dnms_selected:
            d_dset = d_dsets[dnm]
            dir_nm = d_dset['dir_nm']
            path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'

            def get_get_pat_num():
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

                def cspc_cinc():
                    """ From dataset description, we have one-to-one mapping of patient to record """
                    if not hasattr(cspc_cinc, 'n'):
                        cspc_cinc.n = 0
                    n = cspc_cinc.n
                    cspc_cinc.n += 1
                    return n

                def cspc_extra_cinc():
                    if not hasattr(cspc_extra_cinc, 'nan'):
                        cspc_cinc.nan = float('nan')
                    return cspc_cinc.nan

                d_f = dict(
                    INCART=lambda rec_: rec_.comments[1],
                    PTB_XL=ptb_xl,
                    PTB_Diagnostic=ptb_diagnostic,
                    CSPC_CinC=cspc_cinc,
                    CSPC_Extra_CinC=cspc_extra_cinc,
                    G12EC=cspc_extra_cinc  # Patient info not available
                )
                return d_f[dnm]

            get_pat_num = get_get_pat_num()

            def get_row(fnm_):
                path_r = fnm_.split('/')
                path_r = '/'.join(path_r[path_r.index(dir_nm):-1])  # From (`datasets` to record file name]
                rec_nm = Path(fnm_).stem
                path_no_ext = fnm_[:fnm_.index('.')]
                rec = wfdb.rdrecord(path_no_ext, sampto=1)  # Don't need to see the signal

                d_args = dict(
                    INCART=[rec],
                    PTB_XL=[f'{path_r}/{rec_nm}'[len(dnm)+1:]],
                    PTB_Diagnostic=[rec_nm],
                    CSPC_CinC=[],
                    CSPC_Extra_CinC=[],
                    G12EC=[]
                )
                pat_nm = get_pat_num(*d_args[dnm])

                return [dnm, pat_nm, rec_nm, path_r]

            recs = sorted(glob.iglob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))[:5]
            df_ = pd.concat([pd.DataFrame([get_row(fnm)], columns=cols) for fnm in recs], ignore_index=True)
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

