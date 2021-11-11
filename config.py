import glob
from math import isnan

import wfdb
from icecream import ic

from util import *
from data_path import *


config = {
    DIR_DSET: dict(
        BIH_MVED=dict(
            nm='MIT-BIH Malignant Ventricular Ectopy Database',
            dir_nm='MIT-BIH-MVED'
        ),
        INCART=dict(
            dir_nm='St-Petersburg-INCART',
            nm='St Petersburg INCART 12-lead Arrhythmia Database',
            rec_fmt='*.dat'  # For glob search
        ),
        PTB_XL=dict(
            nm='PTB-XL, a large publicly available electrocardiography dataset',
            dir_nm='PTB-XL',
            rec_fmt='records500/**/*.dat',
        ),
        PTB_Diagnostic=dict(
            nm='PTB Diagnostic ECG Database',
            dir_nm='PTB-Diagnostic',
            rec_fmt='**/*.dat',
            path_label='RECORDS'
        ),
        CSPC=dict(
            nm='China Physiological Signal Challenge 2018',
            dir_nm='CSPC-2018',
            rec_fmt='*.mat',
        ),
        CSPC_CinC=dict(
            nm='China Physiological Signal Challenge 2018 - from CinC',
            dir_nm='CSPC-2018-CinC',
            rec_fmt='*.mat'
        ),
        CSPC_Extra_CinC=dict(
            nm='China Physiological Signal Challenge 2018, unused/extra - from CinC',
            dir_nm='CSPC-2018-Extra-CinC',
            rec_fmt='*.mat'
        ),
        G12EC=dict(
            nm='Georgia 12-lead ECG Challenge (G12EC) Database',
            dir_nm='Georgia-12-Lead',
            rec_fmt='*.mat'
        ),
        my=dict(
            nm='Stefan-12-Lead-Combined',
            dir_nm='Stef-Combined',
            rec_labels='records.csv'
        )
    )
}


df = get_my_rec_labels()
for dnm in df['dataset'].unique():
    df_ = df[df['dataset'] == dnm]
    d_dset = config[DIR_DSET][dnm]

    d_dset['n_rec'] = df_.shape[0]

    uniqs = df_['patient_name'].unique()
    d_dset['n_pat'] = 'Unknown' if len(uniqs) == 1 and isnan(uniqs[0]) else len(uniqs)

    path = f'{PATH_BASE}/{DIR_DSET}/{d_dset["dir_nm"]}'
    rec_path = next(glob.iglob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))
    d_dset['fqs'] = wfdb.rdrecord(rec_path[:rec_path.index('.')], sampto=1).fs


if OS == 'Windows':
    for k in keys(config):
        val = get(config, k)
        if type(val) is str:
            set_(config, k, val.replace('/', '\\'))


if __name__ == '__main__':
    import json

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{DIR_PROJ}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)
