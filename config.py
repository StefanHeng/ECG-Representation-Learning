import glob

import wfdb
from icecream import ic

from util import *
from data_path import *


d_incart = dict(
    dir_nm='St-Petersburg-INCART',
    nm='St Petersburg INCART 12-lead Arrhythmia Database',
    rec_fmt='*.dat'  # For glob search
)
rnm = 'I01'
fnm = f'../datasets/{d_incart["dir_nm"]}/{rnm}'
d_incart['fqs'] = wfdb.rdrecord(fnm, sampto=1).fs


config = {
    DIR_DSET: dict(
        BIH_MVED=dict(
            nm='MIT-BIH Malignant Ventricular Ectopy Database',
            dir_nm='MIT-BIH-MVED'
        ),
        INCART=d_incart,
        PTB_XL=dict(
            nm='PTB-XL, a large publicly available electrocardiography dataset',
            dir_nm='PTB-XL',
            rec_fmt='records500/**/*.dat',
            fqs=500
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
        my=dict(
            nm='Stefan-12-Lead-Combined',
            dir_nm='Stef-Combined'
        )
    )
}

for dnm in ['INCART', 'PTB_XL', 'PTB_Diagnostic']:
    d_dset = config[DIR_DSET][dnm]
    dir_nm = d_dset['dir_nm']
    path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
    d_dset['n_rec'] = len(glob.glob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))


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
