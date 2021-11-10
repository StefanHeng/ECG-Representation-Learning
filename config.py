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


config = dict(
    datasets=dict(
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
        my=dict(
            nm='Stefan-12-Lead-Combined',
            dir_nm='Stef-Combined'
        )
    )
)


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
