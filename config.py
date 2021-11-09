import wfdb
from icecream import ic


d_incart = dict(
    dir_nm='St-Petersburg-INCART',
    nm='St Petersburg INCART 12-lead Arrhythmia Database'
)
rnm = 'I01'
fnm = f'../datasets/{d_incart["dir_nm"]}/{rnm}'
d_incart['fqs'] = wfdb.rdrecord(fnm, sampto=1).fs


config = dict(
    datasets=dict(
        BIH_MVED=dict(
            dir_nm='MIT-BIH-MVED',
            nm='MIT-BIH Malignant Ventricular Ectopy Database'
        ),
        INCART=d_incart,
        PTB_XL=dict(
            dir_nm='PTB-XL/records500',
            nm='PTB-XL, a large publicly available electrocardiography dataset',
            fqs=500
        )
    )
)

if __name__ == '__main__':
    import json
    from data_path import *

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)
