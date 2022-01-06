from math import isnan

from util import *
from data_path import *


config = {
    'meta': dict(
        path_base=PATH_BASE,
        dir_proj=DIR_PROJ,
        dir_dset=DIR_DSET
    ),
    DIR_DSET: dict(
        BIH_MVED=dict(
            nm='MIT-BIH Malignant Ventricular Ectopy Database',
            dir_nm='MIT-BIH-MVED'
        ),
        INCART=dict(
            dir_nm='St-Petersburg-INCART',
            nm='St Petersburg INCART 12-lead Arrhythmia Database',
            rec_fmt='*.dat',  # For glob search
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
            rec_fmt='*.mat',
        ),
        CSPC_Extra_CinC=dict(
            nm='China Physiological Signal Challenge 2018, unused/extra - from CinC',
            dir_nm='CSPC-2018-Extra-CinC',
            rec_fmt='*.mat',
        ),
        G12EC=dict(
            nm='Georgia 12-lead ECG Challenge (G12EC) Database',
            dir_nm='Georgia-12-Lead',
            rec_fmt='*.mat',
        ),
        CHAP_SHAO=dict(
            nm='Chapman University, Shaoxing People’s Hospital 12-lead ECG Database',
            dir_nm='Chapman-Shaoxing',
            rec_fmt='ECGData/*.csv',
            # Taken from paper
            # *A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients*
            fqs=500
        ),
        CODE_TEST=dict(
            nm='CODE-test: An annotated 12-lead ECG dataset',
            dir_nm='CODE-test',
            rec_fmt='ecg_tracings.hdf5',
            fqs=400
        ),
        my=dict(
            nm='Stefan-12-Lead-Combined',
            dir_nm='Stef-Combined',
            fnm_labels='records.csv',
            rec_fmt='%s-combined.hdf5',
            rec_fmt_denoised='%s-denoised.hdf5',
        )
    ),
    'datasets_export': dict(
        total=['INCART', 'PTB_XL', 'PTB_Diagnostic', 'CSPC_CinC', 'CSPC_Extra_CinC', 'G12EC', 'CHAP_SHAO', 'CODE_TEST'],
        support_wfdb=['INCART', 'PTB_XL', 'PTB_Diagnostic', 'CSPC_CinC', 'CSPC_Extra_CinC', 'G12EC']
    ),
    'random_seed': 77,
    'pre_processing': dict(
        zheng=dict(
            low_pass=dict(
                passband=50,
                stopband=60,
                passband_ripple=1,
                stopband_attenuation=2.5
            ),
            nlm=dict(
                smooth_factor=1.5,
                window_size=10
            )
        )
    )
}


df = get_my_rec_labels()
sup = config['datasets_export']['support_wfdb']
for dnm in config['datasets_export']['total']:
    df_ = df[df['dataset'] == dnm]
    d_dset = config[DIR_DSET][dnm]

    d_dset['n_rec'] = df_.shape[0]

    uniqs = df_['patient_name'].unique()
    d_dset['n_pat'] = 'Unknown' if len(uniqs) == 1 and isnan(uniqs[0]) else len(uniqs)

    if dnm in sup:
        path = f'{PATH_BASE}/{DIR_DSET}/{d_dset["dir_nm"]}'
        rec_path = next(glob.iglob(f'{path}/{d_dset["rec_fmt"]}', recursive=True))
        d_dset['fqs'] = wfdb.rdrecord(rec_path[:rec_path.index('.')], sampto=1).fs


for k in keys(config):  # Accommodate other OS
    val = get(config, k)
    if k[k.rfind('.')+1:] == 'dir_nm':
        set_(config, k, os.path.join(*val.split('/')))


for dnm, d in config[DIR_DSET].items():
    if 'rec_fmt' in d:
        fmt = d['rec_fmt']
        d['rec_ext'] = fmt[fmt.index('.'):]


if __name__ == '__main__':
    import json

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{DIR_PROJ}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)
