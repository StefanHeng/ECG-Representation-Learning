import glob
import math

import wfdb

from ecg_transformer.util import *
import ecg_transformer.util.ecg as ecg_util
from ecg_transformer.preprocess import get_ptbxl_splits


config_dict = {
    'meta': dict(
        path_base=PATH_BASE,
        dir_proj=DIR_PROJ,
        dir_dset=DIR_DSET
    ),
    'datasets': {
        'BIH-MVED': dict(
            nm='MIT-BIH Malignant Ventricular Ectopy Database',
            dir_nm='MIT-BIH-MVED'
        ),
        'INCART': dict(
            dir_nm='St-Petersburg-INCART',
            nm='St Petersburg INCART 12-lead Arrhythmia Database',
            rec_fmt='*.dat',  # For glob search
        ),
        'PTB-XL': dict(
            nm='PTB-XL, a large publicly available electrocardiography dataset',
            dir_nm='PTB-XL',
            rec_fmt='records500/**/*.dat',  # the 500Hz signals
        ),
        'PTB-Diagnostic': dict(
            nm='PTB Diagnostic ECG Database',
            dir_nm='PTB-Diagnostic',
            rec_fmt='**/*.dat',
            path_label='RECORDS'
        ),
        'CSPC': dict(
            nm='China Physiological Signal Challenge 2018',
            dir_nm='CSPC-2018',
            rec_fmt='*.mat',
        ),
        'CSPC-CinC': dict(
            nm='China Physiological Signal Challenge 2018 - from CinC',
            dir_nm='CSPC-2018-CinC',
            rec_fmt='*.mat',
        ),
        'CSPC-Extra-CinC': dict(
            nm='China Physiological Signal Challenge 2018, unused/extra - from CinC',
            dir_nm='CSPC-2018-Extra-CinC',
            rec_fmt='*.mat',
        ),
        'G12EC': dict(
            nm='Georgia 12-lead ECG Challenge (G12EC) Database',
            dir_nm='Georgia-12-Lead',
            rec_fmt='*.mat',
        ),
        'CHAP-SHAO': dict(
            nm='Chapman University, Shaoxing People’s Hospital 12-lead ECG Database',
            dir_nm='Chapman-Shaoxing',
            rec_fmt='ECGData/*.csv',
            # Taken from paper
            # *A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients*
            fqs=500
        ),
        'CODE-TEST': dict(
            nm='CODE-test: An annotated 12-lead ECG dataset',
            dir_nm='CODE-test',
            rec_fmt='ecg_tracings.hdf5',
            fqs=400
        ),
        'my': dict(
            nm='Stefan-12-Lead-Combined',
            dir_nm='Stef-Combined',
            fnm_labels='records.csv',
            rec_fmt='%s-combined.hdf5',
            rec_fmt_denoised='%s-denoised.hdf5',
        )
    },
    'datasets-export': dict(
        total=['INCART', 'PTB-XL', 'PTB-Diagnostic', 'CSPC-CinC', 'CSPC-Extra-CinC', 'G12EC', 'CHAP-SHAO', 'CODE-TEST'],
        support_wfdb=['INCART', 'PTB-XL', 'PTB-Diagnostic', 'CSPC-CinC', 'CSPC-Extra-CinC', 'G12EC']
    ),
    'random-seed': 77,
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


def extract_ptb_codes():
    # Set supervised downstream task, PTB-XL, label information
    def is_nan(x) -> bool:
        return not isinstance(x, str) and math.isnan(x)

    def map_row(row: pd.Series) -> Dict:
        return dict(
            aspects=[k for k, v in row.iteritems() if k in ('diagnostic', 'form', 'rhythm') and v == 1],
            diagnostic_class=row.diagnostic_class if not is_nan(row.diagnostic_class) else None,
            diagnostic_subclass=row.diagnostic_subclass if not is_nan(row.diagnostic_subclass) else None
        )
    dnm = 'PTB-XL'
    path_ptb = os.path.join(PATH_BASE, DIR_DSET, get(config_dict, f'{DIR_DSET}.{dnm}.dir_nm'))
    df_ptb = pd.read_csv(
        os.path.join(path_ptb, 'scp_statements.csv'),
        usecols=['Unnamed: 0', 'diagnostic', 'form', 'rhythm', 'diagnostic_class', 'diagnostic_subclass'],
        index_col=0
    )
    codes = {code: map_row(row) for code, row in df_ptb.iterrows()}
    id2code = list(df_ptb.index)  # Stick to the same ordering
    assert len(id2code) == 71
    code2id = {c: i for i, c in enumerate(id2code)}
    set_(config_dict, f'datasets.{dnm}.code', dict(codes=codes, code2id=code2id, id2code=id2code))


def extract_datasets_meta():
    # Extract more metadata per dataset
    # 1) this script without this function
    # 2) run `ecg_transformer.preprocess.data_export.py`
    # 3) run the script with function
    df_label = ecg_util.get_my_rec_labels()
    sup = config_dict['datasets-export']['support_wfdb']
    for dnm in config_dict['datasets-export']['total']:
        df_ = df_label[df_label['dataset'] == dnm]
        d_dset = config_dict[DIR_DSET][dnm]

        d_dset['n_rec'] = df_.shape[0]

        uniqs = df_['patient_name'].unique()
        d_dset['n_pat'] = 'Unknown' if len(uniqs) == 1 and math.isnan(uniqs[0]) else len(uniqs)

        if dnm in sup:
            path_ = f'{PATH_BASE}/{DIR_DSET}/{d_dset["dir_nm"]}'
            rec_path = next(glob.iglob(f'{path_}/{d_dset["rec_fmt"]}', recursive=True))
            d_dset['fqs'] = wfdb.rdrecord(rec_path[:rec_path.index('.')], sampto=1).fs


def set_ptbxl_train_stats():
    """
    Get per-channel mean and standard deviation based on the training set, see `ecg_transformer.preprocess.transform.py`
    """
    n_sample = None

    def _get_single(type: str) -> Dict:
        dset = get_ptbxl_splits(n_sample=n_sample, dataset_args=dict(normalize=('std', 1), type=type))[0]
        std1_normalizer = dset.transform.transforms[0].normalizers[0]
        mean, std = std1_normalizer.norm_meta
        mean, std = mean.flatten().tolist(), std.flatten().tolist()
        return dict(mean=mean, std=std)
    # d_dset: Dict[str, Any] = config_dict['datasets']['PTB-XL']
    # d_dset['train-stats'] = dict(
    set_(config_dict, 'datasets.PTB-XL.train-stats', {k: _get_single(k) for k in ['original', 'denoised']})


def set_paths():
    # Accommodate other OS
    for key in keys(config_dict):
        val = get(config_dict, key)
        if key[key.rfind('.')+1:] == 'dir_nm':
            set_(config_dict, key, os.path.join(*val.split('/')))


def wrap_config():
    for dnm, d in config_dict[DIR_DSET].items():
        if 'rec_fmt' in d:
            fmt = d['rec_fmt']
            d['rec_ext'] = fmt[fmt.index('.'):]


extract_ptb_codes()
extract_datasets_meta()
set_ptbxl_train_stats()
set_paths()
wrap_config()


if __name__ == '__main__':
    import json

    from icecream import ic

    fl_nm = 'config.json'
    ic(config_dict)
    with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', fl_nm), 'w') as f:
        json.dump(config_dict, f, indent=4)
