import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from icecream import ic
import wfdb
import wfdb.processing

from util import *
from data_preprocessor import DataPreprocessor

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
        with open(r, 'r') as f:
            lns = f.readlines()
            lns[0] = remove_1st_occurrence(lns[0], '.mat')
        with open(r, 'w') as f:
            f.write(''.join(lns))


class RecDataExport:
    """
    Integrate & export collected 12-lead ECG datasets, including pre-processed ECG signals and labels
    """

    def __init__(self, fqs=250):
        """
        :param fqs: (Potentially re-sampling) frequency
        """
        self.d_dsets = config('datasets')
        self.dsets_exp = config('datasets_export')
        self.d_my = config('datasets.my')
        self.path_exp = os.path.join(PATH_BASE, DIR_DSET, self.d_my['dir_nm'])

        self.dp = DataPreprocessor()

        self.lbl_cols = ['dataset', 'patient_name', 'record_name', 'record_path']
        self.fqs = fqs

    def __call__(self):
        # self.export_labels()
        self.export_dset('G12EC')

    def rec_nms(self, dnm):
        d_dset = self.d_dsets[dnm]
        return sorted(
            glob.iglob(os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'], d_dset['rec_fmt']), recursive=True)
        )

    def get_label_df(self, dnm):
        d_dset = self.d_dsets[dnm]
        dir_nm = d_dset['dir_nm']
        # path = f'{PATH_BASE}/{DIR_DSET}/{dir_nm}'
        path = os.path.join(PATH_BASE, DIR_DSET, dir_nm)

        def get_get_pat_num():
            def incart(fnm_):
                path_no_ext = fnm_[:fnm_.index('.')]
                rec = wfdb.rdrecord(path_no_ext, sampto=1)  # Don't need to see the signal
                return rec.comments[1]

            def ptb_xl(path_r_no_dset):
                if not hasattr(ptb_xl, 'df__'):
                    ptb_xl.df__ = pd.read_csv(
                        os.path.join(path, 'ptbxl_database.csv'), usecols=['patient_id', 'filename_hr']
                    )
                return int(ptb_xl.df__[ptb_xl.df__.filename_hr == path_r_no_dset].iloc[0]['patient_id'])

            def ptb_diagnostic(rec_nm):
                if not hasattr(ptb_diagnostic, 'df__'):
                    fnm = config(f'{DIR_DSET}.{dnm}.path_label')
                    with open(os.path.join(PATH_BASE, DIR_DSET, dir_nm, fnm)) as f:
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

        def get_row_code_test(rec_nms_):
            assert len(rec_nms_) == 1  # Only 1 hdf5 file
            fnm = rec_nms_[0]
            path_r, rec_nm = get_relative_path_n_name(fnm)
            n_pat = h5py.File(fnm, 'r')['tracings'].shape[0]
            return [[dnm, i, rec_nm, path_r] for i in range(n_pat)]

        # recs = sorted(glob.iglob(os.path.join(path, d_dset['rec_fmt']), recursive=True))[:5]
        rec_nms = self.rec_nms(dnm)[:5]
        if dnm == 'CODE_TEST':
            rows = get_row_code_test(rec_nms)
        else:
            rows = [get_row(fnm) for fnm in rec_nms]
        df = pd.concat([pd.DataFrame([r], columns=self.lbl_cols) for r in rows], ignore_index=True)
        ic(df)
        # dfs.append(df_)
        return df

    def export_labels(self):
        df = pd.concat([self.get_label_df(dnm) for dnm in self.dsets_exp['total']], ignore_index=True)
        df = df.apply(lambda x: x.astype('category'))
        ic(df)

        fnm = os.path.join(self.path_exp, self.d_my['fnm_labels'])
        df.to_csv(fnm)
        print(f'Exported to {fnm}')

    def export_dset(self, dnm, resample=True, normalize=''):
        """
        Normalization by mean & std across the entire dataset
        :param dnm: Dataset name
        :param resample: If true, resample to export `fqs`
        :param normalize: Normalization approach
            If `0-mean`, normalize to mean of 0 and standard deviation of 1
            If `3std`, normalize data within 3 standard deviation to range of [-1, 1]
        :return:
        """
        assert dnm in self.dsets_exp['total']
        d_dset = self.d_dsets[dnm]

        rec_nms = self.rec_nms(dnm)[:5]
        # sigs = np.stack([wfdb.rdrecord(nm.removesuffix(d_dset['rec_suffix'])).p_signal.T for nm in rec_nms])
        print(f'{now()}| Reading in {len(rec_nms)} records... ')
        sigs = np.stack(  # Concurrency
            list(conc_map(lambda nm: wfdb.rdrecord(nm.removesuffix(d_dset['rec_ext'])).p_signal.T, rec_nms))
        )
        fqs = d_dset['fqs']
        print(f'{now()}| ... of shape {sigs.shape} and frequency {fqs}Hz')
        shape = sigs.shape
        # ic(shape)
        # ic(fqs)
        assert len(shape) == 3 and shape[0] == len(rec_nms) and shape[1] == 12
        assert not np.isnan(sigs).any()

        if resample and self.fqs != fqs:
            print(f'{now()}| Resampling signals to {self.fqs}Hz... ')
            sigs = np.stack(list(conc_map(  # `resample_sig` seems to work with 1D signal only
                lambda sig: np.stack([wfdb.processing.resample_sig(s, fqs, self.fqs)[0] for s in sig]), sigs)
            ))
            fqs = self.fqs
        dsets = dict(
            ori=sigs
        )

        print(f'{now()}| Denoising signals... ')
        # sigs_denoised = np.stack(
        #     list(conc_map(lambda sig: np.stack([self.dp.zheng(s) for s in sig]), sigs))
        # )
        sigs_denoised = np.stack(
            [np.stack([self.dp.zheng(s) for s in sig]) for sig in sigs]
        )
        dsets['denoised'] = sigs_denoised

        mean, std = sigs.mean(), sigs.std()
        sigs_normalized = (sigs_denoised - mean) / std
        dsets['normalized'] = sigs_normalized

        fnm = os.path.join(self.path_exp, self.d_my['rec_fmt'] % dnm)
        print(f'{now()}| Writing processed signals to [{stem(fnm, ext=True)}]...')
        open(fnm, 'a').close()  # Create file in OS
        fl = h5py.File(fnm, 'w')
        fl.attrs['meta'] = json.dumps(dict(
            dnm=dnm,
            mean=mean,
            std=std,
            fqs=fqs
        ))
        print(f'{now()}| Metadata attributes created: {list(fl.attrs.keys())}')
        for dnm, data in dsets:
            fl.create_dataset(dnm, data=data)
        # fl.create_dataset('data-processed', data=[1, 2, 3])
        # for idx_brg, test_nm in enumerate(self.FLDR_NMS):
        #     group = fl.create_group(test_nm)
        #     for acc in ['hori', 'vert']:
        #         arr_extr = np.stack([
        #             self.get_feature_series(idx_brg, func, acc) for k, func in extr.D_PROP_FUNC.items()
        #         ])
        #         group.create_dataset(acc, data=arr_extr)
        print(f'{now()}| Dataset processing complete: {[nm for nm in fl]}')


if __name__ == '__main__':
    de = RecDataExport()
    de()

    def sanity_check():
        """
        Check the data processing result
        """

    # fix_g12ec_headers()

