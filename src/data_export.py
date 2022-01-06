import glob
from pathlib import Path
from typing import Union

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
    Integrate & export the 12-lead ECG datasets collected, in standardized format `hdf5`

    See the MATLAB versions for exporting denoised signals
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

    def __call__(self, resample: Union[str, bool] = False):
        # self.export_labels()
        # for dnm in self.dsets_exp['total']:
        #     self.export_dset(dnm)
        # TODO: Test run
        self.export_dset('CHAP_SHAO', resample)

    def rec_nms(self, dnm):
        d_dset = self.d_dsets[dnm]
        return sorted(
            glob.iglob(os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'], d_dset['rec_fmt']), recursive=True)
        )

    def get_label_df(self, dnm):
        d_dset = self.d_dsets[dnm]
        dir_nm = d_dset['dir_nm']
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

        rec_nms = self.rec_nms(dnm)[:5]
        if dnm == 'CODE_TEST':
            rows = get_row_code_test(rec_nms)
        else:
            rows = [get_row(fnm) for fnm in rec_nms]
        df = pd.concat([pd.DataFrame([r], columns=self.lbl_cols) for r in rows], ignore_index=True)
        ic(df)
        return df

    def export_labels(self):
        df = pd.concat([self.get_label_df(dnm) for dnm in self.dsets_exp['total']], ignore_index=True)
        df = df.apply(lambda x: x.astype('category'))
        ic(df)

        fnm = os.path.join(self.path_exp, self.d_my['fnm_labels'])
        df.to_csv(fnm)
        print(f'Exported to {fnm}')

    def export_dset(self, dnm, resample):
        """
        :param dnm: Dataset name
        :param resample: If true, resample to export `fqs`
            If `single`, keep *only* the resampled copy
        """

        assert dnm in self.dsets_exp['total']
        d_dset = self.d_dsets[dnm]

        rec_nms = self.rec_nms(dnm)
        # sigs = np.stack([wfdb.rdrecord(nm.removesuffix(d_dset['rec_suffix'])).p_signal.T for nm in rec_nms])
        print(f'{now()}| Reading in {len(rec_nms)} records of [{dnm}]... ')
        sigs = np.stack(list(conc_map(lambda nm: fnm2sigs(nm, dnm), rec_nms)))  # Concurrency
        fqs = d_dset['fqs']
        print(f'{now()}|     ... of shape {sigs.shape} and frequency {fqs}Hz')
        shape = sigs.shape
        assert len(shape) == 3 and shape[0] == len(rec_nms) and shape[1] == 12
        assert not np.isnan(sigs).any()

        resample = resample and self.fqs != fqs
        if resample:
            def resampler(s):
                return wfdb.processing.resample_sig(s, fqs, self.fqs)[0]
            ic()
            print(f'{now()}| Resampling signals to {self.fqs}Hz... ')
            sigs_ = np.stack(list(conc_map(  # `resample_sig` seems to work with 1D signal only
                lambda sig: np.stack([resampler(s) for s in sig]), sigs)
            ))
            fqs = self.fqs
            ic()
        dsets = dict(data=sigs_ if resample else sigs)
        if not resample == 'single':
            dsets['ori'] = sigs
        attrs = dict(dnm=dnm, fqs=fqs, resampled=resample)

        fnm = os.path.join(self.path_exp, self.d_my['rec_fmt'] % dnm)
        print(f'{now()}| Writing processed signals to [{stem(fnm, ext=True)}]...')
        open(fnm, 'a').close()  # Create file in OS
        fl = h5py.File(fnm, 'w')
        fl.attrs['meta'] = json.dumps(attrs)
        print(f'{now()}| Metadata attributes created: {list(fl.attrs.keys())}')
        for nm, data in dsets.items():
            fl.create_dataset(nm, data=data)
        print(f'{now()}| Dataset processing complete: {[nm for nm in fl]}')


if __name__ == '__main__':
    # fix_g12ec_headers()

    def export():
        de = RecDataExport()
        de(resample='single')
    # export()

    def sanity_check():
        """
        Check the data processing result from MATLAB
        """
        dnm = 'CHAP_SHAO'
        d_dset = config(f'datasets.my')
        # ic(d_dset['rec_fmt'] % dnm)
        path_exp = os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'])
        fnm = os.path.join(path_exp, d_dset['rec_fmt_denoised'] % dnm)
        ic(fnm)
        rec = h5py.File(fnm, 'r')
        ic(rec, list(rec.keys()), list(rec.attrs))

        data = rec['data']
        ic(type(data), data.shape, data[0, :3, :5])
        ic(np.where(data != 0))
        ic(rec.attrs['meta'])

        # s, truth_denoised = get_nlm_denoise_truth(verbose=False)[:2]
        # ic(truth_denoised[:10])
    sanity_check()
