import glob
from pathlib import Path

import h5py
import wfdb.processing
from tqdm import tqdm, trange

from ecg_transformer.util import *
import ecg_transformer.util.ecg as ecg_util


def fix_g12ec_headers():
    """
    The 1st row of header files in G12EC datasets has an extra `.mat` in the record name
    """
    recs = ecg_util.get_rec_paths('G12EC')
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

        self.lbl_cols = ['dataset', 'patient_name', 'record_name', 'record_path']
        self.fqs = fqs

        self.logger = None

    def __call__(self, resample: Union[str, bool] = False):
        self.logger: logging.Logger = get_logger('ECG Record Export')
        dnms = self.dsets_exp["total"]
        self._log_info(f'Exporting ECG records on datasets {logi(dnms)}... ')
        # self.export_record_info()
        # for dnm in dnms[1:2]:
        for dnm in dnms:
            self.export_record_data(dnm, resample=resample)
        # self.export_record_data('CHAP_SHAO', resample)  # TODO: debugging

    def get_rec_nms(self, dnm):
        d_dset = self.d_dsets[dnm]
        return sorted(
            glob.iglob(os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'], d_dset['rec_fmt']), recursive=True)
        )

    def get_dset_record_info(self, dnm, return_df=True) -> Union[pd.DataFrame, List[List]]:
        d_dset = self.d_dsets[dnm]
        dir_nm = d_dset['dir_nm']
        path_ = os.path.join(PATH_BASE, DIR_DSET, dir_nm)

        def get_get_pat_num():
            def incart(fnm_):
                path_no_ext = fnm_[:fnm_.index('.')]
                rec = wfdb.rdrecord(path_no_ext, sampto=1)  # Don't need to see the signal
                return rec.comments[1]

            def ptb_xl(path_r_no_dset):
                if not hasattr(ptb_xl, 'df__'):
                    ptb_xl.df__ = pd.read_csv(
                        os.path.join(path_, 'ptbxl_database.csv'), usecols=['patient_id', 'filename_hr']
                    )
                return int(ptb_xl.df__[ptb_xl.df__.filename_hr == path_r_no_dset].iloc[0]['patient_id'])

            def ptb_diagnostic(rec_nm):
                if not hasattr(ptb_diagnostic, 'df__'):
                    fnm_ = config(f'{DIR_DSET}.{dnm}.path_label')
                    with open(os.path.join(PATH_BASE, DIR_DSET, dir_nm, fnm_)) as f:
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

        def get_relative_path_n_name(fnm_):
            """
            :return: 2-tuple of from (`datasets` to record file name], and file name without extension
            """
            path_r = fnm_.split('/')
            return '/'.join(path_r[path_r.index(dir_nm):-1]), Path(fnm_).stem

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
            fnm_ = rec_nms_[0]
            path_r, rec_nm = get_relative_path_n_name(fnm_)
            n_pat = h5py.File(fnm_, 'r')['tracings'].shape[0]
            rows_ = []
            for i in trange(n_pat, desc='CODE_TEST', unit='rec'):
                rows_.append([dnm, i, rec_nm, path_r])
            return rows_

        self._log_info(f'Getting record info for {logi(dnm)}... ')
        rec_nms = self.get_rec_nms(dnm)[:10]
        if dnm == 'CODE_TEST':
            rows = get_row_code_test(rec_nms)
        else:
            rows = []
            for fnm in tqdm(rec_nms, desc=dnm, unit='rec'):
                rows.append(get_row(fnm))
        return pd.DataFrame(rows, columns=self.lbl_cols) if return_df else rows

    def export_record_info(self):
        self._log_info(f'Exporting dataset record info... ')
        df = pd.DataFrame(
            sum([self.get_dset_record_info(dnm, return_df=False) for dnm in self.dsets_exp['total']], start=[]),
            columns=self.lbl_cols
        )
        df = df.apply(lambda x: x.astype('category'))

        fnm = os.path.join(self.path_exp, self.d_my['fnm_labels'])
        df.to_csv(fnm)
        self._log_info(f'ECG record info exported to {logi(fnm)}')

    def _log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def export_record_data(self, dnm, resample: Union[bool, str] = True):
        """
        :param dnm: Dataset name
        :param resample: If true, resample to export `fqs`
            If `single`, keep *only* the resampled copy
        """
        if self.logger is not None:
            self.logger.info(f'Exporting {logi(dnm)} data... ')
        assert dnm in self.dsets_exp['total']
        d_dset = self.d_dsets[dnm]

        rec_nms = self.get_rec_nms(dnm)
        # for rnm in rec_nms:
        #     ic(fnm2sigs(rnm, dnm).shape)
        sigs = np.stack(batched_conc_map(lambda fnms_, s_, e_: [ecg_util.fnm2sigs(nm_, dnm) for nm_ in fnms_[s_:e_]], rec_nms))
        fqs = d_dset['fqs']
        d_rec = dict(n=len(rec_nms), shape=sigs.shape, frequency=fqs)
        self._log_info(f'Loaded record data: {log_dict(d_rec)}')
        shape = sigs.shape
        assert len(shape) == 3 and shape[0] == len(rec_nms) and shape[1] == 12
        assert not np.isnan(sigs).any()

        _resample = resample and self.fqs != fqs
        sigs_ = []
        if _resample:
            def _resampler(sig: np.array) -> np.array:  # `resample_sig` seems to work with 1D signal only
                return wfdb.processing.resample_sig(sig, fqs, self.fqs)[0]

            def resampler(sigs__: np.ndarray) -> np.ndarray:
                return np.stack([_resampler(sig) for sig in sigs__])
            self._log_info(f'Resampling to {logi(self.fqs)}Hz... ')
            lst_sigs = []
            for s in tqdm(sigs, desc='Resampling', unit='rec'):
                lst_sigs.append(resampler(s))
            sigs_ = np.stack(lst_sigs)
            fqs = self.fqs
        dsets = dict(data=sigs_ if resample else sigs)
        if _resample and resample != 'single':
            dsets['ori'] = sigs
        attrs = dict(dnm=dnm, fqs=fqs, resampled=resample)

        fnm = os.path.join(self.path_exp, self.d_my['rec_fmt'] % dnm)
        self._log_info(f'Writing processed signals to {logi(fnm)}...')
        open(fnm, 'a').close()  # Create file in OS
        fl = h5py.File(fnm, 'w')
        fl.attrs['meta'] = json.dumps(attrs)
        # TODO: reduce memory usage
        self._log_info(f'Metadata attributes {logi(list(fl.attrs.keys()))} added')
        for nm, data in dsets.items():
            fl.create_dataset(nm, data=data)
        self._log_info(f'HDF5 dataset on {logi(dnm)} with splits {logi([nm for nm in fl])} written to file ')


if __name__ == '__main__':
    from icecream import ic

    np.random.seed(config('random-seed'))
    # fix_g12ec_headers()

    def export():
        de = RecDataExport(fqs=250)
        de(resample='single')
        # de(resample=True)
    # export()

    def sanity_check():
        """
        Check the MATLAB h5 output working properly
        """
        dnm = 'CHAP_SHAO'
        d_dset = config(f'datasets.my')
        path_exp = os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'])
        fnm = os.path.join(path_exp, d_dset['rec_fmt_denoised'] % dnm)
        ic(fnm)
        rec = h5py.File(fnm, 'r')
        ic(rec, list(rec.keys()), list(rec.attrs))

        data = rec['data']
        ic(type(data), data.shape, data[0, :3, :5])
        ic(rec.attrs['meta'])

        # Check which signal is denoised, those not-yet denoised are filled with 0
        idx_filled = np.array([not np.all(d == 0) for d in data])
        ic(idx_filled.shape, idx_filled[:10])
        ic(np.count_nonzero(idx_filled))
    # sanity_check()

    def check_matlab_out():
        """
        Check the MATLAB data processing output quality
        """
        from matplotlib.widgets import Button

        # dnm = 'CHAP_SHAO'
        dnm = 'PTB_XL'
        d_dset = config(f'datasets.my')
        path_exp = os.path.join(PATH_BASE, DIR_DSET, d_dset['dir_nm'])
        rec_ori = h5py.File(os.path.join(path_exp, d_dset['rec_fmt'] % dnm), 'r')
        rec_den = h5py.File(os.path.join(path_exp, d_dset['rec_fmt_denoised'] % dnm), 'r')
        data_den, data_ori = rec_den['data'], rec_ori['data']  # Share frequency
        ic(data_ori.shape)
        n_sig, n_ch, l_ch = data_ori.shape

        sig, truth_denoised = ecg_util.get_nlm_denoise_truth(verbose=False)[:2]
        ic(sig[:10], truth_denoised[:10], sig.shape)
        # ecg_util.plot_1d(
        #     [sig, truth_denoised],
        #     label=['Original, resampled', 'Denoised'],
        #     title=f'[{dnm}] output generated from dataset',
        #     # e=2**11
        # )

        # Pick a channel randomly
        def _step(s, c):
            plt.cla()
            ecg_util.plot_1d(
                [data_ori[s, c], data_den[s, c]],
                label=['Original, resampled', 'Denoised'],
                title=f'[{dnm}] Processed Signal random plot: signal {s+1} channel {c+1}',
                new_fig=False,
                show=False,
                # e=2**10
            )
            plt.draw()

        class PlotFrame:
            def __init__(self, i=0, n_s=n_sig, n_c=n_ch):
                self.n_s = n_s  # TODO: until full dataset ready
                self.n_s = 1024
                self.n_c = n_c
                n = self.n_s * self.n_c
                self.idxs = np.arange(n)
                np.random.shuffle(self.idxs)

                self.idx = i
                self.clp = clipper(0, n-1)
                self._set_curr_idx()

            def _set_curr_idx(self):
                self.i_s, self.i_c = self.idxs[self.idx] // self.n_c, self.idxs[self.idx] % self.n_c

            def next(self, event):
                prev_idx = self.idx
                self.idx = self.clp(self.idx+1)
                if prev_idx != self.idx:
                    self._set_curr_idx()
                    _step(self.i_s, self.i_c)

            def prev(self, event):
                prev_idx = self.idx
                self.idx = self.clp(self.idx-1)
                if prev_idx != self.idx:
                    self._set_curr_idx()
                    _step(self.i_s, self.i_c)

        plt.figure(figsize=(18, 6))

        init = 0
        pf = PlotFrame(i=init)
        ax = plt.gca()
        btn_next = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Next')
        btn_next.on_clicked(pf.next)
        btn_prev = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Previous')
        btn_prev.on_clicked(pf.prev)
        plt.sca(ax)

        # _step(pf.i_s, pf.i_c)
        _step(77, 0)
        plt.show()
    check_matlab_out()


