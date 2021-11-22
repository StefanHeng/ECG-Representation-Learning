"""
Taken from [ECGDenoisingTool](https://github.com/zheng120/ECGDenoisingTool),
used in paper *Optimal Multi-Stage Arrhythmia Classification Approach*
"""

# ***************************************************************************
# Copyright 2017-2019, Jianwei Zheng, Chapman University,
# zheng120@mail.chapman.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Jianwei Zheng.

import math
import sys

import numpy as np
from scipy import signal
from loess import loess_1d

from util import *


def force_odd(x):
    return 2 * math.floor(x / 2) + 1


class DataPreprocessor:
    CONFIG = config('pre_processing.zhang.low_pass')

    def zheng(self, sig):
        sig = self.butterworth_low_pass(sig)
        sig -= self.rloess(sig)
        return sig

    """
    low_pass=dict(
        passband=50,
        stopband=60,
        passband_ripple=1,
        stopband_attenuation=2.5
    )
    """

    @staticmethod
    def butterworth_low_pass(
            sig,
            fqs=500,
            w_pass=CONFIG['passband'],
            w_stop=CONFIG['stopband'],
            r_pass=CONFIG['passband_ripple'],
            r_stop=CONFIG['stopband_attenuation']
    ):
        nyq = 0.5 * fqs
        ord_, wn = signal.buttord(w_pass / nyq, w_stop / nyq, r_pass, r_stop)
        return signal.filtfilt(*signal.butter(ord_, wn, btype='low'), sig)

    @staticmethod
    def rloess(sig, n):
        """
        :param sig: 1D array to apply robust LOESS
        :param n: Number of points for calculating smoothed value, if float, treated as a fraction

        .. note:: Assumes signal is uniformly distributed,
        hence force an odd number of points as in MATLAB implementation
        """
        # float64 to ensure float output, due to package implementation
        if isinstance(n, float):
            n = force_odd(int(sig.size * n) -1)
        return loess_1d.loess_1d(np.arange(sig.size).astype(np.float64), sig, degree=2, npoints=n)[1]

    @staticmethod
    def nlm_1d_darbon_denoising(sig, n_var, p, patch_hw):
        if isinstance(p, int):  # scalar has been entered; expand into patch sample index vector
            p = p - 1  # Python start index from 0
            p_vec = np.array(range(-p, p + 1))
        else:
            p_vec = p  # use the vector that has been input
        sig = np.array(sig)
        # debug = [];
        n = len(sig)

        denoised_sig = np.empty(len(sig))  # NaN * ones(size(signal));
        denoised_sig[:] = np.nan
        # to simplify, don't bother denoising edges
        i_start = patch_hw + 1
        i_end = n - patch_hw
        denoised_sig[i_start: i_end] = 0

        # debug.iStart = iStart;
        # debug.iEnd = iEnd;

        # initialize weight normalization
        z = np.zeros(len(sig))
        cnt = np.zeros(len(sig))

        # convert lambda value to  'h', denominator, as in original Buades papers
        n_patch = 2 * patch_hw + 1
        h = 2 * n_patch * n_var ** 2

        for idx in p_vec:  # loop over all possible differences: s - t
            # do summation over p - Eq.3 in Darbon
            k = np.array(range(n))
            kplus = k + idx
            igood = np.where((kplus >= 0) & (kplus < n))  # ignore OOB data; we could also handle it
            ssd = np.zeros(len(k))
            ssd[igood] = (sig[k[igood]] - sig[kplus[igood]]) ** 2
            sdx = np.cumsum(ssd)

            for ii in range(i_start, i_end):  # loop over all points 's'
                # Eq 4;this is in place of point - by - point MSE
                distance = sdx[ii + patch_hw] - sdx[ii - patch_hw - 1]
                # but note the - 1; we want to include the point ii - iPatchHW

                w = math.exp(-distance / h)  # Eq 2 in Darbon
                t = ii + idx  # in the papers, this is not made explicit

                if 0 < t < n:
                    denoised_sig[ii] = denoised_sig[ii] + w * sig[t]
                    z[ii] = z[ii] + w
                    # cnt[ii] = cnt[ii] + 1
                    # print('ii',ii)
                    # print('t',t)
                    # print('w',w)
                    # print('denoisedSig[ii]', denoisedSig[ii])
                    # print('Z[ii]',Z[ii])
        # loop over shifts

        # now apply normalization
        denoised_sig = denoised_sig / (z + sys.float_info.epsilon)
        denoised_sig[0: patch_hw + 1] = sig[0: patch_hw + 1]
        denoised_sig[- patch_hw:] = sig[- patch_hw:]
        # debug.Z = Z;

        return denoised_sig  # ,debug


if __name__ == '__main__':
    import os

    from util import *

    # ic([force_odd(x) for x in range(10)])

    dnm = 'CHAP_SHAO'
    fnm = get_rec_paths(dnm)[77]

    def get_sig_eg():
        fnm_stem = stem(fnm)
        DBG_PATH = os.path.join(PATH_BASE, DIR_DSET, config(f'{DIR_DSET}.{dnm}.dir_nm'), 'my_denoise_debugging')
        ic(fnm, fnm_stem)
        ic(DBG_PATH)

        df = pd.read_csv(fnm)
        df_de = pd.read_csv(fnm.replace('ECGData', 'ECGDataDenoised'), header=None)
        ic(len(df))
        ic(df_de.head(6))
        ic(df_de.iloc[:6, 0])

        fnm_lowpass = os.path.join(DBG_PATH, f'{fnm_stem}, lowpass.csv')
        fnm_rloess = os.path.join(DBG_PATH, f'{fnm_stem}, rloess.csv')

        return (
            df.iloc[:]['I'],
            df_de.iloc[:][0],
            pd.read_csv(fnm_lowpass, header=None).iloc[:, 0].to_numpy(),
            pd.read_csv(fnm_rloess, header=None).iloc[:, 0].to_numpy()
        )

    s, s_d, truth_lowpass, truth_rloess = get_sig_eg()
    # plot_1d([s, s_d], label=['Original', 'Denoised, truth'], e=2 ** 10)
    dp = DataPreprocessor()

    def check_lowpass():
        lowpass = dp.butterworth_low_pass(s)
        ic(lowpass[:5], truth_lowpass[:5])
        plot_1d([s, lowpass, truth_lowpass], label=['raw', 'my', 'ori'])
        np.testing.assert_almost_equal(lowpass, truth_lowpass, decimal=0)
    # check_lowpass()

    def check_loess():
        # rloess = dp.zheng(s)
        # rloess = dp.rloess(lowpass)
        rloess = dp.rloess(truth_lowpass, n=500)
        ic(rloess[:5], truth_rloess[:5])
        plot_1d([s, rloess, truth_rloess], label=['raw', 'my', 'ori'])
        # plot_1d([s, s - rloess, s - truth_rloess], label=['raw', 'my smoothed', 'ori smoothed'], save=False)
        np.allclose(rloess, truth_rloess, atol=10)
    check_loess()
