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
import matplotlib.pyplot as plt
from loess import loess_1d
from my_loess import loess
from agramfort_lowess import lowess
import statsmodels.api as sm


class DataPreprocessor:
    def zheng(self, sig):
        """
        ```matlab
            OrigECG  = DataFile(:,j);
            Fs=500;
            fp=50;fs=60;
            rp=1;rs=2.5;
            wp=fp/(Fs/2);ws=fs/(Fs/2);
            [n,wn]=buttord(wp,ws,rp,rs);
            [bz,az] = butter(n,wn);
            LPassDataFile=filtfilt(bz,az,OrigECG);

            t = 1:length(LPassDataFile);
            yy2 = smooth(t,LPassDataFile,0.1,'rloess');
            BWRemoveDataFile = (LPassDataFile-yy2);
            Dl1=BWRemoveDataFile;
            for k=2:length(Dl1)-1
                Dl1(k)=(2*Dl1(k)-Dl1(k-1)-Dl1(k+1))/sqrt(6);
            end
            NoisSTD = 1.4826*median(abs(Dl1-median(Dl1)));
            DenoisingData(:,j)= NLM_1dDarbon(BWRemoveDataFile,(1.5)*(NoisSTD),5000,10);
        ```
        """
        sig = self.butterworth_low_pass(sig)
        # sig -= self.rloess(sig)
        # return sig
        return self.rloess(sig)

    def butterworth_low_pass(self, sig):
        fqs = 500
        nyq = 0.5 * fqs
        w_pass = 50 / nyq
        w_stop = 60 / nyq  # Not sure why stopband corner frequency is 60Hz
        r_pass, r_stop = 1, 2.5  # What are passband ripple and stopband attenuation for?
        # return lfilter(*r, signal)
nlm
        ord_, wn = signal.buttord(w_pass, w_stop, r_pass, r_stop)
        b, a = signal.butter(ord_, wn, btype='low')
        # ic(ord_, wn)
        return signal.filtfilt(b, a, sig)

    def rloess(self, sig):
        # _, y, _ = loess_1d.loess_1d(np.arange(sig.size), sig, degree=2, frac=0.1)
        _, y, _ = loess_1d.loess_1d(np.arange(sig.size), sig, degree=2, npoints=601)
        # ic(xout, yout, wout)
        # _, y = loess(np.arange(sig.size), sig, alpha=0.1, robustify=True, poly_degree=2)
        # y = lowess(np.arange(sig.size), sig, f=0.1, iter=5)
        # y = sm.nonparametric.lowess(sig, np.arange(sig.size), frac=0.05, it=5, return_sorted=False)
        return y

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

    dnm = 'CHAP_SHAO'
    fnm = get_rec_paths(dnm)[77]
    df = pd.read_csv(fnm)
    ic(len(df))
    df_de = pd.read_csv(fnm.replace('ECGData', 'ECGDataDenoised'), header=None)
    ic(df_de.head(6))
    ic(df_de.iloc[:2**11, 0])
    ic(fnm)

    DBG_PATH = '/Users/stefanh/Documents/UMich/Research/ECG-Classify/datasets/Chapman-Shaoxing/my_denoise_debugging'
    fnm_lowpass = os.path.join(DBG_PATH, 'MUSE_20180111_163412_52000, lowpass.csv')
    truth_lowpass = pd.read_csv(fnm_lowpass, header=None).iloc[:, 0].to_numpy()
    fnm_rloess = os.path.join(DBG_PATH, 'MUSE_20180111_163412_52000, rloess.csv')
    truth_rloess = pd.read_csv(fnm_rloess, header=None).iloc[:, 0].to_numpy()

    # plot_single(df.iloc[:2**11]['I'], label='Lead I', title='Original')
    # plot_single(df_de.iloc[:2**11][0], label='Lead I', title='Denoised')

    s = df.iloc[:]['I']
    dp = DataPreprocessor()
    # lowpass = dp.zheng(s)
    # ic(lowpass[:5], truth_lowpass[:5])
    # np.testing.assert_almost_equal(lowpass, truth_lowpass, decimal=0)
    rloess = dp.zheng(s)
    ic(rloess[:5], truth_rloess[:5])
    plot_1d([s, rloess, truth_rloess], label=['raw', 'mine', 'ori'])
    np.testing.assert_almost_equal(rloess, truth_rloess, decimal=0)

