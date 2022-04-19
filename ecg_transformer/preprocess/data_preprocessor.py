"""
Taken from [ECGDenoisingTool](https://github.com/zheng120/ECGDenoisingTool),
used in paper *Optimal Multi-Stage Arrhythmia Classification Approach*
"""

import sys
import math

import numpy as np
from scipy import signal, stats

from ecg_transformer.util.util import *


def force_odd(x):
    return 2 * math.floor(x / 2) + 1


class DataPreprocessor:
    CONFIG = config('pre_processing.zheng')

    def zheng(self, sig, fqs=500):
        """
        Zheng et al's denoising approach
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
        sig = self.butterworth_low_pass(sig)
        sig -= self.rloess(sig, n=fqs)
        return self.nlm(sig)

    @staticmethod
    def butterworth_low_pass(
            sig,
            fqs=500,
            w_pass=CONFIG['low_pass']['passband'],
            w_stop=CONFIG['low_pass']['stopband'],
            r_pass=CONFIG['low_pass']['passband_ripple'],
            r_stop=CONFIG['low_pass']['stopband_attenuation']
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
        from loess import loess_1d
        if isinstance(n, float):
            n = force_odd(int(sig.size * n) -1)
        return loess_1d.loess_1d(np.arange(sig.size).astype(np.float64), sig, degree=2, npoints=n)[1]

    @staticmethod
    def est_noise_std(arr):
        res = arr.copy()
        for i in range(1, arr.size-1):
            res[i] = (2*res[i] - res[i-1] - res[i+1]) / math.sqrt(6)
        return stats.median_abs_deviation(1.4826 * (res - np.median(res)))

    @staticmethod
    def nlm(
            sig,
            scale=CONFIG['nlm']['smooth_factor'],
            sch_wd=None,
            patch_wd=CONFIG['nlm']['window_size']
    ):
        """
        :param sig: 1D array to apply Nonlocal means denoising
        :param scale: Gaussian scale factor for smoothness control
        :param sch_wd: Max search distance
        :param patch_wd: Patch window size, half-width

        Modified from [Jianwei Zheng's implementation]
        (https://github.com/zheng120/ECGDenoisingTool/blob/master/NLMDenoising20191120.py)
        """
        if sch_wd is None:
            sch_wd = sig.size
        if isinstance(sch_wd, int):  # scalar has been entered; expand into patch sample index vector
            sch_wd = sch_wd - 1  # Python start index from 0
            p_vec = np.array(range(-sch_wd, sch_wd + 1))
        else:
            p_vec = sch_wd  # use the vector that has been input
        sig = np.array(sig)
        n = len(sig)

        denoised_sig = np.empty(len(sig))  # NaN * ones(size(signal));
        denoised_sig[:] = np.nan
        # to simplify, don't bother denoising edges
        i_start = patch_wd + 1
        i_end = n - patch_wd
        denoised_sig[i_start: i_end] = 0

        # initialize weight normalization
        z = np.zeros(len(sig))

        # convert scale to  'h', denominator, as in original Buades papers
        n_patch = 2 * patch_wd + 1
        scale *= DataPreprocessor.est_noise_std(sig)
        h = 2 * n_patch * scale ** 2

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
                # Note the - 1; we want to include the point ii - iPatchHW
                distance = sdx[ii + patch_wd] - sdx[ii - patch_wd - 1]

                w = math.exp(-distance / h)  # Eq 2 in Darbon
                t = ii + idx  # in the papers, this is not made explicit

                if 0 < t < n:
                    denoised_sig[ii] = denoised_sig[ii] + w * sig[t]
                    z[ii] = z[ii] + w

        # Apply normalization
        denoised_sig = denoised_sig / (z + sys.float_info.epsilon)
        denoised_sig[0: patch_wd + 1] = sig[0: patch_wd + 1]
        denoised_sig[- patch_wd:] = sig[- patch_wd:]
        return denoised_sig

    @staticmethod
    def normalize(sig, method='3std', mean=None, std=None):
        """
        :param sig: Signal to normalize
        :param method: Normalization approach
            If `0-mean`, normalize to mean of 0 and standard deviation of 1
            If `3std`, normalize data within 3 standard deviation to range of [-1, 1]
        :param mean: If not given, default to mean of entire signal
        :param std: If not given, default to std of entire signal
        """
        if mean is None:
            mean = sig.mean()
        if std is None:
            std = sig.std()

        if method == '0-mean':
            return (sig - mean) / std
        elif method == '3std':
            pass


if __name__ == '__main__':
    from icecream import ic

    # ic([force_odd(x) for x in range(10)])

    s, truth_denoised, truth_lowpass, truth_rloess, truth_localres, truth_after2nd = get_nlm_denoise_truth(
        verbose=False
    )
    # plot_1d([s, truth_denoised], label=['Original', 'Denoised, truth'], e=2 ** 10)
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
        ic(np.allclose(rloess, truth_rloess, atol=10))
    # check_loess()

    def check_nlm_1d():
        after_2nd = truth_lowpass - truth_rloess
        assert math.isclose(dp.est_noise_std(after_2nd), 7.4435, abs_tol=1e-3)  # Value from MATLAB output
        denoised = dp.nlm(after_2nd, scale=1.5, sch_wd=after_2nd.size, patch_wd=10)
        # Results in greater difference
        # denoised = dp.nlm(after_2nd, scale=7.4435, sch_wd=after_2nd.size, patch_wd=10)
        ic(denoised[:5], truth_denoised[:5])
        plot_1d([denoised, truth_denoised], label=['my', 'ori'], e=2**11)
        plot_1d(denoised-truth_denoised, label='difference', e=2**11)
        ic(np.allclose(denoised, truth_denoised, atol=10))
    # check_nlm_1d()

    def check_runtime():
        # sig = get_signal_eg('INCART')[:, 0]  # Doesn't terminate locally, on `INCART`
        # ic(sig.shape)
        ic(s.shape)
        ic()
        ic(dp.zheng(s))
        ic()
    check_runtime()
