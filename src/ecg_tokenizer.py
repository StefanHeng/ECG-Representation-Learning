from enum import Enum
from typing import Iterable

from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from util import *
from ecg_loader import EcgLoader


class EcgTokenizer:
    """
    Tokenize ECG signals into symbols, given normalized signals in range [0, 1]
    """

    # E_PAD = Enum('Padding', 'Zero Reflect Wrap')
    # D_PAD = dict(  # TODO: other padding schemes?
    #     zero=E_PAD.Zero,
    #     reflect=E_PAD.Reflect,
    #     wrap=E_PAD.Wrap
    # )

    def __init__(self, k: int = 2**3, pad='zero'):
        """
        :param k: Length of each segment
        :param pad: Signal padding scheme

        .. note:: Signals are padded at the end until sample length reaches a multiple of `k`
        """
        self.k = k
        # self.pad_ = EcgTokenizer.D_PAD[pad]
        self.pad = pad

    def pad(self, sig):
        """
        :param sig: 2D array of shape C x L
        """
        # if not hasattr(self.pad, 'd_pad'):
        #     self.pad.d_pad = dict()
        n_pad = sig.shape[1] % self.k

        def zero():
            return np.pad(sig, ((0, 0), (0, n_pad)), 'constant')  # Defaults to 0, pads to the right only

        if not hasattr(self.pad, 'D_PAD'):
            self.pad.D_PAD = dict(
                zero=zero
            )

        if n_pad == 0:
            return sig
        else:
            # if self.pad_ == EcgTokenizer.E_PAD.Zero:
            #     return np.pad(sig, ((0, 0), (0, n_pad)), 'constant')  # Defaults to 0, pads to the right only
            if self.pad_ in self.pad.D_PAD:
                return self.pad.D_PAD[self.pad_]()
            else:
                log(f'Invalid padding scheme {self.pad}', 'err')
                exit(1)

    def fit(self, sigs: Iterable[np.ndarray], method='spectral'):
        """
        Symbols for each signal channel learned separately

        TODO: Learn symbol across all channels
        """
        pass


if __name__ == '__main__':
    from icecream import ic

    el = EcgLoader('CHAP_SHAO')  # TODO: Generalize to multiple datasets
    et = EcgTokenizer()

    s = el[0]
    ic(s.shape)
    s_p = et.pad(s)
    ic(s_p, s_p.shape)
    # ic(np.padding_func)

    et.fit(el[:24])
