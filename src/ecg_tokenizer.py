from enum import Enum
from typing import Iterable

from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
# from sklearn.mixture import GaussianMixture

from util import *
from ecg_loader import EcgLoader


class EcgTokenizer:
    """
    Tokenize ECG signals into symbols, given normalized signals in range [0, 1]
    """
    D_PAD_F = dict(  # TODO: other padding schemes?
        zero=lambda sig, n_pad: np.pad(sig, ((0, 0), (0, n_pad)), 'constant')  # Defaults to 0, pads to the right only
    )

    def __init__(self, k: int = 2**3, pad='zero'):
        """
        :param k: Length of each segment
        :param pad: Signal padding scheme

        .. note:: Signals are padded at the end until sample length reaches a multiple of `k`
        """
        self.k = k
        self.pad_ = pad

    def pad(self, sig):
        """
        :param sig: 2D array of shape C x L, or 3D array of shape N x C x L

        The last dimension is padded
        """
        sp = sig.shape
        sp, l = sp[:-1], sp[-1]
        n_pad = self.k - l % self.k
        # ic(l, n_pad, self.k)

        # def zero():
        #     return np.pad(sig, ((0, 0), (0, n_pad)), 'constant')  # Defaults to 0, pads to the right only
        if n_pad == 0:
            return sig
        else:
            if self.pad_ in EcgTokenizer.D_PAD_F:
                sig = sig.reshape(-1, l)  # Enforce 2D matrix
                return EcgTokenizer.D_PAD_F[self.pad_](sig, n_pad).reshape(sp + (l+n_pad,))
            else:
                log(f'Invalid padding scheme {self.pad}', 'err')
                exit(1)

    def fit(self, sigs: np.ndarray, method='spectral'):
        """
        sigs: Array of shape N x C x L

        Symbols for each signal channel learned separately
            Clustering labels are assigned for each channel, for N x C labels in total, in the input order

        TODO: Learn symbol across all channels
        """
        if self.pad_:
            sigs = self.pad(sigs)
            # ic(sigs.shape, self.k, sigs.shape[-1] % self.k)
            assert sigs.shape[-1] % self.k == 0
        n_rec = sigs.shape[0]
        # ic(sigs.shape)
        segs = sigs.reshape(-1, self.k)
        means = segs.mean(axis=-1, keepdims=True)
        # ic(means.shape)
        segs -= means  # Set mean of each segment to 0
        # ic(segs.mean(axis=-1))
        # ic(segs.shape)
        log(f'Clustering {logs(n_rec, c="i")} signals, making {logs(segs.shape[0], c="i")} segments')
        if method == 'spectral':
            # Too slow to run
            c = SpectralClustering(n_clusters=2**4, random_state=config('random_seed'))  # Defaults to RBF affinity
            c = c.fit(segs)
            ic(c.labels_)
        elif method == 'hierarchical':
            c = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=0.05)
            ic()
            c = c.fit(segs)
            ic()
            lbs = c.labels_
            ic(np.unique(lbs))
        elif method == 'dbscan':
            c = DBSCAN(eps=0.01, min_samples=5)
            ic()
            c = c.fit(segs)
            ic()
            lbs = c.labels_
            ic(np.unique(lbs, return_counts=True))
        elif method == 'optics':
            c = OPTICS(max_eps=0.05, min_samples=5)
            ic()
            c = c.fit(segs)
            ic()
            lbs = c.labels_
            ic(np.unique(lbs))
        elif method == 'birch':
            c = Birch(threshold=0.05, n_clusters=None)
            ic()
            c = c.fit(segs)
            ic()
            lbs = c.labels_
            ic(np.unique(lbs))
        pass


if __name__ == '__main__':
    from icecream import ic

    el = EcgLoader('CHAP_SHAO')  # TODO: Generalize to multiple datasets
    et = EcgTokenizer(k=32)

    def sanity_check():
        s = el[0]
        ic(s.shape)
        s_p = et.pad(s)
        ic(s_p, s_p.shape)
    # sanity_check()

    et.fit(el[:128], method='dbscan')
