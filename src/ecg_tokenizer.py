# from enum import Enum

from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch

from util import *
from ecg_loader import EcgLoader


def cluster(data: np.ndarray, method='spectral', cls_kwargs=None):
    """
    :param data: Data points to cluster, in N x D
    :param method: Clustering method
    :param cls_kwargs: Arguments to the clustering method
    :return:
    """
    d_kwargs = dict(
        hierarchical=dict(n_clusters=None, linkage='average'),
        dbscan=dict(min_samples=5),
        optics=dict(min_samples=5),
        birch=dict(n_clusters=None)
    )
    kwargs = d_kwargs[method] | cls_kwargs
    # ic(cls_kwargs)

    def hierarchical():
        assert 'distance_threshold' in kwargs
        return AgglomerativeClustering(**kwargs)

    def dbscan():
        assert 'eps' in kwargs
        return DBSCAN(**kwargs)

    def optics():
        assert 'max_eps' in kwargs
        return OPTICS(**kwargs)

    def birch():
        assert 'threshold' in kwargs
        return Birch(**kwargs)
    d_f = dict(
        hierarchical=hierarchical,
        dbscan=dbscan,
        optics=optics,
        birch=birch
    )
    return d_f[method]().fit(data)


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

        if n_pad == 0:
            return sig
        else:
            if self.pad_ in EcgTokenizer.D_PAD_F:
                sig = sig.reshape(-1, l)  # Enforce 2D matrix
                return EcgTokenizer.D_PAD_F[self.pad_](sig, n_pad).reshape(sp + (l+n_pad,))
            else:
                log(f'Invalid padding scheme {self.pad}', 'err')
                exit(1)

    def fit(self, sigs: np.ndarray, method='spectral', cls_kwargs=None, plot_dist: Union[int, bool] = False):
        """
        :param sigs: Array of shape N x C x L
        :param method: Clustering method
        :param cls_kwargs: Arguments to the clustering method
        :param plot_dist: If True, the counts for each cluster is plotted
            If integer give, the first most common classes are plotted

        Symbols for each signal channel learned separately
            Clustering labels are assigned for each channel, for N x C labels in total, in the input order

        TODO: Learn symbol across all channels
        """
        if self.pad_:
            sigs = self.pad(sigs)
            assert sigs.shape[-1] % self.k == 0
        n_rec = sigs.shape[0]
        segs = sigs.reshape(-1, self.k)
        log(f'Clustering {logs(n_rec, c="i")} signals => {logs(segs.shape[0], c="i")} segments')
        means = segs.mean(axis=-1, keepdims=True)
        segs -= means  # Set mean of each segment to 0

        ic()
        cls = cluster(segs, method=method, cls_kwargs=cls_kwargs)
        ic()
        lbs, counts = np.unique(cls.labels_, return_counts=True)
        ic(lbs, counts, lbs.shape, counts.shape)

        counts = np.flip(np.sort(counts))
        ic(counts)
        rank = np.arange(counts.size)+1
        ic(rank)
        if plot_dist:
            n = None if plot_dist is True else plot_dist
            y = np.flip(np.sort(counts))[:n]
            plt.figure(figsize=(18, 6))
            # sns.barplot(x=rank, y=counts, palette='flare')
            plt.plot(rank[:n], y, marker='o', lw=0.5, ms=1, label='# sample')
            scale = 10
            (a_, b_), (x_, y_) = fit_power_law(rank, counts, return_fit=scale)
            a_, b_ = round(a_, 2), round(b_, 2)
            n_ = n*scale
            plt.plot(x_[:n_], y_[:n_], lw=0.4, ls='-', label=fr'Fitted power law: ${a_} x^{{{b_}}}$')
            # ic(ret)
            plt.xlabel('Cluster, ranked')
            plt.ylabel('Frequency')
            plt.title('Rank-frequency plot after clustering')
            plt.legend()
            plt.show()


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

    # et.fit(el[:16], method='dbscan', cls_kwargs=dict(eps=0.01, min_samples=3))
    # et.fit(el[:128], method='birch', cls_kwargs=dict(threshold=0.05))
    et.fit(
        el[:8],
        method='hierarchical', cls_kwargs=dict(distance_threshold=0.02),
        plot_dist=40
    )

