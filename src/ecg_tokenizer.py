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

        self.centers = None  # of dim (N_cls, k); centers[i] stores the cluster mean for id `i`

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

    def fit(
            self, sigs: np.ndarray, method='spectral', cls_kwargs=None,
            plot_dist: Union[bool, int] = False,
            plot_segments: Union[bool, tuple[int, int]] = False):
        """
        :param sigs: Array of shape N x C x L
        :param method: Clustering method
        :param cls_kwargs: Arguments to the clustering method
        :param plot_dist: If True, the counts for each cluster is plotted
            If integer give, the first most common classes are plotted
        :param plot_segments: If 2-tuple given, plots the segment centroid for each cluster in grid

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
        lbs = cls.labels_  # Treated as the token id
        ic()
        ids_vocab, counts = np.unique(cls.labels_, return_counts=True)
        idxs_sort = np.argsort(-counts)
        ic(ids_vocab.shape, ids_vocab, counts)

        if plot_dist:
            n = None if plot_dist is True else plot_dist
            # y = np.flip(np.sort(counts))[:n]
            y = counts[idxs_sort][:n]
            rank = (np.arange(counts.size) + 1)[:n]

            plt.figure(figsize=(18, 6))
            plt.plot(rank, y, marker='o', lw=0.5, ms=1, label='# sample')

            scale = 10
            (a_, b_), (x_, y_) = fit_power_law(rank, y, return_fit=scale)
            a_, b_ = round(a_, 2), round(b_, 2)
            n_ = n*scale
            plt.plot(x_[:n_], y_[:n_], lw=0.4, ls='-', label=fr'Fitted power law: ${a_} x^{{{b_}}}$')
            log(f'R-squared for fitted curve: {logs(r2(y_, a_ * np.power(x_, b_)), c="i")}')

            plt.xlabel('Cluster, ranked')
            plt.ylabel('Frequency')
            plt.title('Rank-frequency plot after clustering')
            plt.legend()
            plt.show()

        # for lb in ids_vocab:
        #     arr = segs[lbs == lb].sum(axis=0)
        #     ic(arr, arr.shape)
        self.centers = np.stack([
            segs[lbs == lb].sum(axis=0) for lb in ids_vocab
        ])
        ic(self.centers, self.centers.shape)
        if plot_segments:
            n_col, n_row = plot_segments
            i_batch = 0
            n_batch = n_row * n_col
            offset = i_batch * n_batch
            idxs_ord = np.arange(n_batch) + offset  # Ordering for display
            idxs_sz = idxs_sort[idxs_ord]  # Internal ordering based on counts
            ic(offset, idxs_ord, counts[idxs_sz])
            mi, ma = self.centers[idxs_sz].min(), self.centers[idxs_sz].max()
            ylim = max(abs(mi), abs(ma)) * 1.25
            ylim = [-ylim, ylim]

            # lw doesn't seem to affect anything
            # sns.set_context(rc={'grid.linewidth': 0.5})
            with sns.axes_style('whitegrid', {'grid.linestyle': ':', 'grid.linewidth': 5}):
                fig = plt.figure(figsize=(n_col * 3, n_row * 2), constrained_layout=False)
                # gs = fig.add_gridspec(n_col, n_row)
                # ax = plt.gca()
                margin_h, plot_sep = 0.125/n_col, 0.125/n_col
                plt.subplots_adjust(
                    left=margin_h, right=1-margin_h/2,
                    top=0.925, bottom=0.125,
                    wspace=plot_sep, hspace=plot_sep*8
                )
                # ic([(r, c) for r in range(n_row) for c in range(n_col)])
                # for r in iter(range(n_row))
                for r, c in iter((r, c) for r in range(n_row) for c in range(n_col)):
                    idx_ord = r * n_col + c
                    ic(r, c, idx_ord)
                    ax_ = fig.add_subplot(n_row, n_col, idx_ord+1)
                    # idx_ord = idx_ord + offset
                    idx_sz = idxs_sz[idx_ord]
                    ax_.plot(self.centers[idx_sz], lw=0.25, marker='o', ms=0.3)
                    ax_.set_title(f'Seg #{idx_ord+1}, sz {counts[idx_sz]}', fontdict=dict(fontsize=8))
                    ax_.set_ylim(ylim)
                    # ax_.axes.xaxis.set_ticks([])
                    # ax_.axes.yaxis.set_ticks([])
                    ax_.axes.xaxis.set_ticklabels([])
                    ax_.axes.yaxis.set_ticklabels([])
                plt.suptitle('Segment plot, ordered by frequency')
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
        method='hierarchical', cls_kwargs=dict(distance_threshold=0.001),
        plot_dist=40,
        plot_segments=(5, 4)
    )
