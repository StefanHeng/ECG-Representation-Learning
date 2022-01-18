# from enum import Enum
import random
import pickle

# from scipy.spatial import KDTree
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, Birch
from matplotlib.widgets import Slider

from util import *
from ecg_loader import EcgLoader


D_EXP = config('path-export')


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

    Distance between segments are computed using l2 norm
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
        self.nn: KDTree = None  # Nearest Neighbor for decoding

    def __repr__(self):
        return f'<{self.__class__.__qualname__} k={self.k} pad={self.pad_}>'

    def save(self):
        """
        Save current tokenizer object into pickle
        """
        fnm = f'ecg-tokenizer, {now(sep="-")}'
        with open(os.path.join(D_EXP, f'{fnm}.pickle'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fnm, dir_=D_EXP):
        with open(os.path.join(dir_, fnm), 'rb') as f:
            tokenizer = pickle.load(f)
            assert isinstance(tokenizer, cls)
            return tokenizer

    def __call__(
            self,
            sig: np.ndarray,
            plot: Union[bool, tuple[int, int]] = False, plot_args: dict = None
    ):
        """
        :param sig: Signals or batch of signals to decode, of dim `d_prev::l`, where `l` is the number of samples
        :param plot: If true, the decoded ids are plotted with input signal in grid
        :return: 2-tuple of (Decoded cluster ids, segment means), each of dim `d_pre::n`,
            where `n` is the number of segments

        .. note:: Signals are decoded along the last dimension
        """
        # TODO: Generalize: now, works on signals of the same sample length only
        sp = sig.shape
        ic(sig.shape)
        sp, l = sp[:-1], sp[-1]
        sig = self.pad(sig)
        if plot:
            segs = sig.reshape(-1, self.k).copy()
        else:
            segs = sig.reshape(-1, self.k)
        means = segs.mean(axis=-1, keepdims=True)
        segs -= means
        dists, idxs = self.nn.query(segs, k=1, return_distance=True)
        ic(dists.max(), dists.min())  # Sanity check
        shape = sp+(-1,)
        ids = idxs.reshape(shape)  # Token ids
        means = means.reshape(shape)
        if plot:
            scale = plot_args['scale'] if 'scale' in plot_args else 1

            ln = ids.shape[-1]
            sig_ = sig.reshape(-1, sig.shape[-1])
            ids_ = ids.reshape(-1, ln)
            means_ = means.reshape(-1, ln)
            dists_ = dists.reshape(-1, ln).sum(axis=-1)
            idxs_sort = np.argsort(-dists_)

            with sns.axes_style('whitegrid', {'grid.linestyle': ':'}):
                n_col, n_row = plot
                sz_bch = n_row * n_col

                cs = sns.color_palette(palette='husl', n_colors=16)
                cs = [cs[2], cs[10], cs[12]]
                fig = plt.figure(figsize=(n_col*6*scale, n_row*2*scale), constrained_layout=False)
                n_ = max(n_col, n_row)
                margin_h, plot_sep = 0.075/n_, 0.075/n_
                bot = 0.025 * n_
                plt.subplots_adjust(
                    left=margin_h/2, right=1-margin_h/2,
                    top=0.95, bottom=bot,
                    wspace=plot_sep, hspace=plot_sep*4
                )
                d_ax = dict()

                def update(idx_, first=False):
                    i_bch = idx_
                    offset = i_bch * sz_bch
                    idxs_ord = np.arange(sz_bch) + offset
                    idxs_dist = idxs_sort[idxs_ord]
                    sigs_ori = sig_[idxs_dist, :]
                    sigs_dec = (self.centers[ids_[idxs_dist]] + means_[idxs_dist, :, np.newaxis]).reshape(sz_bch, -1)

                    vals = np.concatenate([sigs_ori[sigs_ori != 0], sigs_dec[sigs_dec != 0]])
                    m, std = vals.mean(), vals.std()
                    mi, ma = m-3*std, m+3*std
                    ylim = [mi, ma]
                    kwargs = dict(lw=0.25, marker='o', ms=0.3)
                    n = sigs_ori.shape[-1]
                    bounds = np.arange(0, math.ceil(n/self.k)+1) * self.k

                    for r, c in iter((r, c) for r in range(n_row) for c in range(n_col)):
                        it_c = iter(cs)
                        idx = r * n_col + c
                        if first:
                            ax = d_ax[idx] = fig.add_subplot(n_row, n_col, idx+1)
                            ax.plot(sigs_ori[idx], label='Signal, original', c=next(it_c), **kwargs)
                            ax.plot(sigs_dec[idx], label='Signal, decoded', c=next(it_c), **kwargs)
                            ax.vlines(
                                x=bounds, ymin=mi, ymax=ma,
                                lw=0.3, alpha=0.7, colors=next(it_c), label='Segment boundaries'
                            )
                        else:
                            # ic(d_ax)
                            ax = d_ax[idx]
                            ln1, ln2 = ax.lines
                            ln1.set_ydata(sigs_ori[idx])
                            ln2.set_ydata(sigs_dec[idx])
                        ax.set_ylim(ylim)
                        ax.set_title(
                            f'Signal #{idxs_dist[idx]}, total dist = {round(dists_[idxs_dist[idx]], 2)}',
                            fontdict=dict(fontsize=8)
                        )
                        ax.axes.xaxis.set_ticklabels([])
                        ax.axes.yaxis.set_ticklabels([])
                        if first and r == 0 and c == 0:
                            ax.legend()
                ax_sld = plt.axes([margin_h*2, bot/2, 1-margin_h*4, 0.01])
                n_bch = math.ceil(sig_.shape[0] / sz_bch)
                init = 0
                slider = Slider(
                    ax_sld, 'Batch #', 0, n_bch-1, valinit=init, valstep=1,
                    color=sns.color_palette(palette='husl', n_colors=7)[3]
                )
                slider.vline._linewidth = 0  # Hides vertical red line marking init value
                update(init, first=True)
                slider.on_changed(update)
                plt.suptitle('Decoding plot, by descending fitness')
                plt.show()
        return ids, means

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
            plot_segments: Union[bool, tuple[int, int]] = False,
            plot_seg_sample: int = None
    ):
        """
        :param sigs: Array of shape N x C x L
        :param method: Clustering method
        :param cls_kwargs: Arguments to the clustering method
        :param plot_dist: If True, the counts for each cluster is plotted
            If integer give, the first most common classes are plotted
        :param plot_segments: If 2-tuple given, plots the segment centroid for each cluster in grid
        :param plot_seg_sample: If `plot_segments` and given, samples from each cluster are plotted

        Symbols for each signal channel learned separately
            Clustering labels are assigned for each channel, for N x C labels in total, in the input order

        TODO: Learn symbol across all channels
        """
        if self.pad_:
            sigs = self.pad(sigs)
            assert sigs.shape[-1] % self.k == 0
        n_rec = sigs.shape[0]
        segs = sigs.reshape(-1, self.k)
        n_segs = segs.shape[0]
        log(f'Clustering {logs(n_rec, c="i")} signals => {logs(n_segs, c="i")} segments')
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

        self.centers = np.stack([
            segs[lbs == lb].mean(axis=0) for lb in ids_vocab
        ])
        self.nn = KDTree(self.centers)
        ic(self.centers, self.centers.shape)
        if plot_segments:
            with sns.axes_style('whitegrid', {'grid.linestyle': ':'}):
                n_col, n_row = plot_segments
                sz_bch = n_row * n_col
                n_samp = plot_seg_sample

                cs = sns.color_palette(palette='husl', n_colors=sz_bch)
                fig = plt.figure(figsize=(n_col * 3, n_row * 2), constrained_layout=False)
                n = max(n_col, n_row)
                margin_h, plot_sep = 0.125/n, 0.125/n
                bot = 0.0125 * n
                plt.subplots_adjust(
                    left=margin_h/2, right=1-margin_h/2,
                    top=0.925, bottom=bot,
                    wspace=plot_sep, hspace=plot_sep*8
                )
                d_lns = dict()
                d_axs = dict()

                def update(idx, first=False):
                    i_bch = idx  # Batch
                    offset = i_bch * sz_bch
                    idxs_ord = np.arange(sz_bch) + offset  # Ordering for display
                    idxs_sz = idxs_sort[idxs_ord]  # Internal ordering based on counts
                    mi, ma = self.centers[idxs_sz].min(), self.centers[idxs_sz].max()
                    ylim = max(abs(mi), abs(ma)) * 1.25
                    ylim = [-ylim, ylim]
                    it_c = iter(cs)
                    for r, c in iter((r, c) for r in range(n_row) for c in range(n_col)):
                        clr = next(it_c)
                        idx_ord = r * n_col + c
                        if first:
                            ax = d_axs[idx_ord] = fig.add_subplot(n_row, n_col, idx_ord+1)
                        else:
                            ax = d_axs[idx_ord]

                        idx_sz: int = idxs_sz[idx_ord]
                        if first:
                            # `plot` returns List containing single element
                            d_lns[idx_ord] = ax.plot(self.centers[idx_sz], lw=0.75, marker='o', ms=0.9, c=clr)[0]
                        else:
                            d_lns[idx_ord].set_ydata(self.centers[idx_sz])
                        if n_samp:
                            kwargs = dict(lw=0.25, marker='o', ms=0.3, c=clr, alpha=0.5)
                            idxs_samp = np.arange(n_segs)[lbs == idx_sz]
                            sz_cls = idxs_samp.shape[0]
                            n_exist = len(ax.lines)
                            n_new = min(sz_cls, n_samp)
                            if sz_cls > n_samp:
                                ys = (segs[idxs_samp[i]] for i in random.sample(range(sz_cls), n_samp))
                            else:
                                ys = (segs[i] for i in idxs_samp)
                            if n_new <= n_exist:
                                for i in range(n_new):
                                    ax.lines[i].set_ydata(next(ys))
                                for _ in range(n_exist - n_new):  # Drop additional lines
                                    ax.lines[-1].remove()
                            else:  # n_exist < n_new
                                for i in range(n_exist):
                                    ax.lines[i].set_ydata(next(ys))
                                for i in range(n_exist, n_new):
                                    ax.plot(next(ys), **kwargs)
                        ax.set_title(f'Seg #{idx_ord+offset+1}, sz {counts[idx_sz]}', fontdict=dict(fontsize=8))
                        ax.set_ylim(ylim)
                        ax.axes.xaxis.set_ticklabels([])
                        ax.axes.yaxis.set_ticklabels([])
                plt.suptitle('Segment plot, ordered by frequency')
                ax_sld = plt.axes([margin_h*4, bot/2, 1-margin_h*8, 0.01])
                n_bch = math.ceil(self.centers.shape[0] / sz_bch)
                init = 0
                slider = Slider(
                    ax_sld, 'Batch #', 0, n_bch-1, valinit=init, valstep=1,
                    color=sns.color_palette(palette='husl', n_colors=7)[3]
                )
                slider.vline._linewidth = 0
                update(init, first=True)
                slider.on_changed(update)
                plt.show()


if __name__ == '__main__':
    from icecream import ic

    random.seed(config('random_seed'))

    el = EcgLoader('CHAP_SHAO')  # TODO: Generalize to multiple datasets
    et = EcgTokenizer(k=16)

    def sanity_check():
        s = el[0]
        ic(s.shape)
        s_p = et.pad(s)
        ic(s_p, s_p.shape)
    # sanity_check()

    def train():
        # et.fit(el[:16], method='dbscan', cls_kwargs=dict(eps=0.01, min_samples=3))
        # et.fit(el[:128], method='birch', cls_kwargs=dict(threshold=0.05))
        et.fit(
            el[:16],
            method='hierarchical', cls_kwargs=dict(distance_threshold=0.0008),
            plot_dist=40,
            plot_segments=(5, 4),
            plot_seg_sample=16
        )
        et.save()
    # train()

    def check_save():
        fnm = 'ecg-tokenizer, 2022-01-17 20-56-59.pickle'
        et = EcgTokenizer.from_pickle(fnm)
        # ic(et, vars(et))
        ic(len(el))
        # et(el[1020:1024], plot=(2, 3))
        et(el[400:420], plot=(2, 3), plot_args=dict(scale=2))
    check_save()
