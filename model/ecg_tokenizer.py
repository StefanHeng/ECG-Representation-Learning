import pickle


from numpy.random import default_rng
from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, Birch, KMeans
from matplotlib.widgets import Slider

from util import *
from model.ecg_loader import EcgLoader


D_EXP = config('path-export')


def cluster_args(method, cls_kwargs=None, cuml=False):
    """
    :return: Clustering full keyword arguments from custom default arguments
    """
    d_kwargs = dict(
        hierarchical=dict(n_clusters=None, linkage='average'),
        dbscan=dict(min_samples=5),
        optics=dict(min_samples=5),
        birch=dict(n_clusters=None),
        kmeans=dict(n_init=8, max_iter=256, init='scalable-k-means++' if cuml else 'k-means++')
    )
    return d_kwargs[method] | cls_kwargs


def cluster(data: np.ndarray = None, method='spectral', cls_kwargs=None, fit=True):
    """
    :param data: Data points to cluster, in N x D
    :param method: Clustering method
    :param cls_kwargs: Arguments to the clustering method
    :param fit: If True, the clustering object is fit on `data`
    """
    kwargs = cluster_args(method, cls_kwargs)

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

    def kmeans():
        assert 'n_clusters' in kwargs
        return KMeans(**kwargs)
    d_f = dict(
        hierarchical=hierarchical,
        dbscan=dbscan,
        optics=optics,
        birch=birch,
        kmeans=kmeans
    )
    return d_f[method]().fit(data) if fit else d_f[method]()


D_CLS_TH = dict(  # Mapping from clustering method to threshold keyword
    hierarchical='distance_threshold',
    dbscan='eps',
    optics='max_eps',
    birch='threshold',
    kmeans='n_clusters'
)
D_CLS_NM = dict(  # Display name of each clustering method
    hierarchical='Agglomerative',
    dbscan='DBSCAN',
    optics='OPTICS',
    birch='Birch',
    kmeans='K-means'
)


class EcgPadder:
    def __init__(self, k: int = 2**3, pad='shift'):
        """
        :param k: Length of each segment
        :param pad: Signal padding scheme, one of [`zero`, `shift`]
            If `zero`, the last segment is padded with 0
            If `shift`, the last segment is padded with the last `k` values
        .. note:: Signals are padded at the end so that sample length is a multiple of `k`
            i.e. Pads to the right only
        """
        self.k = k
        self.pad = pad

    def __call__(self, sig):
        """
        :param sig: 2D array of shape C x L, or 3D array of shape N x C x L

        The last dimension is padded
        """
        sp = sig.shape
        sp, l = sp[:-1], sp[-1]
        n_pad = self.k - (l % self.k)
        w = (0, 0), (0, n_pad)  # TODO: generalize?

        def pad_shift(  # To preserve as much morphology
                a: np.ndarray,
                pad_width: tuple[int, int],
                iaxis: int,  # per numpy.pad; intended for last axis only
                kwargs
        ):
            strt, end = pad_width
            if not(strt == 0 and end == 0):  # Omit all but the last axis
                a[:strt] = a[strt:strt*2]
                a[-end:] = a[-2*end:-end]

        def zero():
            return np.pad(sig, pad_width=w, mode='constant', constant_values=0)

        def shift():
            return np.pad(sig, pad_width=w, mode=pad_shift, constant_values=0)
        d_f = dict(  # TODO: other padding schemes?
            zero=zero,
            shift=shift
        )

        if n_pad == 0:
            return sig
        else:
            sig = sig.reshape(-1, l)  # Enforce 2D mat
            return d_f[self.pad]().reshape(sp + (l+n_pad,))


class EcgTokenizer:
    """
    Tokenize ECG signals into symbols, given normalized signals in range [0, 1]

    Distance between segments are computed using l2 norm
    """

    def __init__(self, k: int = 2**3, pad='shift', backend='sklearn'):
        """
        :param k: Number of samples in each segment, where each segment is assigned a token id
        :param pad: Padding scheme, see `EcgPadder`
        :param backend: clustering algorithms backend, one of [`sklearn`, `nex`, `cuml`
        """
        self.k = k
        self.padder = EcgPadder(k, pad)

        self.centers = None  # of dim (N_cls, k); centers[i] stores the cluster mean for id `i`
        self.lens = None  # of dim (N_cls); cluster sizes
        self.nn: KDTree = None  # Nearest Neighbor for decoding
        # Customized NN, filtering out centroids by count or relative size
        self.nns: dict[Union[int, float], EcgTokenizer.CustNN] = {}

        self.fit_method = None  # Hyperparameters for the last call to `fit`
        self.n_sig = None
        self.cls_th = None

        self.backend = backend
        assert backend in ['sklearn', 'nex', 'cuml']
        if backend == 'nex':
            ic(self.backend)
            from sklearnex import patch_sklearn  # patch & re-import
            patch_sklearn()
            from sklearn.neighbors import KDTree
            from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, Birch, KMeans

    def __repr__(self):
        return f'<{self.__class__.__qualname__} k={self.k} pad={self.pad_}>'

    def save(self):
        """
        Save current tokenizer object into pickle
        """
        fnm = f'ecg-tokenizer, {now(sep="-")}, k={self.k}, cls={self.fit_method}, n={self.n_sig}, e={self.cls_th}'
        with open(os.path.join(D_EXP, f'{fnm}.pickle'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fnm, dir_=D_EXP):
        with open(os.path.join(dir_, fnm), 'rb') as f:
            tokenizer = pickle.load(f)
            assert isinstance(tokenizer, cls)
            return tokenizer

    class CustNN:
        """
        Nearest neighbor wrapper, with a lowerbound thresholding on allowed points
        """
        def __init__(self, data: np.ndarray, lens: np.ndarray, th: Union[int, float]):
            """
            :param data: array of clustered centroids, see `EcgTokenizer`
            :param lens: array of cluster sizes, see `EcgTokenizer`
            :param th: Lowerbound threshold on lens to remove centroids
                If int, as absolute threshold
                If float in range (0, 1), as relative threshold across the total number of points
            """
            n = data.shape[0]
            if isinstance(th, int):
                self.th = th
            else:
                assert isinstance(th, float) and 0 < th < 1
                self.th = round(lens.sum() * th)
            self.data = data[lens >= th]
            self.nn = KDTree(self.data)
            log(f'Custom Nearest Neighbor instantiated '
                f'with sizes below {th} removed - # cluster {logi(n)} -> {logi(self.data.shape[0])}')

        def __getitem__(self, idx):
            return self.data[idx]

        def query(self, pts: np.ndarray, **kwargs):
            return self.nn.query(pts, **kwargs)

    def __call__(
            self,
            sig: Union[int, np.ndarray], th: Union[int, float] = None,
            plot: Union[bool, tuple[int, int]] = False, plot_args: dict = None
    ):
        """
        :param sig: signals or batch of signals to encode, of dim `d_prev::l`, where `l` is the number of samples
        :param th: Lowerbound threshold for filtering clusters by size; See `EcgTokenizer.CustNN`
        :param plot: If true, the decoded ids are plotted with input signal in grid
        :param plot_args: Sample signal plot arguments
            Supported keys: [`scale`]
        :return: 2-tuple of (Decoded cluster ids, segment means), each of dim `d_pre::n`,
            where `n` is the number of segments

        .. note:: Signals are decoded along the last dimension
        """
        # TODO: Generalize: now, works on signals of the same sample length only
        sp = sig.shape
        sp, l = sp[:-1], sp[-1]
        sig = self.padder(sig)
        if plot:
            segs = sig.reshape(-1, self.k).copy()
        else:
            segs = sig.reshape(-1, self.k)
        means = segs.mean(axis=-1, keepdims=True)
        segs -= means

        if th is None:
            dists, idxs = self.nn.query(segs, k=1, return_distance=True)
        else:
            if th not in self.nns:
                self.nns[th] = EcgTokenizer.CustNN(self.centers, self.lens, th)
            dists, idxs = self.nns[th].query(segs, k=1, return_distance=True)
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
                margin_h, plot_sep = 0.1/n_, 0.1/n_
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
                    sigs_dec = (self.decode(ids_[idxs_dist], th=th) + means_[idxs_dist, :, np.newaxis]).reshape(sz_bch, -1)

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
                            ax = d_ax[idx]
                            ln1, ln2 = ax.lines
                            ln1.set_ydata(sigs_ori[idx])
                            ln2.set_ydata(sigs_dec[idx])
                        ax.set_ylim(ylim)
                        ax.set_title(
                            f'Signal #{idxs_dist[idx]}, total dist = {round(dists_[idxs_dist[idx]], 2)}',
                            fontdict=dict(fontsize=8)
                        )
                        if first:
                            ax.tick_params(axis='x', labelsize=7)
                            ax.tick_params(axis='y', labelsize=7)
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
                t = rf'Decoded signal plot by descending fitness, with {D_CLS_NM[self.fit_method]} clustering '\
                    rf'on $k={self.k}$, $n={self.n_sig}$, $\epsilon={self.cls_th}$'
                if th is not None:
                    t += f', th={self.nns[th].th}'
                plt.suptitle(t)
                plt.show()
        return ids, means

    def decode(self, idx: Union[int, np.ndarray], th=None) -> np.ndarray:
        if th is None:
            return self.centers[idx]
        else:
            return self.nns[th][idx]

    def fit(
            self, sigs: np.ndarray, method='spectral', cls_kwargs=None,
            plot_args: dict[str] = None, save=False
    ):
        """
        :param sigs: Array of shape N x C x L
        :param method: Clustering method
        :param cls_kwargs: Arguments to the clustering method
        :param plot_args: Plotting keyword arguments
            plot_dist - Union[bool, int]: If True, the counts for each cluster is plotted, ordered by cluster size
                If True, cluster size is inferred
                If integer given, the first most common classes are plotted
            plot_segments - Union[bool, tuple[int, int]]: If 2-tuple given,
                plots the segment centroid for each cluster in grid
            plot_seg_sample - int: If `plot_segments` and given, samples from each cluster are plotted
            seed - int: Random sampling seed in segment cluster plot
            scale - Union[int, float]: Plotting scale
            save_fig_ - bool: If true, plots are saved as png
            birch_args - dict: If given, cluster sizes below and equal to `birch_th` are further clustered and merged
        :param save: If true, the trained tokenizer is saved as pickle

        Symbols for each signal channel learned separately
            Clustering labels are assigned for each channel, for N x C labels in total, in the input order

        TODO: Learn symbol across all channels
        """
        if plot_args is None:
            plot_args = dict()
        plot_dist, plot_segments, plot_seg_sample, seed, scale, save_fig_ = (
            plot_args.get(k, False) for k in [
                'plot_dist', 'plot_segments', 'plot_seg_sample', 'seed', 'scale', 'save_fig_'
            ]
        )
        if scale is False:
            scale = 1

        self.fit_method = method
        self.n_sig = len(sigs)
        self.cls_th = cls_kwargs[D_CLS_TH[method]]
        sigs = self.padder(sigs)
        n_rec = sigs.shape[0]
        segs = sigs.reshape(-1, self.k)
        n_segs = segs.shape[0]
        log(f'Clustering {logi(n_rec)} signals => {logi(n_segs)} length-{logi(self.k)} segments... ')
        means = segs.mean(axis=-1, keepdims=True)
        segs -= means  # Set mean of each segment to 0

        # segs_ = segs.copy()
        log(f'Begin clustering with [{logi(D_CLS_NM[self.fit_method])}] and l2 norm threshold [{logi(self.cls_th)}]')
        strt = datetime.datetime.now()
        # cls_2nd = cluster(method=method, cls_kwargs=cls_kwargs, fit=False)
        # cls = cluster(segs, method=method, cls_kwargs=(cls_kwargs | dict(n_clusters=cls_2nd)))

        # cls_kmeans = cluster(method='kmeans', cls_kwargs=dict(n_clusters=1024 + 512), fit=False)
        # cls = cluster(segs, method=method, cls_kwargs=(cls_kwargs | dict(n_clusters=cls_kmeans)))

        cls = cluster(segs, method=method, cls_kwargs=cls_kwargs)
        # np.testing.assert_array_equal(segs, segs_)
        lbs = cls.labels_  # Treated as the token id
        log(f'Clustering completed in {logi(fmt_dt(datetime.datetime.now()-strt))}')

        ids_vocab, counts = np.unique(lbs, return_counts=True)  # `ids_vocab` sorted ascending
        msk_cls = ids_vocab != -1  # For `DBSCAN`, points with labels -1 are outliers
        count_out = None
        if not np.all(msk_cls):  # Outlier label available; Ensure np.arange(ids_vocab.size) == np.sort(ids_vocab)
            idx_out = np_index(ids_vocab, -1)
            count_out = counts[idx_out]
            ids_vocab, counts = ids_vocab[msk_cls], counts[msk_cls]
        idxs_sort = np.argsort(-counts)
        s = f'{logi(ids_vocab.size)} clusters produced, with counts {counts[idxs_sort][:64]} '
        log((s + f'... and outlier size {logi(count_out)}') if count_out is not None else s)

        # lb2lb = np.arange(ids_vocab.size)[idxs_sort]
        # ic(lbs, idxs_sort, counts[idxs_sort[:5]])  # TODO: process sub-clusters after birch?
        # ic(np.argsort(idxs_sort)[1630])
        # Old label => New label that's ordered by size, i.e. Label with the largest cluster size has label 0

        # lb2lb = np.argsort(idxs_sort)
        # lbs = lb2lb[lbs]
        # ids_vocab, counts = np.unique(lbs, return_counts=True)
        # np.testing.assert_array_equal(counts[::-1], np.sort(counts))
        # ic(ids_vocab, counts)
        # idxs_sort = np.argsort(-counts)
        # ic(idxs_sort, counts[idxs_sort[:5]])
        # # np.testing.assert_array_equal(idxs_sort, np.arange(ids_vocab.size))
        #
        # # Not necessarily true that, idxs_sort == np.arange(ids_vocab.size),
        # # since labels with the same cluster sizes may not be ordered by *label* magnitude
        # ic(ids_vocab[counts <= 3])  # Cluster labels smaller than a threshold
        # exit(1)

        if plot_dist:
            counts_sort = counts[idxs_sort]
            if plot_dist is True:  # Plot cluster sizes
                # ratio = 0.6  # Empirically set  # Approach 1, up until cover the majority of the data
                # sums = np.cumsum(counts_sort)
                # n = np.where(sums > sums[-1] * ratio)[0][0]

                ratio = 0.003  # Approach 2, up until change in cluster size is not great
                min_diff = max(counts_sort.size * ratio, 5)
                diffs = -np.diff(counts_sort)
                for i_ in range(diffs.size-1, 0, -1):  # non-ascending
                    if diffs[i_-1] < diffs[i_]:
                        diffs[i_-1] = diffs[i_]
                ext = int(min(segs.size*ratio, 20))  # Extend a bit more
                n = np.where(diffs < min_diff)[0][0] + ext
            else:
                n = plot_dist
                assert isinstance(n, int)
            y = counts_sort[:n]
            rank = (np.arange(counts.size) + 1)[:n]

            plt.figure(figsize=(18*scale, 6*scale))
            plt.plot(rank, y, marker='o', lw=0.5, ms=1, label='# sample')

            precision = 10
            (a_, b_), (x_, y_) = fit_power_law(rank, y, return_fit=precision)
            a_, b_ = round(a_, 2), round(b_, 2)
            n_ = n*precision
            plt.plot(x_[:n_], y_[:n_], lw=0.4, ls='-', label=fr'Fitted power law: ${a_} x^{{{b_}}}$')
            r2_ = round(r2(y, a_ * np.power(rank, b_)), 5)
            ax_ = plt.gca()
            ax_.text(0.75, 0.95, rf'$R^2 = {r2_}$', transform=ax_.transAxes)
            log(f'R-squared for fitted curve: {logi(r2_)}')

            plt.xlabel('Cluster, ranked')
            plt.ylabel('Frequency')
            t = rf'{D_CLS_NM[self.fit_method]} Clustering Rank-frequency plot ' \
                rf'with $k={self.k}$, $n={self.n_sig}$, $\epsilon={self.cls_th}$'
            plt.title(t)
            plt.legend()
            if save_fig_:
                t = t.replace('$', '').replace(r'\epsilon', 'e')
                save_fig(f'{t}, {now(sep="-")}')
            else:
                plt.show()

        self.lens = np.array([len(np.where(lbs == lb)[0]) for lb in ids_vocab])
        self.centers = np.stack([segs[lbs == lb].mean(axis=0) for lb in ids_vocab])
        self.nn = KDTree(self.centers)
        # np.testing.assert_array_equal(segs, segs_)

        # Enforce the final `ids_vocab` = np.arange(|ids_vocab|), by modifying the labels returned from clustering
        if ids_vocab.size != ids_vocab[-1]+1:  # Some integer labels have cluster size of 0
            log('Clustering labels not consecutive - modifying labels... ', c='w')
            lb2idx = np.full(ids_vocab[-1]+1, nan)  # Such clusters will have nan assigned
            vocab = set(ids_vocab)
            id_assign = 0
            for id_ori in ids_vocab:
                if id_ori in vocab:
                    lb2idx[id_ori] = id_assign
                    id_assign += 1
            lbs = lb2idx[lbs]  # Map to new labels
            assert np.all(~np.isnan(lbs))
            ids_vocab = np.arange(ids_vocab.size)
            np.testing.assert_array_equal(np.unique(lbs), ids_vocab)

        if plot_segments:
            rng = default_rng() if seed is None else default_rng(seed)

            with sns.axes_style('whitegrid', {'grid.linestyle': ':'}):
                n_col, n_row = plot_segments
                sz_bch = n_row * n_col
                n_samp = plot_seg_sample
                n_vocab = ids_vocab.size
                n_bch = math.ceil(self.centers.shape[0] / sz_bch)

                cs = sns.color_palette(palette='husl', n_colors=sz_bch)
                fig = plt.figure(figsize=(n_col*3*scale, n_row*2*scale), constrained_layout=False)
                n = max(n_col, n_row)
                margin_h, plot_sep = 0.125/n, 0.125/n
                bot = 0.0125 * n
                plt.subplots_adjust(
                    left=margin_h/2, right=1-margin_h/2,
                    top=0.925, bottom=bot,
                    wspace=plot_sep, hspace=plot_sep*8
                )
                d_axs = dict()

                def update(idx, first=False):
                    i_bch = idx  # Batch
                    offset = i_bch * sz_bch

                    if i_bch == n_bch-1:  # Last batch
                        n_plot = n_vocab % sz_bch
                        if n_plot == 0:
                            n_plot = sz_bch
                    else:
                        n_plot = min(sz_bch, n_vocab)
                    idxs_ord = np.arange(n_plot) + offset  # Ordering for display
                    idxs_sz = idxs_sort[idxs_ord]  # Internal ordering based on counts
                    mi, ma = self.centers[idxs_sz].min(), self.centers[idxs_sz].max()
                    ylim = max(abs(mi), abs(ma)) * 1.25
                    update.ylim = ylim = [-ylim, ylim]
                    it_c = iter(cs)
                    for r, c in iter(
                            (r, c) for r in range(n_row) for c in range(n_col) if r*n_col + c + offset < n_vocab
                    ):
                        clr = next(it_c)
                        idx_ord = r*n_col + c
                        if first:
                            ax = d_axs[idx_ord] = fig.add_subplot(n_row, n_col, idx_ord+1)
                        else:
                            ax = d_axs[idx_ord]

                        idx_sz: int = idxs_sz[idx_ord]
                        if first:
                            # `plot` returns List containing single element
                            ax.plot(self.centers[idx_sz], lw=0.75, marker='o', ms=0.9, c=clr)
                        else:
                            ax.lines[0].set_ydata(self.centers[idx_sz])  # Centroid plot is the 1st `Line`
                        if n_samp:
                            kwargs = dict(lw=0.25, marker='o', ms=0.3, c=clr, alpha=0.5)
                            idxs_samp = np.where(lbs == idx_sz)[0]
                            sz_cls = idxs_samp.size
                            n_exist = len(ax.lines)-1  # The 1st line for centroid
                            n_new = min(sz_cls, n_samp)
                            if sz_cls > n_samp:
                                ys = (segs[idxs_samp[i]] for i in rng.choice(sz_cls, size=n_samp, replace=False))
                            else:
                                ys = (segs[i] for i in idxs_samp)
                            if n_new <= n_exist:
                                for i in range(n_new):
                                    ax.lines[i+1].set_ydata(next(ys))
                                for _ in range(n_exist-n_new):  # Drop additional lines
                                    ax.lines[-1].remove()
                            else:  # n_exist < n_new
                                for i in range(n_exist):
                                    ax.lines[i+1].set_ydata(next(ys))
                                for i in range(n_exist, n_new):
                                    ax.plot(next(ys), **kwargs)
                            assert len(ax.lines) == n_new + 1
                            np.testing.assert_array_equal(ax.lines[0].get_ydata(), self.centers[idx_sz])
                        ax.set_title(f'Seg #{idx_ord+offset+1}, sz {counts[idx_sz]}', fontdict=dict(fontsize=8*scale))
                        ax.set_ylim(ylim)
                        ax.axes.xaxis.set_ticklabels([])
                        ax.axes.yaxis.set_ticklabels([])

                        if r == 0 and c == 0:  # Set only once
                            idxs = [idx for idx, txt in enumerate(fig.texts) if ('Y axis' in txt._text)]
                            for i in reversed(idxs):
                                del fig.texts[i]
                            y_mi, y_ma = ylim
                            y_mi, y_ma = sig_d(y_mi, n=3), sig_d(y_ma, n=3)
                            plt.figtext(0.8, 0.96, f'Y axis: $[{y_mi}, {y_ma}]$', fontdict=dict(fontsize=10*scale))
                    for idx in range(n_plot):
                        d_axs[idx].set_visible(True)
                    for idx in range(n_plot, sz_bch):
                        d_axs[idx].set_visible(False)

                init = 0
                update(init, first=True)
                if n_bch > 1:
                    ax_sld = plt.axes([margin_h * 4, bot / 2, 1 - margin_h * 8, 0.01])

                    slider = Slider(
                        ax_sld, 'Batch #', 0, n_bch-1, valinit=init, valstep=1,
                        color=sns.color_palette(palette='husl', n_colors=7)[3]
                    )
                    slider.vline._linewidth = 0
                    slider.on_changed(update)
                t = rf'{D_CLS_NM[self.fit_method]} Cluster centroid plot by frequency '
                if n_samp:
                    t += rf'with {n_samp} random samples & '
                t += rf'with $k={self.k}$, $n={self.n_sig}$, $\epsilon={self.cls_th}$'
                plt.suptitle(t)
                if save_fig_:
                    t = t.replace('$', '').replace(r'\epsilon', 'e')
                    save_fig(f'{t}, 1st frame, {now(sep="-")}')  # Save both frames
                    update(n_bch-1)
                    save_fig(f'{t}, last frame, {now(sep="-")}')  # Save both frames
                else:
                    plt.show()
        if save:
            self.save()


if __name__ == '__main__':
    from icecream import ic

    seed_ = config('random_seed')

    el = EcgLoader(dataset_name='CHAP_SHAO', normalize=3.5)  # TODO: Generalize to multiple datasets
    et = EcgTokenizer(k=8)
    # et = EcgTokenizer(k=8, backend='nex')

    def sanity_check():
        s = el[0]
        ic(s.shape)
        s_p = et.padder(s)
        ic(s_p, s_p.shape)
    # sanity_check()

    def train():
        et.fit(
            # el[:16], method='hierarchical', cls_kwargs=dict(distance_threshold=4e-4),
            # el[:2], method='dbscan', cls_kwargs=dict(eps=8e-3),
            # el[:2], method='birch', cls_kwargs=dict(threshold=6e-4),
            el[:32], method='kmeans', cls_kwargs=dict(
                n_clusters=256 * 16,
                # verbose=True,
                random_state=config('random_seed'),
                algorithm='full'
            ),
            plot_args=dict(
                plot_dist=True,
                plot_segments=(5, 4),
                plot_seg_sample=64,
                seed=seed_,
                save_fig_=True,
            ),
            # save=True
        )
    train()

    def check_save():
        # fnm = 'ecg-tokenizer, 2022-01-19 20-42-28, k=16, cls=birch, n=256, e=0.0006.pickle'
        fnm = 'ecg-tokenizer, 2022-01-24 22-07-41, k=8, cls=kmeans, n=256, e=1024.pickle'
        et = EcgTokenizer.from_pickle(fnm)
        # ic(et, vars(et))
        ic(len(el))
        # et(el[1020:1024], plot=(2, 3))
        et(el[400:420], th=3, plot=(2, 3), plot_args=dict(scale=2))
    # check_save()
