"""
Transformations on multichannel 1D signals similar to those for 2D images in torchvision
"""

from scipy.stats import norm

from ecg_transformer.util import *


_NormArg = Union[str, Tuple[str, Union[float, int]]]
NormArg = Union[_NormArg, List[_NormArg]]
Transform = Callable[[np.ndarray], np.ndarray]


class NormalizeSingle:
    def __init__(self, arr, scheme: str = 'std', arg: Union[float, int] = None):
        """
        :param scheme: normalize scheme: one of [`global`, `std`, `norm`, `none`]
            Normalization is done per channel/lead
            If `global`, normalize by global minimum and maximum
            If `std`, normalize by subtracting mean & dividing 1 standard deviation
            If `norm`, normalize by subtracting mean & dividing range based on standard deviation percentile
        :param arg:  Intended for `norm` or `std` scheme
            FYI for `norm`:
                pnorm(1) ~= 0.841
                pnorm(2) ~= 0.977
                pnorm(3) ~= 0.999
                pnorm(4) ~= 0.99997
        """
        assert scheme in ['global', 'std', 'norm', 'none']
        self.norm_meta = None
        self.scheme = scheme
        # TODO: not sure why many functions return nan, even when no nan elements are present
        if scheme in ['std', 'norm']:
            if arg is None:
                arg = 1 if scheme == 'std' else 2
            else:
                assert isinstance(arg, (float, int))
            self.arg = arg
        else:
            self.arg = None

        # shapes are (1, 12, 1)
        if scheme == 'global':
            self.norm_meta = (
                np.nanmin(arr, axis=(0, -1), keepdims=True),
                np.nanmax(arr, axis=(0, -1), keepdims=True)
            )
        elif scheme == 'std':
            self.norm_meta = (
                np.nanmean(arr, axis=(0, -1), keepdims=True),
                np.nanstd(arr, axis=(0, -1), keepdims=True) * arg
            )
        elif scheme == 'norm':
            p = norm().cdf(arg) * 100
            self.norm_meta = (
                np.nanpercentile(arr, 100-p, axis=(0, -1), keepdims=True),
                np.nanpercentile(arr, p, axis=(0, -1), keepdims=True)
            )
        if self.norm_meta is not None:
            a, b = self.norm_meta
            self.norm_meta = a.astype(np.float32), b.astype(np.float32)

    def __call__(self, arr):
        if self.scheme == 'none':
            return arr
        else:
            if self.scheme in ['global', 'norm']:
                mi, ma = self.norm_meta
                sub, div = mi, (ma - mi)
            else:  # 'std'
                sub, div = self.norm_meta
            if arr.ndim == 2:
                sub, div = sub[0], div[0]  # (1, 12, 1) -> (12, 1)
            return (arr - sub) / div

    @staticmethod
    def _a2s(arr):
        return np.array2string(arr.flatten(), precision=2, separator=',', suppress_small=True, max_line_width=256)

    def __repr__(self):
        str_arg = f'arg={self.arg}' if self.arg is not None else ''
        return f'<{self.__class__.__qualname__} {self.scheme} {str_arg}>'


class Normalize:
    def __init__(self, arr: np.ndarray, normalize: NormArg = (('norm', 3), ('std', 1))):
        """
        :param arr: (n_samples, n_channels, n_leads) array for computing normalization statistics
        :param normalize: Normalization or a sequence of normalizations, as 2-tuples of (scheme, arg)
        """
        if isinstance(normalize, str) or isinstance(normalize, (tuple, list)) and not isinstance(normalize[0], tuple):
            norm_args = [normalize]
        else:
            norm_args = normalize
        norm_args = [(pr if isinstance(pr, tuple) else (pr,)) for pr in norm_args]
        assert all((isinstance(pr, tuple) and len(pr) in [1, 2]) for pr in norm_args)
        self.normalizers = []
        for pr in norm_args:
            normzer = NormalizeSingle(arr, *pr)
            self.normalizers.append(normzer)
            arr = normzer(arr)  # the normalizations are done sequentially

    def __call__(self, arr: np.array):
        assert arr.ndim in [2, 3]
        for normalizer in self.normalizers:
            arr = normalizer(arr)
        return arr

    def __repr__(self):
        return f'<{self.__class__.__qualname__} normalizers={self.normalizers}>'


class TimeEndPad:
    """
    pad along the last dimension, i.e. the time dimension, in the end, until multiples of k
    """
    def __init__(self, k: int, pad_kwargs: Dict = None):
        self.k = k
        self.pad_kwargs = pad_kwargs or dict()

    def __call__(self, arr: np.ndarray):
        l = arr.shape[-1]
        n_pad = self.k - (l % self.k)
        return np.pad(arr, pad_width=[(0, 0)] * (arr.ndim-1) + [(0, n_pad)], **self.pad_kwargs)

    def __repr__(self):
        return f'<{self.__class__.__qualname__} k={self.k}>'


class RandomResizedCrop:
    def __init__(self, scale=(0.5, 1)):
        pass


if __name__ == '__main__':
    from icecream import ic

    import ecg_transformer.util.ecg as ecg_util
    from ecg_transformer.preprocess import EcgDataset

    def check_norm_meta():
        dset = EcgDataset()
        a = dset[dset.idxs_processed]
        ic(a.shape)

        n = Normalize(a, 'global')
        ic(n, n.normalizers[0].norm_meta)
        n = Normalize(a, ('std', 3))
        ic(n, n.normalizers[0].norm_meta)
        n = Normalize(a, ('norm', 3))
        ic(n, n.normalizers[0].norm_meta)
    # check_norm_meta()

    def sample_sigs(dset: EcgDataset, n: int):
        n_sig, (n_ch, l) = len(dset), dset.dataset.shape[1:]
        idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))
        idxs_ch = np.random.randint(n_ch, size=n)
        return dset[idxs_sig][range(n), idxs_ch]

    def check_normalize(n: int = 128):
        # ed = EcgDataset(normalize='global')  # outliers are too large
        ed = EcgDataset(normalize='std')
        # ed = EcgDataset(normalize='norm')
        # a pretty aggressive 2-stage normalization, to clip as many signals to [0, 1]?
        # ed = EcgDataset(normalize=[('norm', 3), ('std', 1)])
        ic(ed.transform)

        sigs = sample_sigs(ed, n)

        # Fraction of signals, points that go beyond range 0, 1
        oor = (0 > sigs) | (sigs > 1)
        oor = dict(signal=np.any(oor, axis=-1).sum() / n, sample=oor.sum() / oor.size)
        ic(oor)

        plt.figure(figsize=(18, 6))
        plt.hlines([-1, 0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
        ecg_util.plot_1d(sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11))
        plt.show()
    # check_normalize()

    def check_normalize_channel(n: int = 128):
        ed = EcgDataset(normalize=[('norm', 3), ('std', 1)])
        ed_n = EcgDataset(normalize='none')
        n_sig, (n_ch, l) = len(ed), ed.dataset.shape[1:]
        for i in range(n_ch):
            idxs_sig = np.sort(np.random.choice(n_sig, size=n, replace=False))
            sigs = ed[idxs_sig][:, i, :]
            sigs_ori = ed_n[idxs_sig][:, i, :]

            oor = (0 > sigs) | (sigs > 1)
            oor = dict(signal=np.any(oor, axis=-1).sum() / n, sample=oor.sum() / oor.size)
            ic(i, oor)

            plt.figure(figsize=(18, 6))
            plt.hlines([-1, 0, 1], xmin=0, xmax=sigs.shape[-1], lw=0.25)
            ecg_util.plot_1d(
                sigs_ori, label='Signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11, c='r'), show=False
            )
            ecg_util.plot_1d(
                sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11, c='m'), show=False
            )
            plt.title(f'Normalize sanity check for channel {i+1}')
            plt.show()
    # check_normalize_channel(n=32)

    def check_pad(n: int = 128):
        patch_size = 64
        pad = TimeEndPad(patch_size, pad_kwargs=dict(mode='constant', constant_values=0))
        ic(pad)
        ed = EcgDataset(normalize=('std', 1), transform=pad)
        sigs = sample_sigs(ed, n)

        plt.figure(figsize=(18, 6))
        ecg_util.plot_1d(sigs, label='Normalized signal', new_fig=False, plot_kwargs=dict(lw=0.1, ms=0.11))
        plt.show()
    check_pad()

