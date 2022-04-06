import os
import re
import sys
import json
import pathlib
import logging
import datetime
import itertools
import concurrent.futures
from typing import List, Dict, Tuple
from typing import Union, Any, Callable, Iterable, TypeVar
from pygments import highlight, lexers, formatters
from functools import reduce
from collections import OrderedDict

import sty
import colorama
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .data_path import PATH_BASE, DIR_PROJ, PKG_NM, DIR_DSET


pd.set_option('expand_frame_repr', False)
pd.set_option('display.precision', 2)
pd.set_option('max_colwidth', 40)


plt.rcParams['figure.constrained_layout.use'] = True
sns.set_style('darkgrid')
sns.set_context(rc={'grid.linewidth': 0.5})

LN_KWARGS = dict(marker='o', ms=0.3, lw=0.25)  # matplotlib line plot default args


nan = float('nan')


def get(dic, ks):
    """
    :param dic: Potentially multi-level dictionary
    :param ks: Potentially `.`-separated keys
    """
    ks = ks.split('.')
    return reduce(lambda acc, elm: acc[elm], ks, dic)


def set_(dic, ks, val):
    ks = ks.split('.')
    node = reduce(lambda acc, elm: acc[elm], ks[:-1], dic)
    node[ks[-1]] = val


def keys(dic, prefix=''):
    """
    :return: Generator for all potentially-nested keys
    """
    def _full(k_):
        return k_ if prefix == '' else f'{prefix}.{k_}'
    for k, v in dic.items():
        if isinstance(v, dict):
            for k__ in keys(v, prefix=_full(k)):
                yield k__
        else:
            yield _full(k)


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', 'config.json'), 'r') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def get_processed_path():
    return os.path.join(PATH_BASE, DIR_DSET, config('datasets.my.dir_nm'))


T = TypeVar('T')
K = TypeVar('K')


def join_its(its: Iterable[Iterable[T]]) -> Iterable[T]:
    out = itertools.chain()
    for it in its:
        out = itertools.chain(out, it)
    return out


def conc_map(fn: Callable[[T], K], it: Iterable[T]) -> Iterable[K]:
    """
    Wrapper for `concurrent.futures.map`

    :param fn: A function
    :param it: A list of elements
    :return: Iterator of `lst` elements mapped by `fn` with concurrency
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.map(fn, it)


def batched_conc_map(
        fn: Callable[[Tuple[List[T], int, int]], K], lst: List[T], n_worker: int = os.cpu_count()
) -> List[K]:
    """
    Batched concurrent mapping, map elements in list in batches

    :param fn: A map function that operates on a batch/subset of `lst` elements,
        given inclusive begin & exclusive end indices
    :param lst: A list of elements to map
    :param n_worker: Number of concurrent workers
    """
    n: int = len(lst)
    if n_worker > 1 and n > n_worker * 4:  # factor of 4 is arbitrary, otherwise not worse the overhead
        preprocess_batch = round(n / n_worker / 2)
        strts: List[int] = list(range(0, n, preprocess_batch))
        ends: List[int] = strts[1:] + [n]  # inclusive begin, exclusive end
        lst_out = []
        for lst_ in conc_map(lambda args_: fn(*args_), [(lst, s, e) for s, e in zip(strts, ends)]):  # Expand the args
            lst_out.extend(lst_)
        return lst_out
    else:
        args = lst, 0, n
        return fn(*args)


def now(as_str=True, for_path=False) -> Union[datetime.datetime, str]:
    """
    # Considering file output path
    :param as_str: If true, returns string; otherwise, returns datetime object
    :param for_path: If true, the string returned is formatted as intended for file system path
    """
    d = datetime.datetime.now()
    fmt = '%Y-%m-%d_%H-%M-%S' if for_path else '%Y-%m-%d %H:%M:%S'
    return d.strftime(fmt) if as_str else d


def sizeof_fmt(num, suffix='B'):
    """ Converts byte size to human-readable format """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def fmt_dt(secs: Union[int, float, datetime.timedelta]):
    if isinstance(secs, datetime.timedelta):
        secs = secs.seconds + (secs.microseconds/1e6)
    if secs >= 86400:
        d = secs // 86400  # // floor division
        return f'{round(d)}d{fmt_dt(secs-d*86400)}'
    elif secs >= 3600:
        h = secs // 3600
        return f'{round(h)}h{fmt_dt(secs-h*3600)}'
    elif secs >= 60:
        m = secs // 60
        return f'{round(m)}m{fmt_dt(secs-m*60)}'
    else:
        return f'{round(secs)}s'


def sig_d(flt: float, n: int = 1):
    """
    :return: first n-th significant digit of `sig_d`
    """
    return float('{:.{p}g}'.format(flt, p=n))


def log(s, c: str = 'log', c_time='green', as_str=False, pad: int = None):
    """
    Prints `s` to console with color `c`
    """
    if not hasattr(log, 'reset'):
        log.reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
    if not hasattr(log, 'd'):
        log.d = dict(
            log='',
            warn=colorama.Fore.YELLOW,
            error=colorama.Fore.RED,
            err=colorama.Fore.RED,
            success=colorama.Fore.GREEN,
            suc=colorama.Fore.GREEN,
            info=colorama.Fore.BLUE,
            i=colorama.Fore.BLUE,
            w=colorama.Fore.RED,

            y=colorama.Fore.YELLOW,
            yellow=colorama.Fore.YELLOW,
            red=colorama.Fore.RED,
            r=colorama.Fore.RED,
            green=colorama.Fore.GREEN,
            g=colorama.Fore.GREEN,
            blue=colorama.Fore.BLUE,
            b=colorama.Fore.BLUE,

            m=colorama.Fore.MAGENTA
        )
    if c in log.d:
        c = log.d[c]
    if as_str:
        return f'{c}{s:>{pad}}{log.reset}' if pad is not None else f'{c}{s}{log.reset}'
    else:
        print(f'{c}{log(now(), c=c_time, as_str=True)}| {s}{log.reset}')


def log_s(s, c):
    return log(s, c=c, as_str=True)


def logi(s):
    """
    Syntactic sugar for logging `info` as string
    """
    return log_s(s, c='i')


def is_float(x: Any, no_int=False, no_sci=False) -> bool:
    try:
        is_sci = isinstance(x, str) and 'e' in x.lower()
        f = float(x)
        is_int = f.is_integer()
        out = True
        if no_int:
            out = out & (not is_int)
        if no_sci:
            out = out & (not is_sci)
        return out
    except ValueError:
        return False


def log_dict(d: Dict, with_color=True, pad_float: int = 5) -> str:
    """
    Syntactic sugar for logging dict with coloring for console output
    """
    def _log_val(v):
        if isinstance(v, dict):
            return log_dict(v, with_color=with_color)
        else:
            if is_float(v):  # Pad only normal, expected floats, intended for metric logging
                if is_float(v, no_int=True, no_sci=True):
                    v = float(v)
                    return log(v, c='i', as_str=True, pad=pad_float) if with_color else f'{v:>{pad_float}}'
                else:
                    return logi(v) if with_color else v
            else:
                return logi(v) if with_color else v
    if d is None:
        d = dict()
    pairs = (f'{k}: {_log_val(v)}' for k, v in d.items())
    pref = log_s('{', c='m') if with_color else '{'
    post = log_s('}', c='m') if with_color else '}'
    return pref + ', '.join(pairs) + post


def log_dict_id(d: Dict) -> str:
    return json.dumps(d, indent=4)


def log_dict_pg(d: Dict) -> str:
    return highlight(log_dict_id(d), lexers.JsonLexer(), formatters.TerminalFormatter())


def log_dict_nc(d: Dict, **kwargs) -> str:
    return log_dict(d, with_color=False, **kwargs)


def hex2rgb(hx: str) -> Union[Tuple[int], Tuple[float]]:
    # Modified from https://stackoverflow.com/a/62083599/10732321
    if not hasattr(hex2rgb, 'regex'):
        hex2rgb.regex = re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$')
    m = hex2rgb.regex.match(hx)
    assert m is not None
    if len(hx) <= 4:
        return tuple(int(hx[i]*2, 16) for i in range(1, 4))
    else:
        return tuple(int(hx[i:i+2], 16) for i in range(1, 7, 2))


class MyTheme:
    """
    Theme based on `sty` and `Atom OneDark`
    """
    COLORS = OrderedDict([
        ('yellow', 'E5C07B'),
        ('green', '00BA8E'),
        ('blue', '61AFEF'),
        ('cyan', '2AA198'),
        ('red', 'E06C75'),
        ('purple', 'C678DD')
    ])
    yellow, green, blue, cyan, red, purple = (
        hex2rgb(f'#{h}') for h in ['E5C07B', '00BA8E', '61AFEF', '2AA198', 'E06C75', 'C678DD']
    )

    @staticmethod
    def set_color_type(t: str):
        """
        Sets the class attribute accordingly

        :param t: One of ['rgb`, `sty`]
            If `rgb`: 3-tuple of rgb values
            If `sty`: String for terminal styling prefix
        """
        for color, hex_ in MyTheme.COLORS.items():
            val = hex2rgb(f'#{hex_}')  # For `rgb`
            if t == 'sty':
                setattr(sty.fg, color, sty.Style(sty.RgbFg(*val)))
                val = getattr(sty.fg, color)
            setattr(MyTheme, color, val)


class MyFormatter(logging.Formatter):
    """
    Modified from https://stackoverflow.com/a/56944256/10732321

    Default styling: Time in green, metadata indicates severity, plain log message
    """
    RESET = sty.rs.fg + sty.rs.bg + sty.rs.ef

    MyTheme.set_color_type('sty')
    yellow, green, blue, cyan, red, purple = (
        MyTheme.yellow, MyTheme.green, MyTheme.blue, MyTheme.cyan, MyTheme.red, MyTheme.purple
    )

    KW_TIME = '%(asctime)s'
    KW_MSG = '%(message)s'
    KW_LINENO = '%(lineno)d'
    KW_FNM = '%(filename)s'
    KW_FUNCNM = '%(funcName)s'
    KW_NAME = '%(name)s'

    DEBUG = INFO = BASE = RESET
    WARN, ERR, CRIT = yellow, red, purple
    CRIT += sty.Style(sty.ef.bold)

    LVL_MAP = {  # level => (abbreviation, style)
        logging.DEBUG: ('DBG', DEBUG),
        logging.INFO: ('INFO', INFO),
        logging.WARNING: ('WARN', WARN),
        logging.ERROR: ('ERR', ERR),
        logging.CRITICAL: ('CRIT', CRIT)
    }

    def __init__(self, with_color=True, color_time=green):
        super().__init__()
        self.with_color = with_color

        sty_kw, reset = MyFormatter.blue, MyFormatter.RESET
        color_time = f'{color_time}{MyFormatter.KW_TIME}{sty_kw}| {reset}'

        def args2fmt(args_):
            if self.with_color:
                return color_time + self.fmt_meta(*args_) + f'{sty_kw} - {reset}{MyFormatter.KW_MSG}' + reset
            else:
                return f'{MyFormatter.KW_TIME}| {self.fmt_meta(*args_)} - {MyFormatter.KW_MSG}'

        self.formats = {level: args2fmt(args) for level, args in MyFormatter.LVL_MAP.items()}
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def fmt_meta(self, meta_abv, meta_style=None):
        if self.with_color:
            return f'{MyFormatter.purple}[{MyFormatter.KW_NAME}]' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNCNM}' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}' \
               f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}' \
               f'{MyFormatter.blue}, {meta_style}{meta_abv}{MyFormatter.RESET}'
        else:
            return f'[{MyFormatter.KW_NAME}] {MyFormatter.KW_FUNCNM}::{MyFormatter.KW_FNM}' \
                   f':{MyFormatter.KW_LINENO}, {meta_abv}'

    def format(self, entry):
        return self.formatter[entry.levelno].format(entry)


def get_logger(name: str, typ: str = 'stdout', file_path: str = None) -> logging.Logger:
    """
    :param name: Name of the logger
    :param typ: Logger type, one of [`stdout`, `file-write`]
    :param file_path: File path for file-write logging
    """
    assert typ in ['stdout', 'file-write']
    logger = logging.getLogger(f'{name} file write' if typ == 'file-write' else name)
    logger.handlers = []  # A crude way to remove prior handlers, ensure only 1 handler per logger
    logger.setLevel(logging.DEBUG)
    if typ == 'stdout':
        handler = logging.StreamHandler(stream=sys.stdout)  # stdout for my own coloring
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        handler = logging.FileHandler(file_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(MyFormatter(with_color=typ == 'stdout'))
    logger.addHandler(handler)
    return logger


def np_index(arr, idx):
    return np.where(arr == idx)[0][0]


def clipper(low, high):
    """
    :return: A clipping function for range [low, high]
    """
    return lambda x: max(min(x, high), low)


def stem(path_, ext=False):
    """
    :param path_: A potentially full path to a file
    :param ext: If True, file extensions is preserved
    :return: The file name, without parent directories
    """
    return os.path.basename(path_) if ext else pathlib.Path(path_).stem


def remove_1st_occurrence(str_, str_sub):
    idx = str_.find(str_sub)
    return str_[:idx] + str_[idx+len(str_sub):]


def save_fig(title, save=True):
    if save:
        fnm = f'{title}.png'
        plt.savefig(os.path.join(PATH_BASE, DIR_PROJ, 'plot', fnm), dpi=300)


if __name__ == '__main__':
    from icecream import ic
    np.random.seed(config('random_seed'))

    ic(config('datasets.BIH_MVED'))

    # ic(remove_1st_occurrence('E00002.mat 12 500 5000 05-May-2020 14:50:55', '.mat'))
