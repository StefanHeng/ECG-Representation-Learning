from typing import List, Dict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ecg_transformer.util import *


def change_bar_width(ax, new_value):
    """
    Modifies the bar width of a matplotlib bar plot

    Credit: https://stackoverflow.com/a/44542112/10732321
    """
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)


def vals2colors(vals: List[float], palette_name: str):
    vals = np.asarray(vals)
    normalized = (vals - min(vals)) / (max(vals) - min(vals))
    # convert to indices
    indices = np.round(normalized * (len(vals) - 1)).astype(np.int32)
    ic(vals, indices)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, n_colors=len(vals))
    return np.array(palette).take(indices, axis=0)


def plot_ptbxl_auroc(code2auc: Dict[str, float], save: bool = True, title: str = None, color_by: str = 'class'):
    """
    Plots per-class AUROC, grouped by semantic meaning of the labels
    """
    ca.check_mismatch('AUC Bin Color Type', color_by, ['class', 'score'])
    dnm = 'PTB-XL'
    d_diag = config(f'datasets.{dnm}.code.diagnostic-class2sub-class2code')

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 24+2, figure=fig)  # Hard-coded based on PTB cateogory taxonomy
    axes = dict()
    sep1, sep2 = 3, 2  # Separate on the longer row also, so that labels don't write over
    axes['NORM'] = fig.add_subplot(gs[0, 0:1+1])  # just 1 signal, give it a larger width
    axes['HYP'] = fig.add_subplot(gs[0, 2+sep1:2+sep1+5])
    axes['MI'] = fig.add_subplot(gs[0, (1+sep1+5)-1+sep1:(1+sep1+5)-1+sep1+14])
    axes['CD'] = fig.add_subplot(gs[1, 0:11])
    axes['STTC'] = fig.add_subplot(gs[1, 11+sep2:])
    sub_classes = ['NORM', 'HYP', 'MI', 'CD', 'STTC']  # Follow the same order, for color assignment

    # codes_all = sum([codes for sub_cls2code in d_diag.values() for codes in sub_cls2code.values()], start=[])
    codes_all = sum([codes for sub_cls in sub_classes for codes in d_diag[sub_cls].values()], start=[])
    # ic(codes_all, len(codes_all))
    # exit(1)
    # n_code = sum(sum(len(codes) for codes in sub_cls2code) for sub_cls2code in d_diag.values())
    n_code = len(codes_all)
    if color_by == 'class':
        color_gap = 4  # consecutive coloring with gap
        cs = sns.color_palette(palette='husl', n_colors=n_code + color_gap * len(d_diag))
    else:
        color_gap, cs = 0, vals2colors([code2auc[c] for c in codes_all], palette_name='Spectral_r')
        cm = sns.color_palette('Spectral_r', as_cmap=True)
        cs = [cm(code2auc[c]) for c in codes_all]  # already in range [0, 1]
    # colors_from_values(y, "YlOrRd")
    clr_count = 0

    for cls in sub_classes:
        sub_cls2code, ax = d_diag[cls], axes[cls]
        codes = sum(sub_cls2code.values(), start=[])
        codes_print = [c.replace('/', '/\n') for c in codes]  # so that fits in plot
        df = pd.DataFrame([{'code': c_p, 'auc': code2auc[code] * 100} for code, c_p in zip(codes, codes_print)])
        cat = CategoricalDtype(categories=codes_print, ordered=True)  # Enforce ordering in plot
        df.code = df.code.astype(cat, copy=False)

        cs_ = cs[clr_count:clr_count+len(codes)]
        clr_count += len(codes) + color_gap
        # cs_ = vals2colors([code2auc[c] for c in codes], palette_name='Spectral_r')
        sns.barplot(data=df, x='code', y='auc', palette=cs_, ax=ax)
        change_bar_width(ax, 0.25)
        cls_desc = config(f'datasets.{dnm}.code.diagnostic-sub-class2description.{cls}')
        ax.set_xlabel(f'{cls_desc} ({cls})', style='italic')
        ax.set_ylabel(None)

    fig.supylabel('Binary Classification AUROC (%)')
    fig.supxlabel('SCP code')
    title = title or 'PTB-XL Diagnostic Classes AUROC plot'
    fig.suptitle(title)
    if save:
        save_fig(title)
    else:
        plt.show()


if __name__ == '__main__':
    import os
    import json

    from icecream import ic

    from ecg_transformer.models import get_eval_path

    def plot_eval():
        model_desc = 'EcgVit-base with Vanilla training'
        title = f'PTB-XL Diagnostic Classes AUROC plot on {model_desc}'

        model_dir_nm = 'EcgVit, base'
        eval_path = os.path.join(get_eval_path(), model_dir_nm, 'evaluation, 2022-04-18_12-47-03.json')
        with open(eval_path, 'r') as f:
            eval_res = json.load(f)
        split = 'eval'
        code2auc = get(eval_res, f'{split}.eval/per_class_auc')
        plot_ptbxl_auroc(code2auc, title=title, save=False, color_by='score')
    plot_eval()
