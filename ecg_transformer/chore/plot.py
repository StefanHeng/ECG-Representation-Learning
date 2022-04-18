from typing import Dict, Iterable

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


def vals2colors(vals: Iterable[float], color_palette: str = 'Spectral_r'):
    vals = np.asarray(vals)
    cmap = sns.color_palette(color_palette, as_cmap=True)
    mi, ma = np.min(vals), np.max(vals)
    norm = (vals - mi) / (ma - mi)
    return cmap(norm)


def set_color_bar(vals, ax, color_palette: str = 'Spectral_r'):
    vals = np.asarray(vals)
    norm = plt.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    sm = plt.cm.ScalarMappable(cmap=color_palette, norm=norm)
    sm.set_array([])
    plt.grid(False)
    plt.colorbar(sm, cax=ax)


def plot_ptbxl_auroc(
        code2auc: Dict[str, float], save: bool = True, title: str = None, color_by: str = 'class',
        color_palette: str = None
):
    """
    Plots per-class AUROC, grouped by semantic meaning of the labels
    """
    code2auc = {c: round(v*100, 1) for c, v in code2auc.items()}
    ca.check_mismatch('AUC Bin Color Type', color_by, ['class', 'score'])
    dnm = 'PTB-XL'
    d_diag = config(f'datasets.{dnm}.code.diagnostic-class2sub-class2code')

    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    gs = GridSpec(2, 24+2, figure=fig)  # Hard-coded based on PTB cateogory taxonomy
    axes = dict()
    sep1, sep2 = 2, 2  # Separate on the longer row also, so that labels don't write over
    ax_cbar = fig.add_subplot(gs[0, :1])
    axes['NORM'] = fig.add_subplot(gs[0, 1+sep1:1+sep1+1+1])  # just 1 code, give it a larger width
    axes['HYP'] = fig.add_subplot(gs[0, (1+sep1+1+1)+sep1:(1+sep1+1+1)+sep1+5])
    axes['MI'] = fig.add_subplot(gs[0, (1+sep1+1+1+sep1+5)+sep1:])
    axes['CD'] = fig.add_subplot(gs[1, 0:11])
    axes['STTC'] = fig.add_subplot(gs[1, 11+sep2:])

    sub_classes = ['NORM', 'HYP', 'MI', 'CD', 'STTC']  # Follow the same order, for color assignment
    codes_all = sum([codes for sub_cls in sub_classes for codes in d_diag[sub_cls].values()], start=[])
    aucs_all = [code2auc[c] for c in codes_all]
    n_code = len(codes_all)
    if color_by == 'class':
        color_gap = 4  # consecutive coloring with gap
        cs = sns.color_palette(palette=color_palette or 'husl', n_colors=n_code + color_gap * len(d_diag))
        ax_cbar.set_visible(False)
    else:
        pnm = color_palette or 'Spectral_r'
        color_gap, cs = 0, vals2colors(aucs_all, color_palette=pnm)
        set_color_bar(aucs_all, ax_cbar, color_palette=pnm)
        ax_cbar.set_xlabel('colorbar')
    clr_count = 0

    for cls in sub_classes:
        sub_cls2code, ax = d_diag[cls], axes[cls]
        codes = sum(sub_cls2code.values(), start=[])
        codes_print = [c.replace('/', '/\n') for c in codes]  # so that fits in plot
        df = pd.DataFrame([{'code': c_p, 'auc': code2auc[code]} for code, c_p in zip(codes, codes_print)])
        cat = CategoricalDtype(categories=codes_print, ordered=True)  # Enforce ordering in plot
        df.code = df.code.astype(cat, copy=False)

        cs_ = cs[clr_count:clr_count+len(codes)]
        clr_count += len(codes) + color_gap
        sns.barplot(data=df, x='code', y='auc', palette=cs_, ax=ax)
        ax.bar_label(ax.containers[0])
        change_bar_width(ax, 0.25)
        cls_desc = config(f'datasets.{dnm}.code.diagnostic-sub-class2description.{cls}')
        ax.set_xlabel(f'{cls_desc} ({cls})', style='italic')
        ax.set_ylabel(None)

    aucs_all = np.asarray(aucs_all)
    ma, mi = np.max(aucs_all), np.min(aucs_all)
    ma, mi = min(round(ma, -1)+10+5, 100+5), max(round(mi, -1)-10, 0)  # for the values on each bar
    for ax in axes.values():
        ax.set_ylim([mi, ma])

    fig.supylabel('Binary Classification AUROC (%)')
    fig.supxlabel('SCP code')
    title = title or 'PTB-XL Diagnostic Classes AUROC plot'
    fig.suptitle(title)
    fig.tight_layout()
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
        model_dir_nm = 'EcgVit, base'
        eval_path = os.path.join(get_eval_path(), model_dir_nm, 'evaluation, 2022-04-18_12-47-03.json')
        with open(eval_path, 'r') as f:
            eval_res = json.load(f)
        split = 'eval'
        code2auc = get(eval_res, f'{split}.eval/per_class_auc')
        # ic(code2auc)

        # color_by = 'class'
        color_by = 'score'

        def pick_cmap():
            cmaps = [
                'mako',
                'CMRmap',
                'RdYlBu',
                'Spectral',
                'bone',
                'gnuplot',
                'gnuplot2',
                'icefire',
                'rainbow',
                'rocket',
                'terrain',
                'twilight',
                'twilight_shifted'
            ]
            for c in cmaps:
                plot_ptbxl_auroc(code2auc, title=c, save=True, color_by=color_by, color_palette=c)
        # pick_cmap()

        model_desc = 'EcgVit-base with Vanilla training'
        title = f'PTB-XL Diagnostic Classes AUROC plot on {model_desc}'
        # cmap = 'Spectral_r'
        cmap = 'mako'
        plot_ptbxl_auroc(code2auc, title=title, save=False, color_by=color_by, color_palette=cmap)
    plot_eval()
