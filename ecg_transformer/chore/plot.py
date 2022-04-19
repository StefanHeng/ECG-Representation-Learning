import math
from typing import Dict
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ecg_transformer.util import *


class PtbxlAucVisualizer:
    def __init__(self, code2auc: Dict[str, float]):
        self.code2auc = {c: round(v*100, 1) for c, v in code2auc.items()}

    def grouped_plot(
            self, save: bool = True, title: str = None, color_by: str = 'class',
            color_palette: str = None
    ):
        """
        Plots per-class AUROC, grouped by semantic meaning of the labels
        """
        ca.check_mismatch('AUC Bin Color Type', color_by, ['class', 'score'])
        dnm = 'PTB-XL'
        d_diag = config(f'datasets.{dnm}.code.diagnostic-class2sub-class2code')
        dnm = 'PTB-XL'
        d_codes = config(f'datasets.{dnm}.code')
        form_codes, rhythm_codes = d_codes['form-codes'], d_codes['rhythm-codes']

        fig = plt.figure(figsize=(16, 12), constrained_layout=False)
        n_row, n_col = 4, 24+2
        gs = GridSpec(n_row, n_col, figure=fig)  # Hard-coded based on PTB cateogory taxonomy
        axes_diag = dict()
        sep1, sep2 = 2, 2  # Separate on the longer row also, so that labels don't write over
        ax_cbar = fig.add_subplot(gs[0, :1])
        axes_diag['NORM'] = fig.add_subplot(gs[0, 1+sep1:1+sep1+1+1])  # just 1 code, give it a larger width
        axes_diag['HYP'] = fig.add_subplot(gs[0, (1+sep1+1+1)+sep1:(1+sep1+1+1)+sep1+5])
        axes_diag['MI'] = fig.add_subplot(gs[0, (1+sep1+1+1+sep1+5)+sep1:])
        axes_diag['CD'] = fig.add_subplot(gs[1, 0:11])
        axes_diag['STTC'] = fig.add_subplot(gs[1, 11+sep2:])
        n_form, n_rhythm = len(form_codes), len(rhythm_codes)
        idx_strt_form, idx_strt_rhythm = n_col//2 - math.ceil((n_form+1)/2), n_col//2 - math.ceil((n_rhythm+1)/2)
        ax_form = fig.add_subplot(gs[2, idx_strt_form:idx_strt_form+n_form])
        ax_rhythm = fig.add_subplot(gs[3, idx_strt_rhythm:idx_strt_rhythm+n_rhythm])

        sub_classes = ['NORM', 'HYP', 'MI', 'CD', 'STTC']  # Follow the same order, for color assignment
        codes_all = sum([codes for sub_cls in sub_classes for codes in d_diag[sub_cls].values()], start=[])
        codes_all += form_codes + rhythm_codes
        aucs_all = [self.code2auc[c] for c in codes_all]
        n_code = len(codes_all)
        if color_by == 'class':
            color_gap = 4  # consecutive coloring with gap
            cs = sns.color_palette(palette=color_palette or 'husl', n_colors=n_code + color_gap * len(d_diag))
            ax_cbar.set_visible(False)
        else:
            pnm = color_palette or 'Spectral_r'
            color_gap, cs = 0, vals2colors(aucs_all, color_palette=pnm)
            set_color_bar(aucs_all, ax_cbar, color_palette=pnm)
        clr_count = 0

        group2plot_meta = OrderedDict()
        for cls in sub_classes:
            sub_cls2code, ax = d_diag[cls], axes_diag[cls]
            codes = sum(sub_cls2code.values(), start=[])
            codes_print = [c.replace('/', '/\n') for c in codes]  # so that fits in plot
            group_desc = config(f'datasets.{dnm}.code.diagnostic-sub-class2description.{cls}')
            group_desc = f'Diagnostic: {group_desc} ({cls})'
            group2plot_meta[cls] = dict(ax=axes_diag[cls], codes=codes, codes_print=codes_print, group_desc=group_desc)
        for group, ax, codes in zip(['form', 'rhythm'], [ax_form, ax_rhythm], [form_codes, rhythm_codes]):
            codes_print = [c.replace('/', '/\n') for c in codes]
            group2plot_meta[group] = dict(ax=ax, codes=codes, codes_print=codes_print, group_desc=group.capitalize())

        for group, meta in group2plot_meta.items():
            ax, codes, codes_print, group_desc = meta['ax'], meta['codes'], meta['codes_print'], meta['group_desc']
            cs_ = cs[clr_count:clr_count + len(codes)]
            clr_count += len(codes) + color_gap
            barplot(x=codes_print, y=[self.code2auc[c] for c in codes], ax=ax, palette=cs_, width=0.375, ylabel=None)
            ax.set_xlabel(group_desc, style='italic')

        aucs_all = np.asarray(aucs_all)
        ma, mi = np.max(aucs_all), np.min(aucs_all)
        ma, mi = min(round(ma, -1)+10+5, 100+5), max(round(mi, -1)-10, 0)  # reserve space for values on top of each bar
        for ax in axes_diag.values():
            ax.set_ylim([mi, ma])

        fig.supylabel('Binary Classification AUROC (%)')
        fig.supxlabel('SCP code')
        title = title or 'PTB-XL per-code AUROC bar plot by group'
        fig.suptitle(title)
        fig.tight_layout()
        save_fig(title) if save else plt.show()

    def sorted_plot(self, save: bool = True, title: str = None):
        codes = sorted(self.code2auc, key=self.code2auc.get, reverse=True)
        dnm = 'PTB-XL'
        code2meta = config(f'datasets.{dnm}.code.codes')

        def aspect2aspect_print(aspect):
            return rf'$\it{{{aspect.capitalize()}}}$'

        def code2aspect(code):
            return ', '.join(aspect2aspect_print(aspect) for aspect in code2meta[code]['aspects'])
        code2desc = config(f'datasets.{dnm}.code.code2description')
        codes_print = [f'{code2aspect(c)}: {code2desc[c].capitalize()}' for c in codes]

        plt.figure(figsize=(14, 14))
        y = [self.code2auc[c] for c in codes]
        barplot(x=codes_print, y=y, palette='mako_r', orient='h', xlabel='SCP code', ylabel='AUROC (%)')

        title = title or 'PTB-XL per-code AUROC sorted bar plot'
        plt.title(title)
        save_fig(title) if save else plt.show()


if __name__ == '__main__':
    import os
    import json

    from icecream import ic

    from ecg_transformer.models import get_eval_path

    model_dir_nm = 'EcgVit, base'
    eval_path = os.path.join(get_eval_path(), model_dir_nm, 'evaluation, 2022-04-18_12-47-03.json')
    with open(eval_path, 'r') as f:
        eval_res = json.load(f)
    split = 'eval'
    c2a = get(eval_res, f'{split}.eval/per_class_auc')
    pav = PtbxlAucVisualizer(c2a)

    model_desc = 'EcgVit-base with Vanilla training'

    def plot_grouped():
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
                pav.grouped_plot(title=c, save=True, color_by=color_by, color_palette=c)
        # pick_cmap()

        title = f'PTB-XL per-code AUROC bar plot by group on {model_desc}'
        # cmap = 'Spectral_r'
        cmap = 'mako'
        # save = True
        save = False
        pav.grouped_plot(title=title, save=save, color_by=color_by, color_palette=cmap)
    # plot_grouped()

    def plot_sorted():
        title = f'PTB-XL per-code AUROC sorted bar plot on {model_desc}'
        save = False
        # save = True
        pav.sorted_plot(title=title, save=save)
    plot_sorted()
