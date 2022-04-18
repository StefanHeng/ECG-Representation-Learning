import os
from typing import Dict

import pandas as pd
from pandas.api.types import CategoricalDtype
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ecg_transformer.util import *
from ecg_transformer.models.ecg_vit import EcgVitConfig, EcgVit
from ecg_transformer.preprocess import transform, get_ptbxl_splits
from ecg_transformer.models.train import MyTrainer


def load_trained(model_key: str = 'ecg-vit-base'):
    model = EcgVit(config=EcgVitConfig.from_defined(model_key))

    fnm = 'model - model={nm=EcgVit, in-sp=12x2560, #p=40, #l=12, #h=12}, ' \
          'n=17441, a=0.0003, dc=0.01, bsz=256, n_ep=32, ep8.pt'
    checkpoint_path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, '2022-04-15_23-48-47', fnm)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)  # Need the pl wrapper cos that's how the model is saved
    model.eval()
    return model


def evaluate():
    model = load_trained()
    ptbxl_type = 'original'
    n_sample = None
    # n_sample = 64

    dnm = 'PTB-XL'
    pad = transform.TimeEndPad(model.config.patch_size, pad_kwargs=dict(mode='constant', constant_values=0))
    stats = config(f'datasets.{dnm}.train-stats.{ptbxl_type}')
    dset_args = dict(type=ptbxl_type, normalize=stats, transform=pad, return_type='pt')
    vl = get_ptbxl_splits(n_sample=n_sample, dataset_args=dset_args).eval
    trainer = MyTrainer(model=model, args=dict(eval_batch_size=16))
    d_out = trainer.evaluate(vl)
    # ic(d_out)
    model_desc = 'EcgVit-base with Vanilla training'
    title = f'PTB-XL Diagnostic Classes AUROC plot on {model_desc}'
    plot_ptbxl_auroc(d_out['eval/per_class_auc'], title=title, save=False)


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


def plot_ptbxl_auroc(code2auc: Dict[str, float], save: bool = True, title: str = None):
    """
    Plots per-class AUROC, grouped by semantic meaning of the labels
    """
    dnm = 'PTB-XL'
    d_diag = config(f'datasets.{dnm}.code.diagnostic-class2sub-class2code')
    # for cls, d_cls in d_diag.items():
    #     ic(cls, len(sum(d_cls.values(), start=[])))

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

    # for i, ax in enumerate(fig.axes):
    #     ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
    #     ax.tick_params(labelbottom=False, labelleft=False)

    n_code = sum(sum(len(codes) for codes in sub_cls2code) for sub_cls2code in d_diag.values())
    color_gap = 4
    cs = sns.color_palette(palette='husl', n_colors=n_code + color_gap * len(d_diag))  # consecutive coloring with gap
    clr_count = 0

    for cls in sub_classes:
        sub_cls2code, ax = d_diag[cls], axes[cls]
        # for sub_cls, code in sub_cls2code.items():
        #     ic(sub_cls, code)
        # ic([{'sub_class': sub_cls, 'auc': code2auc[code]} for sub_cls, code in sub_cls2code.items()])
        codes = sum(sub_cls2code.values(), start=[])
        codes_print = [c.replace('/', '/\n') for c in codes]  # so that fits in plot
        df = pd.DataFrame([{'code': c_p, 'auc': code2auc[code] * 100} for code, c_p in zip(codes, codes_print)])
        cat = CategoricalDtype(categories=codes_print, ordered=True)  # Enforce ordering in plot
        df.code = df.code.astype(cat, copy=False)

        cs_ = cs[clr_count:clr_count+len(codes)]
        clr_count += len(codes) + color_gap
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
    from icecream import ic

    def check_load():
        model = load_trained()
        ic(type(model), get_model_num_trainable_parameter(model))
    # check_load()

    evaluate()
