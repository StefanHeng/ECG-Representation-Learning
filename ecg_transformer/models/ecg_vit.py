"""
Vision Transformer adapted to 1D ECG signals

Intended fpr vanilla, supervised training
"""
import os
import re

import torch
from torch import nn
from transformers import PretrainedConfig
from vit_pytorch import ViT
from vit_pytorch.recorder import Recorder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ecg_transformer.util import *
from ecg_transformer.util.models import ModelOutput
import ecg_transformer.util.ecg as ecg_util
from ecg_transformer.preprocess import get_ptbxl_dataset
from ecg_transformer.chore import barplot


class EcgVitConfig(PretrainedConfig):
    pattern_model_name = re.compile(rf'^(?P<name>\S+)-(?P<size>\S+)$')

    def __init__(
            self,
            max_signal_length: int = 2560,
            patch_size: int = 64,
            num_channels: int = 12,
            hidden_size: int = 512,  # Default parameters are 2/3 of ViT base model sizes
            num_hidden_layers: int = 8,
            num_attention_heads: int = 8,
            intermediate_size: int = 2048,
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            num_class: int = 71,  # specific to ECG supervised classification
            **kwargs
    ):
        self.max_signal_length = max_signal_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_class = num_class
        super().__init__(**kwargs)
        self.size = None

    @classmethod
    def from_defined(cls, model_name):
        """
        A few model sizes I defined
        """
        ca(model_name=model_name)
        conf = cls()
        m = cls.pattern_model_name.match(model_name)
        nm, size = m.group('name'), m.group('size')
        conf.size = size
        assert nm == 'ecg-vit'
        if size == 'debug':
            conf.hidden_size = 64
            conf.num_hidden_layers = 4
            conf.num_attention_heads = 4
            conf.intermediate_size = 256
        elif size == 'tiny':
            conf.hidden_size = 256
            conf.num_hidden_layers = 4
            conf.num_attention_heads = 4
            conf.intermediate_size = 1024
        elif size == 'small':
            conf.hidden_size = 512
            conf.num_hidden_layers = 8
            conf.num_attention_heads = 8
            conf.intermediate_size = 2048
        elif size == 'base':
            conf.hidden_size = 768
            conf.num_hidden_layers = 12
            conf.num_attention_heads = 12
            conf.intermediate_size = 3072
        elif size == 'large':
            conf.hidden_size = 1024
            conf.num_hidden_layers = 24
            conf.num_attention_heads = 16
            conf.intermediate_size = 4096
        return conf


class EcgVit(nn.Module):
    def __init__(self, num_class: int = 71, config=EcgVitConfig(), loss_reduction: str = 'mean'):
        super().__init__()
        hd_sz, n_head = config.hidden_size, config.num_attention_heads
        assert hd_sz % n_head == 0
        dim_head = hd_sz // n_head
        self.config = config
        _md_args = dict(
            image_size=(1, self.config.max_signal_length),  # height is 1
            patch_size=(1, self.config.patch_size),
            num_classes=num_class,
            dim=self.config.hidden_size,
            depth=self.config.num_hidden_layers,
            heads=self.config.num_attention_heads,
            mlp_dim=self.config.intermediate_size,
            pool='cls',
            channels=self.config.num_channels,
            dim_head=dim_head,
            dropout=self.config.hidden_dropout_prob,
            emb_dropout=self.config.attention_probs_dropout_prob
        )
        self.vit = ViT(**_md_args)
        self._loss_reduction = loss_reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=loss_reduction)  # TODO: more complex loss, e.g. weighting?
        self.loss_weight = None

        C, L = self.config.num_channels, self.config.max_signal_length
        cls_nm = self.__class__.__qualname__
        n_pch, n_l, n_h = L // self.config.patch_size, self.config.num_hidden_layers, self.config.num_attention_heads
        self.meta = {
            'name': cls_nm, 'input shape': f'{C} x {L}', '#patch': n_pch, '#layer': n_l, '#head': n_h
        }
        self.meta_str = log_dict_p({'nm': cls_nm, 'in-sp': f'{C}x{L}', '#p': n_pch, '#l': n_l, '#h': n_h})

    def to_str(self):
        return f'{self.__class__.__qualname__}, {self.config.size}'

    @property
    def loss_reduction(self):
        return self._loss_reduction

    @loss_reduction.setter
    def loss_reduction(self, r):
        self.loss_fn.reduction = self._loss_reduction = r

    def forward(self, sample_values: torch.FloatTensor, labels: torch.LongTensor = None):
        logits = self.vit(sample_values.unsqueeze(-2))   # Add dummy height dimension
        loss = None
        if labels is not None:
            if self.loss_weight:  # modify the loss function each call
                weight = torch.tensor(self.loss_weight, device=labels.device)
                # Map weights by each label
                self.loss_fn = nn.BCEWithLogitsLoss(weight=weight[labels.long()], reduction=self.loss_reduction)
            loss = self.loss_fn(input=logits, target=labels)
        return ModelOutput(loss=loss, logits=logits)


def load_trained(model_key: str = 'ecg-vit-base'):
    model = EcgVit(config=EcgVitConfig.from_defined(model_key))

    fnm = 'model - model={nm=EcgVit, in-sp=12x2560, #p=40, #l=12, #h=12}, ' \
          'n=17441, a=0.0003, dc=0.01, bsz=256, n_ep=32, ep8.pt'
    checkpoint_path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, '2022-04-15_23-48-47', fnm)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)  # Need the pl wrapper cos that's how the model is saved
    model.eval()
    return model


class EcgVitVisualizer:
    def __init__(self, model: EcgVit, palette_correct: str = 'YlGn', palette_incorrect: str = 'OrRd'):
        self.model = model
        self.model.eval()
        self.palette_correct, self.palette_incorrect = palette_correct, palette_incorrect

    def __call__(self, sample_values: torch.FloatTensor, labels: torch.LongTensor, save: bool = False):
        assert sample_values.ndim == 2 and sample_values.size(0) == 12, \
            f'Expect a single 12-lead signal but got {logi(sample_values.shape)}'
        L, patch_size = sample_values.size(-1), self.model.config.patch_size
        assert L % patch_size == 0, f'Signal sample length must be divisible by model patch size, ' \
                                    f'but got {log_dict(L=L, patch_size=patch_size)}'
        vit = Recorder(self.model.vit)  # can't use my forward pass cos return is different
        with torch.no_grad():
            logits, attn = vit(sample_values.unsqueeze(0).unsqueeze(-2))  # Add dummy batch & height dimension
            loss = nn.BCEWithLogitsLoss()(input=logits.squeeze(), target=labels).item()
        vit.eject()

        # inspired by https://epfml.github.io/attention-cnn/;
        # following the logic from https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
        attn = attn.squeeze(0).mean(dim=0)  # average over all heads; B x H x L x P x P => L x P x P
        attn += torch.eye(attn.size(1))  # reverse residual connections; TODO: why diagonals??
        attn /= attn.sum(dim=-1, keepdim=True)  # normalize all keys for each query

        attn_res = torch.empty_like(attn)
        attn_res[0] = attn[0]
        for i in range(1, attn.size(0)):  # start from the bottom, multiply out the attentions on higher layers
            attn_res[i] = attn[i] @ attn[i - 1]
        attn_res = attn_res[:, 0, 1:]  # attention scores from each `cls` query to every other patch token on each layer
        attn_res /= attn_res.max()  # normalize all scores, ready for visualization
        assert torch.all((0 <= attn_res) & (attn_res <= 1))  # sanity check
        sig, attn_res = sample_values.numpy(), attn_res.numpy()

        probs = torch.sigmoid(logits.squeeze())
        assert torch.all((0 <= probs) & (probs <= 1))  # sanity check
        top_n = min((probs > 0.6).sum(), 5)  # number of predictions to show
        idxs_top = torch.argsort(probs, descending=True)[:top_n]
        dnm = 'PTB-XL'
        id2code = config(f'datasets.{dnm}.code.id2code')
        code2id = config(f'datasets.{dnm}.code.code2id')
        str_lbs = [id2code[idx] for idx in torch.nonzero(labels).squeeze().tolist()]
        str_preds, confs = [id2code[idx] for idx in idxs_top], probs[idxs_top].tolist()
        pred_correct = [p in str_lbs for p in str_preds]
        for lb in str_lbs:  # Also ensure predictions for each correct label is shown
            if lb not in str_preds:
                str_preds.append(lb)
                confs.append(probs[code2id[lb]].item())
                pred_correct.append(False)

        fig = plt.figure()
        n_lb, n_pd = len(str_lbs), len(str_preds)
        n_col_lb, n_col_sig, row_sep, col_sep = 6, 32, 1, 1
        n_row, n_col = 2 * (n_lb+n_pd) + row_sep * 3, n_col_lb + n_col_sig + col_sep * 1
        gs = GridSpec(n_row, n_col, figure=fig)
        ax_lb = fig.add_subplot(gs[:n_lb, :n_col_lb])
        ax_pd = fig.add_subplot(gs[n_lb+row_sep:n_lb+row_sep+n_pd, :n_col_lb])
        idx_strt_bar = n_lb+row_sep+n_pd+row_sep
        h_bar = 1
        ax_cb_c = fig.add_subplot(gs[idx_strt_bar:idx_strt_bar+h_bar, :n_col_lb])
        ax_cb_i = fig.add_subplot(gs[idx_strt_bar+h_bar+row_sep:idx_strt_bar+h_bar+row_sep+h_bar, :n_col_lb])
        ax_sig = fig.add_subplot(gs[:, n_col_lb+col_sep:])

        loss = f'{loss:.3f}'
        plt.figtext(0.1, 0.96, rf'$loss = {loss}$')

        cmap_correct = sns.color_palette(self.palette_correct, as_cmap=True)  # with integer 1 the color is weird
        cmap_incorrect = sns.color_palette(self.palette_incorrect, as_cmap=True)
        y, cs = [100] * len(str_lbs), [cmap_correct(1.)] * len(str_lbs)
        w = 0.125
        barplot(x=str_lbs, y=y, ax=ax_lb, palette=cs, orient='h', width=w, xlabel='Ground truths', with_value=False)
        y = [round(c*100, 1) for c in confs]
        vals = y + [100]
        cs = [(cmap_correct(conf) if c else cmap_incorrect(conf)) for p, conf, c in zip(str_preds, confs, pred_correct)]
        barplot(
            x=str_preds, y=y, ax=ax_pd, palette=cs, orient='h', width=w, xlabel='Predictions', ylabel='Confidence'
        )
        ma, mi = 105, max(round(min(confs)*100, -1)-10, 0)  # for there's always 100-confidence ground truth
        for ax in [ax_lb, ax_pd]:
            ax.set_xlim([mi, ma])
        ax_lb.axes.xaxis.set_ticks([])
        ax_pd.axes.xaxis.set_ticks([])
        set_color_bar(vals, ax=ax_cb_c, color_palette=self.palette_correct, orientation='horizontal')
        set_color_bar(vals, ax=ax_cb_i, color_palette=self.palette_incorrect, orientation='horizontal')

        ecg_util.plot_ecg(
            sig, xlabel='timestep', ylabel='V', title='Input signal', legend=False, ax=ax_sig, gap_factor=1.5
        )
        mi, ma = ax_sig.get_ylim()
        h = ma - mi
        cmap = sns.color_palette('Blues_r', as_cmap=True)  # higher is more saturated
        c_edge = cmap(1)
        i_layer = self.model.config.num_hidden_layers - 1  # last layer
        for i_pch, attn_score in zip(range(L // patch_size), attn_res[i_layer]):
            strt = i_pch * patch_size
            c = cmap(attn_score)
            rect = patches.Rectangle(xy=(strt, mi), width=patch_size, height=h, facecolor=c, alpha=attn_score)
            ax_sig.add_patch(rect)
            if strt != 0:
                ax_sig.axvline(x=strt, lw=0.2, c=c_edge)
        title = f'[CLS] <= Patch token Attention Map at layer {i_layer+1}'
        plt.suptitle(title)
        save_fig(title) if save else plt.show()


if __name__ == '__main__':
    import pickle

    from icecream import ic

    def check_forward_pass():
        ev = EcgVit()
        # ic(ev)
        sigs = torch.randn(4, 12, 2560)
        ic(ev.vit.to_patch_embedding(torch.randn(4, 12, 1, 2560)).shape)

        labels_ = torch.zeros(4, 71)
        labels_[[0, 0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6]] = 1
        ic(labels_)
        loss_, logits_ = ev(sigs, labels_)
        ic(sigs.shape, loss_, logits_.shape)
    # check_forward_pass()

    mdl = load_trained()
    evv = EcgVitVisualizer(mdl)
    dsets = get_ptbxl_dataset(type='original', pad=mdl.config.patch_size, std_norm=True)

    def check_visualize_attn():
        dnm = 'PTB-XL'
        code_norm = 'NORM'  # normal heart beat
        code2id = config(f'datasets.{dnm}.code.code2id')
        id_norm = code2id[code_norm]

        inputs = None
        # for inputs in tqdm(dsets.test):
        for inputs in dsets.test:
            if inputs['labels'][id_norm] == 1:  # found a sample with normal heart beat
                break
        assert inputs is not None

        evv(**inputs)
    # check_visualize_attn()

    def visualize_a_few():
        from ecg_transformer.models.evaluate import get_eval_path

        fnm = 'eval_edge_example_samples, 2022-04-19_00-41-46'
        path_out = os.path.join(get_eval_path(), 'samples', mdl.to_str())
        with open(os.path.join(path_out, f'{fnm}.pkl'), 'rb') as f:
            samples = pickle.load(f)
        # ic(samples)

        def check_one():
            idx_high_loss = get(samples, 'test.high')[1]  # pick the 1st one arbitrarily
            sample = dsets.test[idx_high_loss]
            evv(**sample)
        # check_one()

        def save():
            for split, idxs in samples['test'].items():
                for idx in idxs:
                    sample = dsets.test[idx]
                    evv(**sample, save=True)
        save()
    visualize_a_few()
