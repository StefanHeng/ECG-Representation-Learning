"""
Vision Transformer adapted to 1D ECG signals

Intended fpr vanilla, supervised training
"""

import torch
from torch import nn
from transformers import PretrainedConfig
from vit_pytorch import ViT

from ecg_transformer.util import *
from ecg_transformer.util.models import ModelOutput


class EcgVitConfig(PretrainedConfig):
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
            **kwargs
    ):
        self.max_signal_length = max_signal_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_class = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        super().__init__(**kwargs)

    @classmethod
    def from_defined(cls, model_name):
        """
        A few model sizes I defined
        """
        model_names = ['ecg-vit-debug', 'ecg-vit-tiny', 'ecg-vit-small', 'ecg-vit-base', 'ecg-vit-large']
        assert model_name in model_names, \
            f'Unexpected model name: expect one of {logi(model_names)}, got {logi(model_name)}'
        conf = cls()
        if model_name == 'ecg-vit-debug':
            conf = cls.from_defined('ecg-vit-tiny')
            conf.num_hidden_layers = 2
        elif model_name == 'ecg-vit-tiny':
            conf.hidden_size = 256
            conf.num_hidden_layers = 8
            conf.num_attention_heads = 8
            conf.intermediate_size = 1024
        elif model_name == 'ecg-vit-small':
            conf.hidden_size = 512
            conf.num_hidden_layers = 8
            conf.num_attention_heads = 8
            conf.intermediate_size = 2048
        elif model_name == 'ecg-vit-base':
            conf.hidden_size = 768
            conf.num_hidden_layers = 12
            conf.num_attention_heads = 12
            conf.intermediate_size = 3072
        elif model_name == 'ecg-vit-large':
            conf.hidden_size = 1024
            conf.num_hidden_layers = 24
            conf.num_attention_heads = 16
            conf.intermediate_size = 4096
        return conf


class EcgVit(nn.Module):
    def __init__(self, num_class: int = 71, config=EcgVitConfig(),):
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
        self.loss_fn = nn.BCEWithLogitsLoss()  # TODO: more complex loss, e.g. weighting?

        C, L = self.config.num_channels, self.config.max_signal_length
        self.meta = {
            'name': self.__class__.__qualname__, 'input shape': f'{C} x {L}', '#patch': L // self.config.patch_size
        }

    def forward(self, sample_values: torch.FloatTensor, labels: torch.LongTensor = None):
        logits = self.vit(sample_values.unsqueeze(-2))   # Add dummy height dimension
        loss = None
        if labels is not None:
            loss = self.loss_fn(input=logits, target=labels)
        from icecream import ic
        ic(loss, logits, logits.isnan().nonzero(), sample_values.isnan().any())
        return ModelOutput(loss=loss, logits=logits)


if __name__ == '__main__':
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
