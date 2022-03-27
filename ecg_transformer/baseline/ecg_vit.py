"""
Vision Transformer adapted to 1D ECG signals

Intended fpr vanilla, supervised training
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vit_pytorch import ViT

from ecg_transformer.preprocess import get_ptbxl_splits
from ecg_transformer.util.model import ModelOutput


class EcgVit(pl.LightningModule):
    def __init__(
            self,
            max_signal_length: int = 2560,
            chunk_size: int = 64,
            num_channels: int = 12,
            num_class: int = 71,
            hidden_size: int = 512,  # Default parameters are 2/3 of ViT base model sizes
            num_hidden_layers: int = 12,
            num_attention_heads: int = 8,
            intermediate_size: int = 2048,
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        dim_head = hidden_size // num_attention_heads
        self.config = dict(
            image_size=(max_signal_length, 1),
            patch_size=(chunk_size, 1),
            num_classes=num_class,
            dim=hidden_size,
            depth=num_hidden_layers,
            heads=num_attention_heads,
            mlp_dim=intermediate_size,
            pool='cls',
            channels=num_channels,
            dim_head=dim_head,
            dropout=hidden_dropout_prob,
            emb_dropout=attention_probs_dropout_prob
        )
        self.vit = ViT(**self.config)
        self.loss_fn = nn.BCEWithLogitsLoss()  # TODO: weighting?

    def forward(self, sample_values: torch.FloatTensor, labels: torch.LongTensor = None):
        logits = self.vit(sample_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(input=logits, target=labels)
        return ModelOutput(loss=loss, logits=logits)

    def training_step(self, batch, batch_idx):
        return self(**batch)

    def validation_step(self, batch, batch_idx):
        return self(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class PtbxlDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        self.dest_tr, self.dset_vl, self.dset_ts = get_ptbxl_splits()

    def train_dataloader(self):
        # TODO: signal transforms
        return DataLoader(self.dset_tr, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.dset_vl, batch_size=64)


if __name__ == '__main__':
    from icecream import ic

    def check_forward_pass():
        ev = EcgVit()
        # ic(ev)
        sigs = torch.randn(4, 12, 2560, 1)
        labels_ = torch.zeros(4, 71)
        labels_[[0, 0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6]] = 1
        ic(labels_)
        loss_, logits_ = ev(sigs, labels_)
        ic(sigs.shape, loss_, logits_.shape)

    data_module = PtbxlDataModule()
    model = EcgVit()
    trainer = pl.Trainer()
    trainer.fit(model, data_module)
