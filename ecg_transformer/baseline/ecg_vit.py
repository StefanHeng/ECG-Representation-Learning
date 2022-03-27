"""
Vision Transformer adapted to 1D ECG signals

Intended fpr vanilla, supervised training
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vit_pytorch import ViT
from transformers import PretrainedConfig

from ecg_transformer.util import *
from ecg_transformer.util.model import ModelOutput
from ecg_transformer.preprocess import get_ptbxl_splits


class EcgVitConfig(PretrainedConfig):
    def __init__(
            self,
            max_signal_length: int = 2560,
            chunk_size: int = 64,
            num_channels: int = 12,
            hidden_size: int = 512,  # Default parameters are 2/3 of ViT base model sizes
            num_hidden_layers: int = 12,
            num_attention_heads: int = 8,
            intermediate_size: int = 2048,
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            **kwargs
    ):
        self.max_signal_length = max_signal_length
        self.chunk_size = chunk_size
        self.num_channels = num_channels
        self.num_class = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        super().__init__(**kwargs)


class EcgVit(pl.LightningModule):
    def __init__(
            self, num_class: int = 71, model_config=EcgVitConfig(), train_args: Dict = None,
    ):
        super().__init__()
        hd_sz, n_head = model_config.hidden_size, model_config.num_attention_heads
        assert hd_sz % n_head == 0
        dim_head = hd_sz // n_head
        self.config = model_config
        _md_args = dict(
            image_size=(self.config.max_signal_length, 1),  # height is 1
            patch_size=(self.config.chunk_size, 1),
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
        self.loss_fn = nn.BCEWithLogitsLoss()  # TODO: weighting?

        self.train_args = train_args

    def forward(self, sample_values: torch.FloatTensor, labels: torch.LongTensor = None):
        ic(sample_values.shape)
        ic(sample_values.unsqueeze(-1).shape)
        logits = self.vit(sample_values.unsqueeze(-1))
        loss = None
        if labels is not None:
            loss = self.loss_fn(input=logits, target=labels)
        return ModelOutput(loss=loss, logits=logits)

    def training_step(self, batch, batch_idx):
        ic(batch)
        return self(**batch)

    def validation_step(self, batch, batch_idx):
        # ic(batch)
        lb, sv = batch['labels'], batch['sample_values']
        ic(lb.shape, sv.shape)
        return self(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.train_args['learning_rate'], weight_decay=self.train_args['weight_decay']
        )
        return optimizer


class PtbxlDataModule(pl.LightningDataModule):
    def __init__(self, train_args: Dict = None, **kwargs):
        super().__init__(**kwargs)
        self.train_args = train_args
        self.dset_tr, self.dset_vl, self.dset_ts = get_ptbxl_splits(self.train_args['n_sample'])

    def train_dataloader(self):
        # TODO: signal transforms
        return DataLoader(self.dset_tr, batch_size=self.train_args['train_batch_size'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dset_vl, batch_size=self.train_args['eval_batch_size'])


class MyTrainer:
    def __init__(self, name='EcgVit Train', train_args: Dict = None):
        self.name = name
        self.save_time = now(for_path=True)

        default_args = dict(
            num_train_epoch=3,
            train_batch_size=64,
            eval_batch_size=64,
            learning_rate=3e-4,
            weight_decay=1e-1,
            warmup_ratio=0.05,
            n_sample=None,
            patience=8,
            precision=32,
            output_dir=os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, self.save_time)
        )
        self.train_args = default_args
        if train_args is not None:
            self.train_args.update(train_args)
        self.logger = None

        self.data_module = PtbxlDataModule(self.train_args)
        self.model = EcgVit(train_args=self.train_args)
        output_dir = self.train_args['output_dir']
        self.trainer = pl.Trainer(
            default_root_dir=output_dir,
            gradient_clip_val=1,
            check_val_every_n_epoch=1,
            max_epochs=self.train_args['num_train_epoch'],
            log_every_n_steps=1,
            precision=self.train_args['precision'],
            weights_save_path=os.path.join(output_dir, 'weights'),
            # num_sanity_val_steps=-1,
            deterministic=True,
            detect_anomaly=True,
            move_metrics_to_cpu=True,
        )

    def train(self):
        self.logger: logging.Logger = get_logger(self.name)
        self.logger.info(f'Launched training model {logi(self.model.config)} '
                         f'with args {log_dict_pg(self.train_args)}... ')
        self.trainer.fit(self.model, self.data_module)


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
    # check_forward_pass()

    trainer = MyTrainer(train_args=dict(n_sample=128, precision=16))
    trainer.train()
