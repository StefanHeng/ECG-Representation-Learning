import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import get_cosine_schedule_with_warmup

from ecg_transformer.util import *
import ecg_transformer.util.train as train_util
from ecg_transformer.preprocess import PtbxlDataModule
from ecg_transformer.models import EcgVitConfig


class EcgVitTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, train_kwargs: Dict = None):
        super().__init__()
        self.model = model
        self.train_args, self.train_meta, self.parent_trainer = (
            train_kwargs['args'], train_kwargs['meta'], train_kwargs['parent']
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.train_args['learning_rate'], weight_decay=self.train_args['weight_decay']
        )
        warmup_ratio, n_step = self.train_args['warmup_ratio'], self.train_meta['#step']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=round(n_step * warmup_ratio), num_training_steps=n_step
        )
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

    def training_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        return dict(loss=loss, logits=logits.detach(), labels=batch['labels'].detach())

    def validation_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        return dict(loss=loss, logits=logits.detach(), labels=batch['labels'].detach())

    def training_step_end(self, step_output):
        loss, logits, labels = step_output['loss'], step_output['logits'], step_output['labels']
        d_log = dict(epoch=self.current_epoch+1, step=self.global_step+1)  # 1-indexed
        lr = self.parent_trainer.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        d_update = {  # colab compatibility
            **dict(learning_rate=lr, loss=loss.detach().item()),
            **train_util.get_accuracy(torch.sigmoid(logits), labels, return_auc=False)
        }
        d_log.update({f'train/{k}': v for k, v in d_update.items()})
        self._log_info(d_log)

    def validation_epoch_end(self, outputs):
        loss = np.array([d['loss'].detach().item() for d in outputs]).mean()
        logits, labels = torch.cat([d['logits'] for d in outputs]), torch.cat([d['labels'] for d in outputs])
        preds_prob = torch.sigmoid(logits)
        d_log = dict(epoch=self.current_epoch+1, step=self.global_step+1)
        d_update = {
            **dict(loss=loss),
            **train_util.get_accuracy(preds_prob, labels)
        }
        d_log.update({f'eval/{k}': v for k, v in d_update.items()})
        self._log_info(d_log)

    def _log_info(self, x):
        if self.parent_trainer is not None:
            self.parent_trainer.log(x)


class MyTrainer:
    def __init__(self, name='EcgVit Train', model_args: Dict = None, train_args: Dict = None, log2console=True):
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
        output_dir, learning_rate, weight_decay, train_batch_size, num_train_epoch = (self.train_args[k] for k in (
            'output_dir', 'learning_rate', 'weight_decay', 'train_batch_size', 'num_train_epoch'
        ))
        if model_args is None:
            model_args = dict()
        model_name = model_args.pop('name', 'vit-small')
        conf = EcgVitConfig.from_defined(model_name)
        self.data_module = PtbxlDataModule(
            train_args=self.train_args, dataset_args=dict(
                init_kwargs=dict(patch_size=conf.patch_size)
            )
        )
        n_step = len(self.data_module.train_dataloader()) * num_train_epoch  # TODO: gradient accumulation not supported
        B, C, L = next(iter(DataLoader(self.data_module.dset_tr, batch_size=2)))['sample_values'].shape
        assert L % conf.patch_size == 0
        self.train_meta = {
            '#step': n_step, '#epoch': num_train_epoch,
            'model input shape': f'B x {C} x {L}', '#patch': L // conf.patch_size
        }
        self.model = EcgVitTrainer(
            train_kwargs=dict(args=self.train_args, meta=self.train_meta, parent=self), **model_args
        )
        self.log_fnm = f'{self.model.__class__.__qualname__}, ' \
                       f'n={len(self.data_module.dset_tr)}, a={learning_rate}, dc={weight_decay}, ' \
                       f'bsz={train_batch_size}, n_ep={num_train_epoch}'

        self.log2console = log2console
        self.logger, self.logger_fl, self.logger_tb, self.trainer = None, None, None, None
        # cos the internal epoch for sanity check eval is always 0
        self._ran_sanity_check_eval, self._eval_epoch_count = False, 1

    def train(self):
        output_dir, n_ep = self.train_args['output_dir'], self.train_args['num_train_epoch']
        self.logger: logging.Logger = get_logger(self.name)
        self.logger_fl = get_logger(
            name=self.name, typ='file-write', file_path=os.path.join(output_dir, f'{self.log_fnm}.log')
        )
        self.logger.info(f'Launched training model {logi(self.model.config)} '
                         f'with args {log_dict_pg(self.train_args)} and {log_dict(self.train_meta)}... ')
        self.logger_fl.info(f'Launched training model {self.model.config} '
                            f'with args {log_dict_id(self.train_args)} and {log_dict_nc(self.train_meta)}... ')
        self.logger_tb = TensorBoardLogger(output_dir, name=f'{self.save_time} - {self.log_fnm}')
        self.trainer = pl.Trainer(
            logger=self.logger_tb,
            default_root_dir=output_dir,
            gradient_clip_val=1,
            check_val_every_n_epoch=1,
            max_epochs=n_ep,
            log_every_n_steps=1,
            gpus=torch.cuda.device_count(),
            accelerator='auto',
            precision=self.train_args['precision'],
            weights_save_path=os.path.join(output_dir, 'weights'),
            num_sanity_val_steps=-1,  # Runs & logs eval before training starts
            deterministic=True,
            detect_anomaly=True,
            move_metrics_to_cpu=True,
        )
        self.trainer.fit(self.model, self.data_module)

    def log(self, msg):
        is_dict = isinstance(msg, dict)
        msg_ = train_util.pretty_log_dict(msg, ref=self.train_meta) if is_dict else msg
        if self.logger is not None and self.log2console:
            self.logger.info(log_dict(msg_) if is_dict else msg_)
        if self.logger_fl is not None:
            self.logger_fl.info(log_dict_nc(msg_) if is_dict else msg_)
        if self.logger_tb is not None and is_dict:
            is_eval = not any('learning_rate' in k for k in msg.keys())  # heuristics to detect eval
            if is_eval:
                step = msg.pop('epoch')
                assert step == self._eval_epoch_count
                if not self._ran_sanity_check_eval:
                    self._ran_sanity_check_eval = True
                    step -= 1  # effectively, eval "step" is for eval and 0-indexed, unlike train logging
                    self._eval_epoch_count -= 1
                self._eval_epoch_count += 1
                del msg['step']
            else:
                step = msg.pop('step')
            msg = {k: v for k, v in msg.items() if ('per_class_auc' not in k and 'epoch' not in k and bool(v))}
            self.logger_tb.log_metrics(msg, step=step)


if __name__ == '__main__':
    from pytorch_lightning.utilities.seed import seed_everything
    from icecream import ic

    seed_everything(config('random-seed'))

    def check_forward_pass():
        ev = EcgVitTrainer()
        # ic(ev)
        sigs = torch.randn(4, 12, 2560)
        ic(ev.vit.to_patch_embedding(torch.randn(4, 12, 1, 2560)).shape)

        labels_ = torch.zeros(4, 71)
        labels_[[0, 0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6]] = 1
        ic(labels_)
        loss_, logits_ = ev(sigs, labels_)
        ic(sigs.shape, loss_, logits_.shape)
    # check_forward_pass()

    def train():
        conf = EcgVitConfig()
        conf.num_hidden_layers = 2
        trainer = MyTrainer(
            model_args=dict(model_config=conf),
            train_args=dict(
                num_train_epoch=8,
                train_batch_size=4,  # TODO: debugging
                eval_batch_size=4,
                warmup_ratio=0.1,
                n_sample=32,
                precision=16
            )
        )
        trainer.train()
    train()
