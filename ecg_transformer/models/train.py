import torch
from torch.nn import Module
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import get_cosine_schedule_with_warmup

from ecg_transformer.util import *
import ecg_transformer.util.train as train_util
from ecg_transformer.preprocess import transform, PtbxlDataModule
from ecg_transformer.models import EcgVitConfig, EcgVit


class EcgVitTrainModule(pl.LightningModule):
    def __init__(self, model: Module, train_args: Dict = None, parent_trainer: 'MyTrainer' = None):
        super().__init__()
        self.model = model
        self.train_args, self.parent_trainer = train_args, parent_trainer

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.train_args['learning_rate'], weight_decay=self.train_args['weight_decay']
        )
        warmup_ratio, n_step = self.train_args['warmup_ratio'], self.train_args['n_step']
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
        lr = self.parent_trainer.get_curr_learning_rate()
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
    def __init__(
            self, name='EcgVit Train', model: Module = None,
            data_module: PtbxlDataModule = None,  train_args: Dict = None
    ):
        self.name = name
        self.save_time = now(for_path=True)
        self.train_args = train_args
        self.output_dir = self.train_args['output_dir'] = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, self.save_time)
        learning_rate, weight_decay, train_batch_size, num_train_epoch, n_step = (self.train_args[k] for k in (
            'learning_rate', 'weight_decay', 'train_batch_size', 'num_train_epoch', 'n_step'
        ))
        self.model, self.data_module = model, data_module
        self.pl_module = EcgVitTrainModule(model=model, train_args=train_args, parent_trainer=self)
        model_meta = model.meta
        self.train_meta = {'model': model_meta, '#epoch': num_train_epoch, '#step': n_step, 'bsz': train_batch_size}
        self.log_fnm = f'{model_meta["name"]}, ' \
                       f'n={len(data_module.dset_tr)}, a={learning_rate}, dc={weight_decay}, ' \
                       f'bsz={train_batch_size}, n_ep={num_train_epoch}'
        self.logger, self.logger_fl, self.logger_tb, self.trainer = None, None, None, None
        # cos the internal epoch for sanity check eval is always 0
        self._ran_sanity_check_eval, self._eval_epoch_count = False, 1

    def train(self):
        n_ep = self.train_args['num_train_epoch']
        self.logger: logging.Logger = get_logger(self.name)
        self.logger_fl = get_logger(
            name=self.name, typ='file-write', file_path=os.path.join(self.output_dir, f'{self.log_fnm}.log')
        )
        self.logger.info(f'Launched training model {logi(self.model.config)} '
                         f'with args {log_dict_pg(self.train_args)} and {log_dict(self.train_meta)}... ')
        self.logger_fl.info(f'Launched training model {self.model.config} '
                            f'with args {log_dict_id(self.train_args)} and {log_dict_nc(self.train_meta)}... ')
        tb_fnm = f'tb - {self.log_fnm}'
        os.makedirs(os.path.join(self.output_dir, tb_fnm), exist_ok=True)
        self.logger_tb = TensorBoardLogger(self.output_dir, name=tb_fnm)
        self.trainer = pl.Trainer(
            logger=self.logger_tb,
            default_root_dir=self.output_dir,
            enable_progress_bar=False,
            gradient_clip_val=1,
            check_val_every_n_epoch=1,
            max_epochs=n_ep,
            log_every_n_steps=1,
            gpus=torch.cuda.device_count(),
            accelerator='auto',
            precision=self.train_args['precision'],
            weights_save_path=os.path.join(self.output_dir, 'weights'),
            num_sanity_val_steps=-1,  # Runs & logs eval before training starts
            deterministic=True,
            detect_anomaly=True,
            move_metrics_to_cpu=True,
        )
        self.trainer.fit(self.pl_module, self.data_module)

    def get_curr_learning_rate(self):
        assert self.trainer is not None
        return self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]

    def log(self, msg):
        is_dict = isinstance(msg, dict)
        assert is_dict
        is_eval = not any('learning_rate' in k for k in msg.keys())  # heuristics to detect eval
        if is_eval:
            step = n_ep = msg['epoch']
            assert n_ep == self._eval_epoch_count
            if not self._ran_sanity_check_eval:
                self._ran_sanity_check_eval = True
                n_ep -= 1  # effectively, eval epoch is 0-indexed, unlike train logging
                self._eval_epoch_count -= 1
            self._eval_epoch_count += 1
            msg['epoch'] = self._eval_epoch_count-1
            del msg['step']
        else:
            step = msg['step']
        msg_ = train_util.pretty_log_dict(msg, ref=self.train_meta) if is_dict else msg
        should_log = True
        if self.train_args['log_per_epoch'] and not is_eval and msg['step'] % self.train_args['steps_per_epoch'] != 0:
            should_log = False
        if self.logger is not None and self.train_args['log_to_console'] and should_log:
            self.logger.info(log_dict(msg_) if is_dict else msg_)
        if self.logger_fl is not None:
            self.logger_fl.info(log_dict_nc(msg_) if is_dict else msg_)
        if self.logger_tb is not None and is_dict:
            if 'step' in msg:
                del msg['step']
            msg = {k: v for k, v in msg.items() if ('per_class_auc' not in k and 'epoch' not in k and bool(v))}
            self.logger_tb.log_metrics(msg, step=step)


def get_train_args(args: Dict = None) -> Dict:
    default_args = dict(
        num_train_epoch=3,
        train_batch_size=64,
        eval_batch_size=64,
        learning_rate=3e-4,
        weight_decay=1e-1,
        warmup_ratio=0.05,
        n_sample=None,
        patience=8,
        precision=16 if torch.cuda.is_available() else 'bf16',
        log_per_epoch=False,
        log_to_console=True
    )
    args_ = default_args
    if args is not None:
        args_.update(args)
    return args_


def get_all_setup(
        model_name: str = 'ecg-vit', model_size: str = 'small', train_args: Dict = None
) -> Tuple[Module, MyTrainer]:
    assert model_name == 'ecg-vit'
    conf = EcgVitConfig.from_defined(f'{model_name}-{model_size}')
    model = EcgVit(config=conf)
    train_args = get_train_args(train_args)

    pad = transform.TimeEndPad(conf.patch_size, pad_kwargs=dict(mode='constant', constant_values=0))  # zero-padding
    dset_args = dict(normalize=('std', 1), transform=pad, return_type='pt')
    data_module = PtbxlDataModule(train_args=train_args, dataset_args=dset_args)

    # TODO: gradient accumulation not supported
    train_args['steps_per_epoch'] = steps_per_epoch = len(data_module.train_dataloader())
    train_args['n_step'] = steps_per_epoch * train_args['num_train_epoch']
    trainer = MyTrainer(model=model, data_module=data_module, train_args=train_args)
    return model, trainer


if __name__ == '__main__':
    from pytorch_lightning.utilities.seed import seed_everything
    from icecream import ic

    seed_everything(config('random-seed'))

    def train():
        model_size = 'debug'

        train_args = dict(
            num_train_epoch=4,
            train_batch_size=2,  # TODO: debugging
            eval_batch_size=2,
            warmup_ratio=0.1,
            n_sample=4,
            precision=16,
            # log_per_epoch=True,
            # log_to_console=False
        )
        model, trainer = get_all_setup(model_size=model_size, train_args=train_args)
        trainer.train()
    train()
