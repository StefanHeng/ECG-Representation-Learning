import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm

from ecg_transformer.util import *
import ecg_transformer.util.train as train_util
from ecg_transformer.preprocess import EcgDataset, transform, PtbxlDataModule
from ecg_transformer.models import EcgVitConfig, EcgVit


class EcgVitTrainModule(pl.LightningModule):
    def __init__(self, model: nn.Module, train_args: Dict = None, parent_trainer: 'MyPlTrainer' = None):
        super().__init__()
        self.model = model
        self.train_args, self.parent_trainer = train_args, parent_trainer

    def forward(self, **kwargs):
        # ic(kwargs['sample_values'])
        return self.model(**kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.train_args['learning_rate'], weight_decay=self.train_args['weight_decay']
        )
        warmup_ratio, n_step = self.train_args['warmup_ratio'], self.train_args['n_step']
        sch, args = self.train_args['schedule'], dict(optimizer=optimizer, num_warmup_steps=round(n_step*warmup_ratio))
        if sch == 'constant':
            sch = get_constant_schedule_with_warmup
        else:
            sch = get_cosine_schedule_with_warmup
            args.update(dict(num_training_steps=n_step))
        scheduler = sch(**args)
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
            **train_util.get_accuracy(torch.sigmoid(logits), labels, return_auc=True)
        }
        d_log.update({f'train/{k}': v for k, v in d_update.items()})
        self._log_info(d_log)

    def validation_epoch_end(self, outputs):
        loss = np.array([d['loss'].detach().item() for d in outputs]).mean()
        logits, labels = torch.cat([d['logits'] for d in outputs]), torch.cat([d['labels'] for d in outputs])
        preds_prob = torch.sigmoid(logits)
        d_log = dict(epoch=self.current_epoch+1, step=self.global_step+1)
        d_update = {**dict(loss=loss), **train_util.get_accuracy(preds_prob, labels)}
        for k, v in d_update.items():
            if not isinstance(v, dict):
                self.parent_trainer.pl_trainer.callback_metrics[k] = torch.tensor(v)  # per `ModelCheckpoint`
        d_log.update({f'eval/{k}': v for k, v in d_update.items()})
        self._log_info(d_log)

    def _log_info(self, x):
        if self.parent_trainer is not None:
            self.parent_trainer.log(x)


class MyPlTrainer:
    """
    Seems to be some problem with either the PL library or my implementation,
        that the logits for all inputs are all the same
    => Deprecated, see `MyTrainer`, where I don't observe the problem with my own training loop
    """
    def __init__(
            self, name='EcgVit Train', model: nn.Module = None,
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
        self.logger, self.logger_fl, self.logger_tb, self.pl_trainer = None, None, None, None
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
        callback = None
        if self.train_args['save_while_training']:
            callback = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(self.output_dir, 'checkpoints'),
                monitor='loss',  # having `eval/loss` seems to break the filename
                filename='checkpoint-{epoch:02d}, {loss:.2f}',
                every_n_epochs=self.train_args['save_every_n_epoch'],
                save_top_k=self.train_args['save_top_k'],
                save_last=True
            )
        self.pl_trainer = pl.Trainer(
            logger=self.logger_tb,
            default_root_dir=self.output_dir,
            enable_progress_bar=False,
            callbacks=callback,
            gradient_clip_val=1,
            # check_val_every_n_epoch=1,
            check_val_every_n_epoch=int(1e10),  # Disable evaluation, TODO: debugging
            max_epochs=n_ep,
            log_every_n_steps=1,
            gpus=torch.cuda.device_count(),
            accelerator='auto',
            precision=self.train_args['precision'],
            num_sanity_val_steps=-1,  # Runs & logs eval before training starts
            deterministic=True,
            detect_anomaly=True,
            move_metrics_to_cpu=True,
        )
        self.pl_trainer.fit(self.pl_module, self.data_module)

    def get_curr_learning_rate(self):
        assert self.pl_trainer is not None
        return self.pl_trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]

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


class MyTrainer:
    """
    My own training loop, fp16 not supported
    """
    tb_ignore_keys = ['step', 'epoch', 'per_class_auc']

    def __init__(
            self, name='EcgVit Train', model: EcgVit = None,
            train_dataset: EcgDataset = None, eval_dataset: EcgDataset = None, args: Dict = None
    ):
        self.name = name
        self.save_time = now(for_path=True)
        self.args = args
        self.output_dir = self.args['output_dir'] = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, self.save_time)
        learning_rate, weight_decay, train_batch_size, num_train_epoch, n_step = (self.args[k] for k in (
            'learning_rate', 'weight_decay', 'train_batch_size', 'num_train_epoch', 'n_step'
        ))

        self.model, self.train_dataset, self.eval_dataset = model, train_dataset, eval_dataset
        self.optimizer, self.scheduler = None, None
        self.epoch, self.step = None, None
        self.t_strt = None

        model_meta = model.meta
        self.train_meta = {'model': model_meta, '#epoch': num_train_epoch, '#step': n_step, 'bsz': train_batch_size}
        self.log_fnm = f'model={model_meta}, ' \
                       f'n={len(train_dataset)}, a={learning_rate}, dc={weight_decay}, ' \
                       f'bsz={train_batch_size}, n_ep={num_train_epoch}'
        self.logger, self.logger_fl, self.tb_writer = None, None, None

    def train(self):
        self.logger = get_logger(self.name)
        self.logger = get_logger(self.name)
        log_path = os.path.join(self.output_dir, f'{self.log_fnm}.log')
        self.logger_fl = get_logger(name=self.name, typ='file-write', file_path=log_path)
        self.logger.info(f'Launched training model {logi(self.model.config)} '
                         f'with args {log_dict_pg(self.args)} and {log_dict(self.train_meta)}... ')
        self.logger_fl.info(f'Launched training model {self.model.config} '
                            f'with args {log_dict_id(self.args)} and {log_dict_nc(self.train_meta)}... ')
        tb_fnm = f'tb - {self.log_fnm}'
        tb_path = os.path.join(self.output_dir, tb_fnm)
        os.makedirs(tb_path, exist_ok=True)
        self.tb_writer = SummaryWriter(tb_path)

        dl = DataLoader(self.train_dataset, batch_size=self.args['train_batch_size'], shuffle=True, pin_memory=True)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args['learning_rate'], weight_decay=self.args['weight_decay']
        )
        warmup_ratio, n_step = self.args['warmup_ratio'], self.args['n_step']
        sch, args = self.args['schedule'], dict(optimizer=self.optimizer, num_warmup_steps=round(n_step*warmup_ratio))
        if sch == 'constant':
            sch = get_constant_schedule_with_warmup
        else:
            sch = get_cosine_schedule_with_warmup
            args.update(dict(num_training_steps=n_step))
        self.scheduler = sch(**args)

        if torch.cuda.is_available():
            self.model.cuda()

        self.epoch, self.step = 0, 0
        best_eval_loss, n_bad_ep = float('inf'), 0
        self.t_strt = datetime.datetime.now()
        if self.args['do_eval']:
            self.evaluate(tqdm_desc='Before training')
        for _ in range(self.args['num_train_epoch']):
            self.epoch += 1
            self.model.train()  # cos at the end of each eval, evaluate
            desc_epoch = self._get_epoch_desc()
            for inputs in tqdm(dl, desc=f'Train {desc_epoch}'):
                self.step += 1
                self.optimizer.zero_grad()

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self.model(**inputs)
                loss, logits = outputs.loss, outputs.logits.detach()
                labels = inputs['labels'].detach()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                self.optimizer.step()
                self.scheduler.step()

                d_log = dict(epoch=self.epoch, step=self.step)  # 1-indexed
                lr = self._get_lr()
                d_update = {  # colab compatibility
                    **dict(learning_rate=lr, loss=loss.detach().item()),
                    **train_util.get_accuracy(torch.sigmoid(logits), labels, return_auc=True)
                }
                d_log.update({f'train/{k}': v for k, v in d_update.items()})
                self.log(d_log)

            save_n_ep = self.args['save_every_n_epoch']
            if save_n_ep and self.epoch > 0 and self.epoch % save_n_ep == 0:
                fnm = f'model - {self.log_fnm}, ep{self.epoch}.pt'
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, fnm))
            if self.args['do_eval']:
                eval_loss = self.evaluate(tqdm_desc=desc_epoch)['eval/loss']

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    n_bad_ep = 0
                else:
                    n_bad_ep += 1
                if n_bad_ep >= self.args['patience']:
                    t = fmt_dt(datetime.datetime.now() - self.t_strt)
                    d_pat = dict(epoch=self.epoch, patience=self.args['patience'], best_eval_cls_acc=best_eval_loss)
                    self.logger.info(f'Training terminated early for {log_dict(d_pat)} in {logi(t)} ')
                    self.logger_fl.info(f'Training terminated early for {log_dict_nc(d_pat)} in {t}')
                    break
        t = fmt_dt(datetime.datetime.now() - self.t_strt)
        self.logger.info(f'Training completed in {logi(t)} ')
        self.logger_fl.info(f'Training completed in {t} ')
        self.logger, self.logger_fl, self.tb_writer = None, None, None  # reset
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'model - {self.log_fnm}.pt'))

    def evaluate(self, tqdm_desc=None):
        self.model.eval()

        bsz = self.args['eval_batch_size']
        dl = DataLoader(self.eval_dataset, batch_size=bsz, shuffle=False)

        tqdm_args = dict(unit='ba')
        if tqdm_desc is not None:
            tqdm_args.update(dict(desc=f'Eval {tqdm_desc}'))
        lst_loss, lst_logits, lst_labels = [], [], []
        for inputs in tqdm(dl, **tqdm_args):  # Each "sample" in dataset is already grouped
            with torch.no_grad():
                output = self.model(**inputs)
            lst_loss.append(output.loss.detach().item())
            lst_logits.append(output.logits.detach().cpu())
            lst_labels.append(inputs['labels'].cpu())

        loss = np.mean(lst_loss)
        logits, labels = torch.cat(lst_logits, dim=0), torch.cat(lst_labels, dim=0)
        preds_prob = torch.sigmoid(logits)
        d_log = dict(epoch=self.epoch+1, step=self.step+1)
        d_update = {**dict(loss=loss), **train_util.get_accuracy(preds_prob, labels)}
        d_log.update({f'eval/{k}': v for k, v in d_update.items()})
        self.log(d_log)
        return d_log

    def log(self, msg: Dict):
        msg_ = train_util.pretty_log_dict(msg, ref=self.train_meta)
        should_log = True
        # eval always logged
        if self.args['log_per_epoch'] and self.model.training and msg['step'] % self.args['steps_per_epoch'] != 0:
            should_log = False
        if self.args['log_to_console'] and should_log:
            self.logger.info(log_dict(msg_))
        self.logger_fl.info(log_dict_nc(msg_))
        msg = {k: v for k, v in msg.items() if MyTrainer._keep_in_tb_write(k, v)}
        for k, v in msg.items():
            self.tb_writer.add_scalar(tag=k, scalar_value=v, global_step=self.step)

    @staticmethod
    def _keep_in_tb_write(k: str, v: Any) -> bool:
        return not any(key in k for key in MyTrainer.tb_ignore_keys) and bool(v)

    def _get_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]

    def _get_epoch_desc(self) -> str:
        n_ep_str = train_util.pretty_single('epoch', self.epoch, {'#epoch': self.args['num_train_epoch']})
        return f'Epoch {n_ep_str.strip()}'


def get_train_args(args: Dict = None, n_train: int = None) -> Dict:
    default_args = dict(
        num_train_epoch=3,
        train_batch_size=64,
        eval_batch_size=64,
        do_eval=True,
        learning_rate=3e-4,
        weight_decay=1e-1,
        warmup_ratio=0.05,
        n_sample=None,
        patience=8,
        precision=16 if torch.cuda.is_available() else 'bf16',
        log_per_epoch=False,  # only epoch-wise evaluation is logged
        log_to_console=True,
        save_every_n_epoch=False,  # only save in the end
        save_top_k=-1  # save all models
    )
    args_ = default_args
    if args is not None:
        args_.update(args)
    # TODO: gradient accumulation not supported
    args_['steps_per_epoch'] = steps_per_epoch = n_train or int(sys.maxsize)  # see `get_all_setup` for PL
    args_['n_step'] = steps_per_epoch * args_['num_train_epoch']
    return args_


def get_all_setup(
        model_name: str = 'ecg-vit', model_size: str = 'small', train_args: Dict = None,
        ptbxl_type: str = 'denoised', with_pl: bool = False
) -> Tuple[nn.Module, MyPlTrainer]:
    assert model_name == 'ecg-vit'
    conf = EcgVitConfig.from_defined(f'{model_name}-{model_size}')
    conf.patch_size = 32  # TODO: debugging
    model = EcgVit(config=conf)

    dnm = 'PTB-XL'
    pad = transform.TimeEndPad(conf.patch_size, pad_kwargs=dict(mode='constant', constant_values=0))  # zero-padding
    stats = config(f'datasets.{dnm}.train-stats.{ptbxl_type}')
    dset_args = dict(type=ptbxl_type, normalize=stats, transform=pad, return_type='pt')
    # dset_args = dict(type=ptbxl_type, normalize=('norm', 3), transform=pad, return_type='pt')

    trainer_args = dict(model=model)
    if with_pl:
        dummy_train_args = get_train_args(train_args)  # kind of ugly
        trainer_args['data_module'] = data_module = PtbxlDataModule(dataset_args=dset_args, train_args=dummy_train_args)
        n_train = len(data_module.train_dataloader())
        trainer_args['train_args'] = get_train_args(train_args, n_train=n_train)
        cls = MyPlTrainer
    else:
        tr, vl, ts = get_ptbxl_splits(n_sample=train_args['n_sample'], dataset_args=dset_args)
        n_train = len(tr)
        trainer_args['train_dataset'], trainer_args['eval_dataset'] = tr, vl
        trainer_args['args'] = get_train_args(train_args, n_train=n_train)
        cls = MyTrainer
    return model, cls(**trainer_args)


if __name__ == '__main__':
    # from pytorch_lightning.utilities.seed import seed_everything
    import transformers

    from icecream import ic

    from ecg_transformer.preprocess import get_ptbxl_splits

    lw = 1024
    ic.lineWrapWidth = lw
    np.set_printoptions(linewidth=lw)
    torch.set_printoptions(linewidth=lw)

    # seed_everything(config('random-seed'))
    transformers.set_seed(config('random-seed'))

    def train():
        model_size = 'debug'
        # model_size = 'tiny'
        t = 'original'

        n_sample = 128
        bsz = 8
        num_train_epoch = 32

        train_args = dict(
            num_train_epoch=num_train_epoch,
            train_batch_size=bsz,
            eval_batch_size=bsz,
            learning_rate=1e-3,
            # warmup_ratio=0.1,
            warmup_ratio=0,
            schedule='constant',
            n_sample=n_sample,
            # precision=16 if torch.cuda.is_available() else 32,
            do_eval=False,
            # log_per_epoch=True,
            log_to_console=False,
            save_every_n_epoch=32,
            save_top_k=2
        )
        model, trainer = get_all_setup(model_size=model_size, train_args=train_args, ptbxl_type=t)
        trainer.train()
    train()
    # profile_runtime(train)

    def fix_check_trained_why_auc_low():

        model_key = 'ecg-vit-base'
        conf = EcgVitConfig.from_defined(model_key)
        model = EcgVit(config=conf)
        model = EcgVitTrainModule(model=model)

        checkpoint_path = os.path.join(
            PATH_BASE, DIR_PROJ, DIR_MDL,
            '2022-04-14_14-59-52', 'checkpoints', 'checkpoint-epochepoch=08, eval-loss=eval', 'loss=0.13.ckpt'
        )
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # ic(type(ckpt), ckpt.keys())
        # ic(ckpt['state_dict'].keys())
        model.load_state_dict(ckpt['state_dict'], strict=True)  # Need the pl wrapper cos that's how the model is saved
        model.eval()
        # ic(model)

        t = 'original'
        dnm = 'PTB-XL'
        pad = transform.TimeEndPad(conf.patch_size, pad_kwargs=dict(mode='constant', constant_values=0))  # zero-padding
        stats = config(f'datasets.{dnm}.train-stats.{t}')
        dset_args = dict(type=t, normalize=stats, transform=pad, return_type='pt')
        tr, vl, ts = get_ptbxl_splits(n_sample=1024, dataset_args=dset_args)
        # dl = DataLoader(tr, batch_size=4)
        dl = DataLoader(vl, batch_size=4)
        for inputs in dl:
            sample_values = inputs['sample_values']
            ic(sample_values.shape, sample_values[:, 0, :20])
            with torch.no_grad():
                outputs = model(**inputs)
            # ic(outputs)
            loss, logits, labels = outputs.loss, outputs.logits, inputs['labels']
            ic(logits)
            ic(train_util.get_accuracy(torch.sigmoid(logits), labels, return_auc=True))
            exit(1)
    # fix_check_trained_why_auc_low()

    def fix_check_why_logits_all_same():
        conf = EcgVitConfig.from_defined('ecg-vit-debug')
        conf.patch_size = 32  # half of the defined
        model = EcgVit(config=conf)

        t = 'original'
        dnm = 'PTB-XL'
        pad = transform.TimeEndPad(conf.patch_size, pad_kwargs=dict(mode='constant', constant_values=0))  # zero-padding
        stats = config(f'datasets.{dnm}.train-stats.{t}')
        dset_args = dict(type=t, normalize=stats, transform=pad, return_type='pt')
        n = 128
        bsz = 32
        tr, vl, ts = get_ptbxl_splits(n_sample=n, dataset_args=dset_args)
        dl = DataLoader(vl, batch_size=bsz, shuffle=True)

        # lr = 3e-4
        lr = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        for n_ep in range(32):
            model.train()  # cos at the end of each eval, evaluate
            for inputs in dl:
                # inputs['sample_values'] = inputs['sample_values'][:, :, :80]
                optimizer.zero_grad()

                sample_values, labels = inputs['sample_values'], inputs['labels']
                outputs = model(**inputs)
                loss, logits = outputs.loss, outputs.logits.detach()

                msk_2_class = torch.any(~labels.eq(labels[0]), dim=0)
                preds_prob = torch.sigmoid(logits)
                bin_preds = preds_prob > 0.5
                # matched = bin_preds.eq(labels)
                # acc = matched.sum().item() / matched.numel()
                # ic(sample_values[:, 0, :4], labels, logits, acc)
                d_metric = train_util.get_accuracy(preds_prob, labels)
                acc, auc = d_metric['binary_accuracy'], d_metric['macro_auc']
                ic(
                    preds_prob[:, msk_2_class], bin_preds[:, msk_2_class], labels[:, msk_2_class], acc, auc,
                    bin_preds.sum().item(),
                    # (~matched).nonzero(),
                )

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                # scheduler.step()
    # fix_check_why_logits_all_same()
    # profile_runtime(fix_check_why_logits_all_same)
