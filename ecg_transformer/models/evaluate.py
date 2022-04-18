import os
import json
from typing import Dict, Any

import torch

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


def get_eval_path() -> str:
    return os.path.join(PATH_BASE, DIR_PROJ, 'evaluations')


def evaluate_trained() -> Dict[str, Dict[str, Any]]:
    model = load_trained()
    ptbxl_type = 'original'
    n_sample = None
    # n_sample = 64  # TODO: debugging

    dnm = 'PTB-XL'
    pad = transform.TimeEndPad(model.config.patch_size, pad_kwargs=dict(mode='constant', constant_values=0))
    stats = config(f'datasets.{dnm}.train-stats.{ptbxl_type}')
    dset_args = dict(type=ptbxl_type, normalize=stats, transform=pad, return_type='pt')
    dsets = get_ptbxl_splits(n_sample=n_sample, dataset_args=dset_args)
    trainer = MyTrainer(model=model, args=dict(eval_batch_size=16))
    # splits = ['train', 'eval', 'test']
    splits = ['eval', 'test']
    split2perf = {s: trainer.evaluate(getattr(dsets, s)) for s in splits}

    model_dir_nm = f'{model.__class__.__qualname__}, {model.config.size}'
    path_out = os.path.join(get_eval_path(), model_dir_nm)
    os.makedirs(path_out, exist_ok=True)
    with open(os.path.join(path_out, f'evaluation, {now(for_path=True)}.json'), 'w') as f:
        json.dump(split2perf, f, indent=4)
    # ic(d_out)
    return split2perf


if __name__ == '__main__':
    from icecream import ic

    def check_load():
        model = load_trained()
        ic(type(model), get_model_num_trainable_parameter(model))
    # check_load()

    evaluate_trained()
