import os
import json
from typing import Dict, Any

from ecg_transformer.util import *
from ecg_transformer.preprocess import get_ptbxl_dataset
from ecg_transformer.models import ecg_vit
from ecg_transformer.models.train import MyTrainer


def get_eval_path() -> str:
    return os.path.join(PATH_BASE, DIR_PROJ, 'evaluations')


def evaluate_trained() -> Dict[str, Dict[str, Any]]:
    model = ecg_vit.load_trained()
    ptbxl_type = 'original'
    n_sample = None
    # n_sample = 64  # TODO: debugging

    dsets = get_ptbxl_dataset(ptbxl_type, pad=model.config.patch_size, std_norm=True, n_sample=n_sample)
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
        model = ecg_vit.load_trained()
        ic(type(model), get_model_num_trainable_parameter(model))
    # check_load()

    evaluate_trained()
