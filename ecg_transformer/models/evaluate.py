import os
import json
import pickle
from typing import Dict, Any

import numpy as np

from ecg_transformer.util import *
from ecg_transformer.preprocess import get_ptbxl_dataset, PtbxlSplitDatasets
from ecg_transformer.models.ecg_vit import EcgVit, load_trained
from ecg_transformer.models.train import MyTrainer


def get_eval_path() -> str:
    return os.path.join(PATH_BASE, DIR_PROJ, 'evaluations')


def evaluate_trained(model: EcgVit, dataset: PtbxlSplitDatasets) -> Dict[str, Dict[str, Any]]:
    trainer = MyTrainer(model=model, args=dict(eval_batch_size=16))
    # splits = ['train', 'eval', 'test']
    splits = ['eval', 'test']
    split2res = {s: trainer.evaluate(getattr(dataset, s)) for s in splits}

    path_out = os.path.join(get_eval_path(), model.to_str())
    os.makedirs(path_out, exist_ok=True)
    with open(os.path.join(path_out, f'evaluation, {now(for_path=True)}.json'), 'w') as f:
        json.dump(split2res, f, indent=4)
    return split2res


def pick_eval_eg(model: EcgVit, dataset: PtbxlSplitDatasets, n_sample: int = 3) -> Dict[str, Dict[str, int]]:
    """
    :return: indices in each split of the dataset, corresponding to the samples with lowest, median & highest loss
    """
    trainer = MyTrainer(model=model, args=dict(eval_batch_size=16))
    splits = ['eval', 'test']
    d_out = dict()
    for split in splits:
        dset = getattr(dataset, split)
        res = trainer.evaluate(dset, loss_reduction='none')
        loss = res['eval/loss']
        idxs = np.argsort(loss)

        def sample(n_: int) -> np.array:
            return np.random.choice(n_, size=n_sample, replace=False)
        n_group = max(round(loss.size / 10), n_sample)  # pick 10% of the data, arbitrary
        idxs_lo, idxs_hi, idxs_me = sample(n_group), sample(n_group), sample(n_group*2)
        sz = loss.size
        d_out[split] = dict(low=idxs[idxs_lo], med=idxs[sz-1 - idxs_me], high=idxs[sz//2 - n_group + idxs_hi])
    path_out = os.path.join(get_eval_path(), 'samples', model.to_str())
    os.makedirs(path_out, exist_ok=True)
    fnm = f'eval_edge_example_samples, {now(for_path=True)}'
    with open(os.path.join(path_out, f'{fnm}.pkl'), 'wb') as f:
        pickle.dump(d_out, f)
    return d_out


if __name__ == '__main__':
    from icecream import ic

    def check_load():
        model = load_trained()
        ic(type(model), get_model_num_trainable_parameter(model))
    # check_load()

    mdl = load_trained()
    ptbxl_type = 'original'
    n = None
    # n = 64  # TODO: debugging

    dsets = get_ptbxl_dataset(ptbxl_type, pad=mdl.config.patch_size, std_norm=True, n_sample=n)

    def run_eval():
        evaluate_trained(mdl, dsets)
    # run_eval()

    ic(pick_eval_eg(mdl, dsets, n_sample=16))

    def check_saved_eval_eg():
        # fnm = 'eval_edge_example_samples, 2022-04-19_00-20-28.npy'
        fnm = 'eval_edge_example_samples, 2022-04-19_00-31-17.pkl'
        path_out = os.path.join(get_eval_path(), 'samples', model.to_str())
        # samples = np.load(os.path.join(path_out, fnm), allow_pickle=True)
        with open(os.path.join(path_out, fnm), 'rb') as f:
            samples = pickle.load(f)
        ic(samples)
    # check_saved_eval_eg()
