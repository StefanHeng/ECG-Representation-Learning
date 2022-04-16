import os

import torch

from ecg_transformer.util import *
from ecg_transformer.models.ecg_vit import EcgVitConfig, EcgVit
from ecg_transformer.models.train import EcgVitTrainModule


def load_trained(model_key: str = 'ecg-vit-base'):
    model = EcgVit(config=EcgVitConfig.from_defined(model_key))

    fnm = 'model - model={nm=EcgVit, in-sp=12x2560, #p=40, #l=12, #h=12}, ' \
          'n=17441, a=0.0003, dc=0.01, bsz=256, n_ep=32, ep8.pt'
    checkpoint_path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, '2022-04-15_23-48-47', fnm)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)  # Need the pl wrapper cos that's how the model is saved
    model.eval()
    return model


if __name__ == '__main__':
    from icecream import ic

    def evaluate():
        model = load_trained()
        ic(type(model), get_model_num_trainable_parameter(model))
    evaluate()
