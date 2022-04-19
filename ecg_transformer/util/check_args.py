from typing import List

from ecg_transformer.util.util import *


class CheckArg:
    """
    Raise errors when common arguments don't match the expected values
    """
    dataset_names = [
        'BIH-MVED', 'INCART',
        'PTB-XL', 'PTB-Diagnostic',
        'CSPC', 'CSPC-CinC', 'CSPC-Extra-CinC',
        'G12EC', 'CHAP-SHAO', 'CODE-TEST'
    ]
    ptbxl_types = ['original', 'denoised']  # my de-noised version using the **other** PTB-XL paper approach

    model_names = ['ecg-vit-debug', 'ecg-vit-tiny', 'ecg-vit-small', 'ecg-vit-base', 'ecg-vit-large']
    optimizer = ['Adam', 'AdamW']
    schedule = ['constant', 'cosine']

    orients = ['v', 'h', 'vertical', 'horizontal']

    @staticmethod
    def check_mismatch(arg_type: str, arg_value: str, expected_values: List[str]):
        if arg_value not in expected_values:
            raise ValueError(f'Unexpected {logi(arg_type)}: '
                             f'expect one of {logi(expected_values)}, got {logi(arg_value)}')

    @staticmethod
    def check_dataset_name(dataset_name: str):
        CheckArg.check_mismatch('Dataset Name', dataset_name, CheckArg.dataset_names)

    @staticmethod
    def check_ptbxl_type(type: str):
        CheckArg.check_mismatch('PTB-XL dataset type', type, CheckArg.ptbxl_types)

    @staticmethod
    def check_model_name(model_name: str):
        CheckArg.check_mismatch('Model Name', model_name, CheckArg.model_names)

    @staticmethod
    def check_optimizer(optimizer: str):
        CheckArg.check_mismatch('Optimizer', optimizer, CheckArg.optimizer)

    @staticmethod
    def check_schedule(schedule: str):
        CheckArg.check_mismatch('Schedule', schedule, CheckArg.schedule)

    @staticmethod
    def check_orient(orient: str):
        CheckArg.check_mismatch('Bar Plot Orientation', orient, CheckArg.orients)

    def __init__(self):
        self.d_name2func = dict(
            dataset_name=CheckArg.check_dataset_name,
            type=CheckArg.check_ptbxl_type,
            model_name=CheckArg.check_model_name,
            optimizer=CheckArg.check_optimizer,
            schedule=CheckArg.check_schedule,
            orient=CheckArg.check_orient
        )

    def __call__(self, **kwargs):
        for k in kwargs:
            self.d_name2func[k](kwargs[k])


ca = CheckArg()


if __name__ == '__main__':
    type = 'original'
    ca(type=type)
