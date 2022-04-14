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

    @staticmethod
    def check_mismatch(arg_type: str, arg_value: str, expected_values: List[str]):
        if arg_value not in expected_values:
            raise ValueError(f'Unexpected {logi(arg_type)}: '
                             f'expect one of {logi(expected_values)}, got {logi(arg_value)}')

    @staticmethod
    def check_ptbxl_type(type: str):
        CheckArg.check_mismatch('PTB-XL dataset type', type, CheckArg.ptbxl_types)

    @staticmethod
    def check_dataset_name(dataset_name: str):
        CheckArg.check_mismatch('Dataset Name', dataset_name, CheckArg.dataset_names)

    def __init__(self):
        self.d_name2func = dict(
            type=CheckArg.check_ptbxl_type,
            dataset_name=CheckArg.check_dataset_name
        )

    def __call__(self, **kwargs):
        for k in kwargs:
            self.d_name2func[k](kwargs[k])


ca = CheckArg()


if __name__ == '__main__':
    type = 'original'
    ca(type=type)
