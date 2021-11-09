import json
from data_path import *


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(f'{PATH_BASE}/config.json') as f:
            config.config = json.load(f)

    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node


if __name__ == '__main__':
    from icecream import ic
    ic(config('datasets.BIH_MVED'))
