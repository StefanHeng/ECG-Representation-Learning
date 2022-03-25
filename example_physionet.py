"""
Credit: PtB-XL, a large publicly available electrocardiography dataset
"""


import pandas as pd
import numpy as np
import wfdb
import ast

from ecg_transformer.util import *


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    ic(data[0])
    exit(1)
    data = np.array([signal for signal, meta in data])
    return data


if __name__ == '__main__':
    from icecream import ic

    path = os.path.join(PATH_BASE, DIR_DSET, config('datasets.PTB_XL.dir_nm'))
    sampling_rate = 500

    labels = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')[:128]
    # ic(labels, labels.scp_codes, type(labels.scp_codes))
    # ic(labels.scp_codes[1])
    labels.scp_codes = labels.scp_codes.apply(lambda x: ast.literal_eval(x))
    # ic(labels.scp_codes, labels.scp_codes[1])

    # signals = load_raw_data(labels, sampling_rate, path)
    signals = np.stack([wfdb.rdsamp(os.path.join(path, fnm))[0] for fnm in labels.filename_hr])  # the 500Hz data
    ic(signals.shape)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    labels['diagnostic_superclass'] = labels.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10
    # Train
    x_tr = signals[np.where(labels.strat_fold != test_fold)]
    y_tr = labels[(labels.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = signals[np.where(labels.strat_fold == test_fold)]
    y_test = labels[labels.strat_fold == test_fold].diagnostic_superclass
    ic(y_tr, x_tr.shape)
    # ic(X_test, y_test)
