"""
Credit: https://github.com/hhi-aml/ecg-selfsupervised
"""


from pathlib import Path
import pickle

from tqdm import tqdm
from skimage import transform
from scipy.ndimage import zoom

from ecg_transformer.util import *

channel_stoi_default = {"i": 0, "ii": 1, "v1": 2, "v2": 3, "v3": 4, "v4": 5, "v5": 6,
                        "v6": 7, "iii": 8, "avr": 9, "avl": 10, "avf": 11, "vx": 12, "vy": 13, "vz": 14}


def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    """adds a column with mean"""
    df[col + "_mean"] = df[col].apply(
        lambda x: np.mean(np.load(x if data_folder is None else data_folder / x, allow_pickle=True), axis=axis))


def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    """adds a column with mean"""
    df[col + "_std"] = df[col].apply(
        lambda x: np.std(np.load(x if data_folder is None else data_folder / x, allow_pickle=True), axis=axis))


def dataset_add_length_col(df, col="data", data_folder=None):
    """add a length column to the dataset df"""
    df[col + "_length"] = df[col].apply(
        lambda x: len(np.load(x if data_folder is None else data_folder / x, allow_pickle=True)))


def dataset_get_stats(df, col="data", simple=True):
    '''creates (weighted) means and stds from mean, std and length cols of the df'''
    if(simple):
        return df[col+"_mean"].mean(), df[col+"_std"].mean()
    else:
        #https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        #or https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
        def combine_two_means_vars(x1,x2):
            (mean1,var1,n1) = x1
            (mean2,var2,n2) = x2
            mean = mean1*n1/(n1+n2)+ mean2*n2/(n1+n2)
            var = var1*n1/(n1+n2)+ var2*n2/(n1+n2)+n1*n2/(n1+n2)/(n1+n2)*np.power(mean1-mean2,2)
            return (mean, var, (n1+n2))

        def combine_all_means_vars(means,vars,lengths):
            inputs = list(zip(means,vars,lengths))
            result = inputs[0]

            for inputs2 in inputs[1:]:
                result= combine_two_means_vars(result,inputs2)
            return result

        means = list(df[col+"_mean"])
        vars = np.power(list(df[col+"_std"]),2)
        lengths = list(df[col+"_length"])
        mean,var,length = combine_all_means_vars(means,vars,lengths)
        return mean, np.sqrt(var)

def save_dataset(df,lbl_itos,mean,std,target_root,filename_postfix="",protocol=4):
    target_root = Path(target_root)
    df.to_pickle(target_root/("df"+filename_postfix+".pkl"), protocol=protocol)

    if(isinstance(lbl_itos,dict)):#dict as pickle
        outfile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "wb")
        pickle.dump(lbl_itos, outfile, protocol=protocol)
        outfile.close()
    else:#array
        np.save(target_root/("lbl_itos"+filename_postfix+".npy"),lbl_itos)

    np.save(target_root/("mean"+filename_postfix+".npy"),mean)
    np.save(target_root/("std"+filename_postfix+".npy"),std)


def filter_ptb_xl(
        df, min_cnt=10,
        categories=[
            "label_all", "label_diag", "label_form", "label_rhythm", "label_diag_subclass", "label_diag_superclass"
        ]
):
    # filter labels
    def select_labels(labels, min_cnt=10):
        lbl, cnt = np.unique([item for sublist in list(labels)
                              for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt >= min_cnt)[0]])

    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl = {}
    for selection in categories:
        label_selected = select_labels(df_ptb_xl[selection], min_cnt=min_cnt)
        df_ptb_xl[selection + "_filtered"] = df_ptb_xl[selection].apply(
            lambda x: [y for y in x if y in label_selected])
        lbl_itos_ptb_xl[selection] = np.array(
            list(set([x for sublist in df_ptb_xl[selection + "_filtered"] for x in sublist])))
        lbl_stoi = {s: i for i, s in enumerate(lbl_itos_ptb_xl[selection])}
        df_ptb_xl[selection + "_filtered_numeric"] = df_ptb_xl[selection +
                                                               "_filtered"].apply(lambda x: [lbl_stoi[y] for y in x])
    return df_ptb_xl, lbl_itos_ptb_xl


def resample_data(sigbufs, channel_labels, fs, target_fs, channels=8, channel_stoi=None, skimage_transform=True,
                  interpolation_order=3):
    channel_labels = [c.lower() for c in channel_labels]
    # https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs / fs
    timesteps_new = int(len(sigbufs) * factor)
    if (channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i, cl in enumerate(channel_labels):
            if (cl in channel_stoi.keys() and channel_stoi[cl] < channels):
                if (skimage_transform):
                    data[:, channel_stoi[cl]] = transform.resize(
                        sigbufs[:, i], (timesteps_new,), order=interpolation_order).astype(np.float32)
                else:
                    data[:, channel_stoi[cl]] = zoom(
                        sigbufs[:, i], timesteps_new / len(sigbufs), order=interpolation_order).astype(np.float32)
    else:
        if (skimage_transform):
            data = transform.resize(
                sigbufs, (timesteps_new, channels), order=interpolation_order).astype(np.float32)
        else:
            data = zoom(sigbufs, (timesteps_new / len(sigbufs), 1),
                        order=interpolation_order).astype(np.float32)
    return data


def prepare_data_ptb_xl(
        data_path, min_cnt=50, target_fs=100, channels=8, channel_stoi=channel_stoi_default,
        target_folder=None, skimage_transform=True, recreate_data=True
):
    target_root_ptb_xl = Path(".") if target_folder is None else target_folder
    # print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if (recreate_data is True):
        # reading df
        ptb_xl_csv = data_path / "ptbxl_database.csv"
        df_ptb_xl = pd.read_csv(ptb_xl_csv, index_col="ecg_id")
        # print(df_ptb_xl.columns)
        df_ptb_xl.scp_codes = df_ptb_xl.scp_codes.apply(
            lambda x: eval(x.replace("nan", "np.nan")))

        # preparing labels
        ptb_xl_label_df = pd.read_csv(data_path / "scp_statements.csv")
        ptb_xl_label_df = ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

        ptb_xl_label_diag = ptb_xl_label_df[ptb_xl_label_df.diagnostic > 0]
        ptb_xl_label_form = ptb_xl_label_df[ptb_xl_label_df.form > 0]
        ptb_xl_label_rhythm = ptb_xl_label_df[ptb_xl_label_df.rhythm > 0]

        diag_class_mapping = {}
        diag_subclass_mapping = {}
        for id, row in ptb_xl_label_diag.iterrows():
            if (isinstance(row["diagnostic_class"], str)):
                diag_class_mapping[id] = row["diagnostic_class"]
            if (isinstance(row["diagnostic_subclass"], str)):
                diag_subclass_mapping[id] = row["diagnostic_subclass"]

        df_ptb_xl["label_all"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys()])
        df_ptb_xl["label_diag"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
        df_ptb_xl["label_form"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
        df_ptb_xl["label_rhythm"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

        df_ptb_xl["label_diag_subclass"] = df_ptb_xl.label_diag.apply(
            lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
        df_ptb_xl["label_diag_superclass"] = df_ptb_xl.label_diag.apply(
            lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])

        df_ptb_xl["dataset"] = "ptb_xl"
        # filter (can be reapplied at any time)
        df_ptb_xl, lbl_itos_ptb_xl = filter_ptb_xl(df_ptb_xl, min_cnt=min_cnt)

        filenames = []
        for id, row in tqdm(list(df_ptb_xl.iterrows())):
            filename = data_path / \
                       row["filename_lr"] if target_fs <= 100 else data_path / \
                                                                   row["filename_hr"]
            sigbufs, header = wfdb.rdsamp(str(filename))
            data = resample_data(sigbufs=sigbufs, channel_stoi=channel_stoi,
                                 channel_labels=header['sig_name'], fs=header['fs'], target_fs=target_fs,
                                 channels=channels, skimage_transform=skimage_transform)
            assert (target_fs <= header['fs'])
            np.save(target_root_ptb_xl / (filename.stem + ".npy"), data)
            filenames.append(Path(filename.stem + ".npy"))
        df_ptb_xl["data"] = filenames

        # add means and std
        dataset_add_mean_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        # dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        # dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        # save means and stds
        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        # save
        save_dataset(df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl,
                     std_ptb_xl, target_root_ptb_xl)
    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(
            target_root_ptb_xl, df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl


if __name__ == '__main__':
    path_ptb_xl = os.path.join(PATH_BASE, DIR_DSET, config('datasets.PTB_XL.dir_nm'))
    # data_folder_ptb_xl = data_root / "ptb_xl/"
    # target_folder_ptb_xl = target_root / ("ptb_xl_fs" + str(target_fs))
    path_target = os.path.join(PATH_BASE, DIR_DSET, config('datasets.PTB_XL.dir_nm')) + '_temp'
    df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(
        path_ptb_xl, min_cnt=0,
        target_fs=500,
        channels=12,
        channel_stoi=channel_stoi_default,
        target_folder=path_target
    )
