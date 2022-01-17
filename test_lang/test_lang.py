from icecream import ic

from util import conc_map, sizeof_fmt

if __name__ == '__main__':
    # import numpy as np
    # from matplotlib import pyplot as plt
    #
    # duration = 4
    # x250 = np.arange(0, duration, 1 / 10)
    # x256 = np.arange(0, duration, 1 / 25)
    # vals = np.sin(x250)
    # vals_inter = np.interp(x256, x250, vals)
    # ic(x250, x256)
    # ic(vals.shape, vals_inter.shape)
    #
    # plt.figure(figsize=(16, 9))
    # plt.plot(x250, vals, marker='o', ms=4, lw=5, label='Original', alpha=0.5)
    # plt.plot(x256, vals_inter, marker='x', ms=4, lw=1, label='Interpolated')  # ls=(0, (2, 5)),
    # plt.title('sin wave resampled')
    # plt.legend()
    # plt.show()

    # import pandas as pd
    # df = pd.concat([pd.DataFrame([i], columns=['A']) for i in range(5)], ignore_index=True)
    # ic(df)

    # sub = '.mat'
    # s = 'E00002.mat 12 500 5000 05-May-2020 14:50:55'
    # idx = s.find(sub)
    # # ic(idx)
    # s_ = s[:idx] + s[idx+len(sub):]
    # # ic(idx + len(sub))
    # # ic(s[:idx], s[:idx+len(sub)], s[10:])
    # ic(s[:idx], s[idx+len(sub):])
    # ic(s, s_)

    # import numpy as np
    # arr = np.arange(5)
    # ic(arr)
    # ic(arr[1:-1])

    # import pandas as pd
    # df = pd.DataFrame([1, 2])
    # ic(df)

    # def alternate(lst1, lst2):
    #     lst = [None] * (len(lst1)+len(lst2))
    #     lst[::2] = lst1
    #     lst[1::2] = lst2
    #     return lst
    # ic(alternate(['f', 'o', 'o'], ['hello', 'world']))

    # import concurrent.futures
    # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #     # lst = [outcome for outcome in executor.map(lambda x: x**2, range(10))]
    #     lst = list(executor.map(lambda x: x**2, range(10)))
    # ic(lst, sum(lst) / 10)
    # ic(clst_map(lambda x: x**2, range(10)))
    # ic(list(clst_map(lambda x: x**2, range(10))))

    # fmt = '%s %s'
    # ic(fmt % (1, 2))

    # import datetime
    # ic(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # import os
    # sz = os.path.getsize('test_lang.py')
    # ic(sz, sizeof_fmt(sz))

    # from enum import Enum
    # CLP_CH = Enum('CaliperChange', 'Add Remove Edit')
    # ic(CLP_CH, vars(CLP_CH), CLP_CH.Add)

    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")
    # ax = sns.barplot(x="day", y="total_bill", data=tips)
    # ax = sns.barplot(x="size", y="total_bill", data=tips, palette="Blues_d")
    import numpy as np
    # x = np.array([1, 2, 3])
    # y = np.array([67, 23, 1])
    x = np.arange(12)+1
    y = np.array([304.08994, 229.13878, 173.71886, 135.75499,
                   111.096794, 94.25109, 81.55578, 71.30187,
                   62.146603, 54.212032, 49.20715, 46.765743])
    # ax = sns.barplot(x=x, y=y, palette='Blues_d')
    ax = sns.barplot(x=x, y=y, palette='flare')

    import scipy.optimize

    def r2(y, y_fit):
        return 1 - (np.square(y - y_fit).sum() / np.square(y - np.mean(y)).sum())

    def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        def pow_law(x_, a, b):
            return a * np.power(x_, b)
        x, y = np.asarray(x).astype(float), np.asarray(y)
        ic(x, pow_law(x, x[0], -1))
        (a_, b_), p_cov = scipy.optimize.curve_fit(f=pow_law, xdata=x, ydata=y, p0=(x[0], -1))
        x_plot = np.linspace(x.min(), x.max(), num=x.size)
        ic(x_plot)
        y_fit = pow_law(x_plot, a_, b_)
        ic(y_fit)
        plt.plot(x_plot, y_fit, label='Fitted power law', lw=2)
        return a_, b_
    fit_power_law(x, y)
    plt.show()
    ic(np.power(np.array([0, 1., 6]), -2))


