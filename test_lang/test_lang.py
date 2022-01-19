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

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # # sns.set_theme(style="whitegrid")
    # # tips = sns.load_dataset("tips")
    # # ax = sns.barplot(x="day", y="total_bill", data=tips)
    # # ax = sns.barplot(x="size", y="total_bill", data=tips, palette="Blues_d")
    # import numpy as np
    # # x = np.array([1, 2, 3])
    # # y = np.array([67, 23, 1])
    # x = np.arange(12)+1
    # y = np.array([304.08994, 229.13878, 173.71886, 135.75499,
    #                111.096794, 94.25109, 81.55578, 71.30187,
    #                62.146603, 54.212032, 49.20715, 46.765743])
    # # ax = sns.barplot(x=x, y=y, palette='Blues_d')
    # ax = sns.barplot(x=x, y=y, palette='flare')
    #
    # ic(np.power(np.array([0, 1., 6]), -2))

    # # Taken from https://github.com/matplotlib/matplotlib/issues/19256
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.widgets import Slider
    #
    # fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    # ax = plt.gca()
    # plt.subplots_adjust(top=0.975, left=0.05, right=0.95, bottom=0.125)
    # ax_slider = plt.axes([0.125, 0.05, 0.75, 0.01])
    # slider = Slider(ax_slider, 'Freq', 0.1, 30.0, valinit=5, valstep=0.5)
    # ic(slider.vline, vars(slider.vline))
    # slider.vline._linewidth = 0  # Hides vertical red line marking init value
    #
    # t = np.arange(0.0, 1.0, 0.001)
    # f0 = 3
    # delta_f = 5.0
    # amp = 5
    # s = amp * np.sin(2 * np.pi * f0 * t)
    # l, = ax.plot(t, s, lw=2)
    #
    # def update(val):
    #     freq = slider.val
    #     ic(val, slider.val)
    #     l.set_ydata(amp * np.sin(2 * np.pi * freq * t))
    #     # ax.figure.canvas.draw_idle()
    # slider.on_changed(update)
    # # plt.show()
    #
    # # ic(np.linalg.norm(np.full(32, 0.1)))
    #
    # gen = (i for i in range(5))
    # ic(next(gen))

    # from gibberish import Gibberish
    # import random
    # random.seed(7)
    # gib = Gibberish()
    # ic(gib.generate_word())
    # ic(gib.generate_words(8))

    import numpy as np
    f = 0.23423423424124
    f_ = np.float64(f)
    ic(f, round(f, 5), type(f_), round(f_, 5))
