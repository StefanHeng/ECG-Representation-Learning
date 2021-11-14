from icecream import ic


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

    sub = '.mat'
    s = 'E00002.mat 12 500 5000 05-May-2020 14:50:55'
    idx = s.find(sub)
    # ic(idx)
    s_ = s[:idx] + s[idx+len(sub):]
    # ic(idx + len(sub))
    # ic(s[:idx], s[:idx+len(sub)], s[10:])
    ic(s[:idx], s[idx+len(sub):])
    ic(s, s_)

