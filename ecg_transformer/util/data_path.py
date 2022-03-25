import os

DIR_DSET = 'datasets'  # Dataset root folder name

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

PATH_BASE = os.sep.join(paths[:-2])
DIR_PROJ = paths[-2]
PKG_NM = paths[-1]  # Package/Module name, e.g. `zeroshot_encoder`

DIR_MDL = 'models'  # Save models


if __name__ == '__main__':
    from icecream import ic
    ic(PATH_BASE, type(PATH_BASE), DIR_PROJ, PKG_NM)
