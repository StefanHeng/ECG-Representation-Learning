from icecream import ic


config = dict(
    datasets=dict(
        BIH_MVED=dict(
            dir_nm='MIT-BIH-MVED',
            nm='MIT-BIH Malignant Ventricular Ectopy Database',
            fqs=257
        ),
        INCART=dict(
            dir_nm='St-Petersburg-INCART',
            nm='St Petersburg INCART 12-lead Arrhythmia Database'
        )
    )
)

if __name__ == '__main__':
    import json
    from data_path import *

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)
