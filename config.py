from icecream import ic


config = dict(
    datasets={
        'MIT-BIH-MVED': 'MIT-BIH Malignant Ventricular Ectopy Database'
    }
)

if __name__ == "__main__":
    import json

    fl_nm = 'config.json'
    ic(config)
    with open(fl_nm, 'w') as f:
        json.dump(config, f, indent=4)
