# VESTA: In-Vehicle Sensing with GNSS Satellite Signals

## Environment

```bash
conda env create -f environment.yml
conda activate in-vehicle-gnss
```

## Layout

```
.
├── environment.yml
├── model/
│   ├── config/gnss_transformer.json
│   ├── dataloader.py
│   ├── transformer.py
│   └── utils.py
├── script/
│   ├── vehicle_interference_cancellation.py
│   ├── constants.py
│   ├── train_gnss_transformer.py
│   ├── train_gnss_autogluon.py
│   └── train_imu_autogluon.py
├── data/demo_data.json
└── figures/
```

## Vehicle interference cancellation

```bash
python script/vehicle_interference_cancellation.py
```

| Argument | Default |
|----------|---------|
| `--data` | `data/demo_data.json` |
| `--output` | `figures/` |


## Multi-satellite fusion network

LSTM per satellite + direction embedding (azimuth/elevation) + cross-satellite attention + parallel heads for multi-target labels (`gt_label[0]`, `gt_label[1]`, …).

```bash
python script/train_gnss_transformer.py
```

Config: `model/config/gnss_transformer.json`.

## AutoGluon baselines

```bash
python script/train_gnss_autogluon.py --overwrite
python script/train_imu_autogluon.py --overwrite
```

- GNSS: same `dataset/GNSS/` layout; features are amp/phase stats + elevation/azimuth for top-`k` satellites (`--top-k`, default 8).
- IMU: `dataset/IMU/**/imu.json` with `time_ms_from_start`, `acc_x/y/z`, `gyro_x/y/z`, `label`.

## Activity classes

Shared order in `model/utils.py` (id = index):

`Default`, `Push hand`, `Nod`, `Turn head`, `Touch`, `Push twice`, `Circle`, `Swipe`, `Pick up`
