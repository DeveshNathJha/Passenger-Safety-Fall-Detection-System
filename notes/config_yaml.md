# Notes: `config/config.yaml`

## Purpose
Central configuration file for the entire project. **All magic numbers live here.** No source file should have hardcoded values — they must always reference this file via `load_config()`.

## Sections

### `data`
Controls dataset sourcing and directory layout.

| Key | Value | Meaning |
|---|---|---|
| `dataset_id` | `"kmknation/mobifall-dataset-v20"` | Kaggle dataset identifier used by `kagglehub` |
| `raw_dir` | `"data/raw"` | Where downloaded files land (gitignored) |
| `processed_dir` | `"data/processed"` | Where `.npy` windowed arrays are saved |
| `processed_X` | `"data/processed/X_windows.npy"` | Numpy array path for features |
| `processed_y` | `"data/processed/y_labels.npy"` | Numpy array path for labels |
| `fall_classes` | `["FOL","FKL","BSC","SDL"]` | MobiFall folder names that map to label=1 |
| `adl_classes` | `["STD","WAL","JOG",...]` | Normal activities → label=0 |
| `sensor_columns` | `["acc_x"..."gyro_z"]` | Column order expected in CSV/inference |

### `windowing`
Controls the sliding window algorithm in `preprocess.py` and `inference.py`.

| Key | Value | Meaning |
|---|---|---|
| `window_size` | `128` | Samples per window. At 50 Hz = 2.56 seconds of data |
| `overlap` | `0.5` | 50% overlap → `step = 64` samples between windows |

> **Why 128?** It's long enough to capture a full fall event (~2.5 s) but short enough for real-time inference. It's also a power-of-2 which is efficient for convolution.

### `model`
Training and inference hyperparameters.

| Key | Value | Meaning |
|---|---|---|
| `selected_architecture` | `"1D-CNN"` | Default model to train/deploy |
| `epochs` | `10` | Training epochs |
| `batch_size` | `64` | Minibatch size |
| `test_size` | `0.2` | 80/20 train-test split |
| `random_state` | `42` | Reproducibility seed |
| `early_stopping_patience` | `5` | Stop if val_loss doesn't improve for 5 epochs |
| `prediction_threshold` | `0.5` | Sigmoid output above which → FALL |
| `labels` | `{0: "Normal", 1: "FALL"}` | Human-readable class names |

### `paths`
Canonical file paths used across all modules.

| Key | File |
|---|---|
| `model_h5` | `models/mobifall_model.h5` – full Keras model |
| `model_tflite` | `models/mobifall_edge_model.tflite` – quantized edge model |
| `logs_dir` | `logs/` – log directory |
| `log_file` | `logs/app.log` – primary log file |

### `api`
FastAPI server configuration.

| Key | Value | Meaning |
|---|---|---|
| `host` | `"0.0.0.0"` | Listen on all interfaces (LAN accessible) |
| `port` | `8000` | HTTP/WebSocket port |
| `sensor_frequency_hz` | `50` | Expected sensor sampling rate |

### `logging`
Controls logger behaviour.

| Key | Value |
|---|---|
| `level` | `"INFO"` (DEBUG/INFO/WARNING/ERROR) |
| `max_bytes` | `5242880` (5 MB per log file) |
| `backup_count` | `3` (keep last 3 rotated files) |

## How to Override per Module
```python
from src.utils.config_loader import load_config
config = load_config()        # loads config/config.yaml
config = load_config("/path/to/other.yaml")  # override for testing
```

## How to Extend
Add any new parameter here rather than hardcoding it. For example, if you add a new preprocessing step:
```yaml
windowing:
  window_size: 128
  overlap: 0.5
  normalize: true          # ← add here, read in preprocess.py
```
