# Notes: `src/preprocess.py`

## Purpose
Handles the entire data pipeline from raw dataset to windowed numpy arrays ready for model training. This module is a direct refactor of Notebook Cell 0.

## Functions

### `download_dataset(config) → str`
Downloads MobiFall v2.0 from Kaggle using `kagglehub`.
- Returns the local path to the downloaded dataset folder.
- Requires Kaggle API credentials configured (`~/.kaggle/kaggle.json`).

### `parse_txt_file(filepath) → pd.DataFrame | None`
The MobiFall `.txt` files contain header lines mixed with data lines. This function:
1. Reads every line in the file.
2. Tries to split by `,` and parse as 6 floats.
3. **Only keeps rows with exactly 6 numeric values.**
4. Returns `None` if no valid rows exist (never raises).

**Output columns:** `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`

### `create_windows(data, label, window_size=128, step=64) → (X, y)`
Core sliding window algorithm.

```
data: shape (T, 6) — raw sensor data for one activity file
                 │
    ┌────────────▼─────────────┐
    │  Window 1: rows [0:128]  │ → label copy
    └──────────────────────────┘
         ┌────────────────────────────────┐
         │  Window 2: rows [64:192]       │  (step=64, 50% overlap)
         └────────────────────────────────┘
                   ...
```

- Output `X` shape: `(N_windows, 128, 6)` — dtype `float32`
- Output `y` shape: `(N_windows,)` — dtype `int32`
- If `len(data) < window_size`, returns empty arrays (no crash).

### `load_and_preprocess(config, dataset_root=None) → (X, y)`
Orchestrates the full pipeline:
1. Downloads dataset if `dataset_root=None`.
2. Walks directory tree looking for folders named in `config.data.fall_classes` or `config.data.adl_classes`.
3. Calls `parse_txt_file()` + `create_windows()` for each valid file.
4. Concatenates all windows and returns final `(X, y)`.

**Label mapping:**
- `fall_classes` folder → label `1` (FALL)
- `adl_classes` folder → label `0` (Normal)

### `save_processed(X, y, config)` / `load_processed(config)`
Serialize/deserialize the windowed arrays to `.npy` format for fast reloading without re-running the full pipeline.

```bash
# First run: ~minutes (download + parse + window)
python src/preprocess.py

# Subsequent runs: ~seconds
# Just call load_processed(config) in train.py
```

### `generate_dummy_data(config) → (X, y)`
Creates `(200, 128, 6)` random arrays for dry-runs and CI testing. **No internet required.**

## CLI Usage
```bash
python src/preprocess.py              # Download + process full dataset
python src/preprocess.py --dry-run   # Use dummy data (no download)
python src/preprocess.py --local-data /path/to/mobifall  # Skip download
python src/preprocess.py --config tests/test_config.yaml  # Custom config
```

## MobiFall Directory Structure Expected
```
mobifall_dataset_v20/
├── FOL/      ← fall class (Forward Fall)
│   ├── sub1/
│   │   └── acc_FOL_sub1_1.txt
├── STD/      ← normal class (Standing)
│   └── ...
```

## Common Issues
| Issue | Cause | Fix |
|---|---|---|
| No valid data found | Wrong folder structure | Print `os.walk()` to debug paths |
| Kaggle download fails | No API key | Add `~/.kaggle/kaggle.json` |
| File skipped (too short) | Recording < 128 samples | Reduce `window_size` or ignore |
