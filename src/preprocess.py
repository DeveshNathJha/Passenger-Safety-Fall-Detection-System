"""
src/preprocess.py
-----------------
Data acquisition, parsing, and sliding-window segmentation
for the MobiFall v2.0 dataset.

Standalone usage:
    python src/preprocess.py                     # download + process
    python src/preprocess.py --dry-run           # validate logic on dummy data
    python src/preprocess.py --local-data data/raw  # skip download, use local files

Outputs:
    data/processed/X_windows.npy  – shape (N, 128, 6)
    data/processed/y_labels.npy   – shape (N,)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Add project root to path so sibling modules resolve correctly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Dataset Download
# ---------------------------------------------------------------------------

def download_dataset(config) -> str:
    """
    Download the MobiFall v2.0 dataset from Kaggle via kagglehub.

    Returns
    -------
    str
        Absolute path to the downloaded dataset directory.
    """
    try:
        import kagglehub  # type: ignore
    except ImportError:
        logger.error("kagglehub is not installed. Run: pip install kagglehub")
        raise

    logger.info(f"Downloading dataset '{config.data.dataset_id}' from Kaggle …")
    path = kagglehub.dataset_download(config.data.dataset_id)
    logger.info(f"Dataset downloaded to: {path}")
    return path


# ---------------------------------------------------------------------------
# 2. File Parsing
# ---------------------------------------------------------------------------

def parse_txt_file(filepath: str) -> pd.DataFrame | None:
    """
    Parse a MobiFall .txt sensor file into a tidy DataFrame.

    The files contain mixed header lines and numeric data rows.
    Only rows with exactly 6 numeric columns are retained.

    Parameters
    ----------
    filepath : str
        Absolute path to the .txt file.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
        or None if no valid rows are found.
    """
    valid_rows = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                try:
                    row = [float(p) for p in parts]
                    valid_rows.append(row)
                except ValueError:
                    continue
    except OSError as exc:
        logger.warning(f"Cannot open file {filepath}: {exc}")
        return None

    if not valid_rows:
        return None

    df = pd.DataFrame(
        valid_rows,
        columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    return df


# ---------------------------------------------------------------------------
# 3. Sliding Window Segmentation
# ---------------------------------------------------------------------------

def create_windows(
    data: np.ndarray,
    label: int,
    window_size: int = 128,
    step: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply sliding-window segmentation to a single activity's sensor data.

    Parameters
    ----------
    data : np.ndarray
        Raw sensor array of shape (T, 6).
    label : int
        Class label for this activity (0 = Normal, 1 = Fall).
    window_size : int
        Number of timesteps per window (default 128).
    step : int
        Stride between consecutive windows (default 64 → 50% overlap).

    Returns
    -------
    (X_windows, y_labels) : (np.ndarray, np.ndarray)
        X shape: (N_windows, window_size, 6)
        y shape: (N_windows,)
    """
    windows, labels = [], []
    n = len(data)
    start = 0
    while start + window_size <= n:
        windows.append(data[start : start + window_size])
        labels.append(label)
        start += step

    if not windows:
        return np.empty((0, window_size, 6)), np.empty((0,), dtype=int)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# 4. Full Pipeline Orchestration
# ---------------------------------------------------------------------------

def load_and_preprocess(config, dataset_root: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk the MobiFall directory tree, parse every .txt file, apply
    sliding-window segmentation, and return (X, y, subjects) arrays.

    Parameters
    ----------
    config : AttrDict
        Loaded application config.
    dataset_root : str | None
        Root directory of the dataset. If None, downloads using kagglehub.

    Returns
    -------
    (X, y, subjects) : (np.ndarray, np.ndarray, np.ndarray)
        X shape: (N, 128, 6)   – float32
        y shape: (N,)          – int32  (0 = Normal, 1 = Fall)
        subjects shape: (N,)   – object (e.g. 'sub01')
    """
    if dataset_root is None:
        dataset_root = download_dataset(config)

    fall_classes = set(config.data.fall_classes)
    adl_classes = set(config.data.adl_classes)
    window_size = config.windowing.window_size
    step = int(window_size * (1 - config.windowing.overlap))

    all_X, all_y, all_subjects = [], [], []
    skipped = 0
    import re

    for root, dirs, files in os.walk(dataset_root):
        # Determine label from parent folder name
        folder_name = os.path.basename(root).upper()
        if folder_name in fall_classes:
            label = 1
        elif folder_name in adl_classes:
            label = 0
        else:
            continue

        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            df = parse_txt_file(fpath)
            if df is None or len(df) < window_size:
                skipped += 1
                logger.debug(f"Skipped (too short or unreadable): {fpath}")
                continue

            # Extract subject ID (e.g. sub01 from BSC_01_sub01.txt)
            match = re.search(r"sub\d+", fname.lower())
            sub_id = match.group(0) if match else "unknown"

            X_w, y_w = create_windows(df.values, label, window_size, step)
            all_X.append(X_w)
            all_y.append(y_w)
            all_subjects.append(np.array([sub_id] * len(y_w), dtype=object))

    if not all_X:
        raise RuntimeError(
            "No valid data found. Check dataset path and folder structure."
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)

    logger.info(
        f"Preprocessing complete — Windows: {X.shape[0]:,} | "
        f"Falls: {(y == 1).sum():,} | Normal: {(y == 0).sum():,} | "
        f"Unique Subjects: {len(np.unique(subjects))} | "
        f"Skipped files: {skipped}"
    )
    return X, y, subjects


# ---------------------------------------------------------------------------
# 5. Save Processed Arrays
# ---------------------------------------------------------------------------

def save_processed(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, config) -> None:
    """
    Save windows, labels, and subjects to disk as .npy files.
    """
    os.makedirs(config.data.processed_dir, exist_ok=True)
    np.save(config.data.processed_X, X)
    np.save(config.data.processed_y, y)
    np.save(config.data.processed_subjects, subjects)
    logger.info(
        f"Saved → {config.data.processed_X} {X.shape} | "
        f"{config.data.processed_y} {y.shape} | "
        f"{config.data.processed_subjects} {subjects.shape}"
    )


def load_processed(config, return_subjects: bool = False) -> tuple:
    """
    Load previously saved .npy window files from disk.
    """
    X = np.load(config.data.processed_X)
    y = np.load(config.data.processed_y)
    if return_subjects:
        subjects = np.load(config.data.processed_subjects, allow_pickle=True)
        logger.info(f"Loaded processed data — X: {X.shape}, y: {y.shape}, subjects: {subjects.shape}")
        return X, y, subjects
    logger.info(f"Loaded processed data — X: {X.shape}, y: {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# 6. Dry-Run Helper (for testing without downloading data)
# ---------------------------------------------------------------------------

def generate_dummy_data(config, return_subjects: bool = False) -> tuple:
    """
    Generate random dummy windows for testing inference / training pipelines
    without requiring the real dataset.

    Returns
    -------
    (X, y) or (X, y, subjects) depending on return_subjects
    """
    rng = np.random.default_rng(config.model.random_state)
    n_windows = 200
    X = rng.random((n_windows, config.windowing.window_size, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=n_windows).astype(np.int32)
    logger.info(f"Generated {n_windows} dummy windows for dry-run.")
    if return_subjects:
        # Generate 10 mock subject IDs
        subjects = np.array([f"sub{rng.integers(1, 11):02d}" for _ in range(n_windows)], dtype=object)
        return X, y, subjects
    return X, y


# ---------------------------------------------------------------------------
# 7. CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="MobiFall preprocessing pipeline"
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to local MobiFall dataset root (skips Kaggle download).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip data loading; generate dummy windows and save to processed/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)

    if args.dry_run:
        logger.info("DRY RUN — generating dummy data.")
        X, y, subjects = generate_dummy_data(config, return_subjects=True)
    else:
        X, y, subjects = load_and_preprocess(config, dataset_root=args.local_data)

    save_processed(X, y, subjects, config)
    logger.info("Preprocessing pipeline finished successfully.")


if __name__ == "__main__":
    main()
