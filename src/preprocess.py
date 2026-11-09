"""
src/preprocess.py
-----------------
Data acquisition, parsing, synchronisation and windowing for:

  human : MobiFall v2.0 (real smartphone recordings, 24 subjects)
  car   : synthetic crash dataset (scripts/generate_car_crash_data.py)

MobiFall stores accelerometer and gyroscope in SEPARATE files with their own
nanosecond timestamps and different, irregular rates (~87 Hz acc, ~200 Hz
gyro). This pipeline:

  1. parses `<CODE>_acc_<subject>_<trial>.txt` and the matching `_gyro_` file
  2. aligns both on their common time span and resamples to 50 Hz
     (the rate the phone demo streams at) by linear interpolation
  3. converts gyroscope rad/s → °/s (browser DeviceMotion reports °/s)
  4. ADL trials → every window labelled 0
     fall trials → only windows containing the impact peak labelled 1
  5. keeps the subject ID of every window for subject-wise splitting

Standalone usage:
    python src/preprocess.py --task human                 # download (kagglehub) + process
    python src/preprocess.py --task human --local-data DIR
    python src/preprocess.py --task car                   # generates synthetic data if missing

Output:
    data/processed/<task>_windows.npz   X (N,128,6) float32 | y (N,) int32 | groups (N,) str
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.signal_processing import clip_to_sensor_range, find_impact_index, resample_uniform, sliding_windows
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MOBIFALL_FILE = re.compile(r"^(?P<code>[A-Z]{3})_(?P<sensor>acc|gyro|ori)_(?P<subject>\d+)_(?P<trial>\d+)\.txt$")


# ---------------------------------------------------------------------------
# 1. Dataset Download
# ---------------------------------------------------------------------------

def download_dataset(config) -> str:
    """Download MobiFall v2.0 from Kaggle via kagglehub and return its local path."""
    try:
        import kagglehub  # type: ignore
    except ImportError:
        logger.error("kagglehub is not installed. Run: pip install kagglehub")
        raise

    logger.info(f"Downloading dataset '{config.data.dataset_id}' from Kaggle …")
    path = kagglehub.dataset_download(config.data.dataset_id)
    logger.info(f"Dataset available at: {path}")
    return path


# ---------------------------------------------------------------------------
# 2. File Parsing
# ---------------------------------------------------------------------------

def parse_sensor_file(filepath: str) -> Optional[np.ndarray]:
    """
    Parse one MobiFall sensor file.

    Lines starting with '#' or '@' are headers; data lines are
    ``timestamp_ns, x, y, z``.

    Returns
    -------
    np.ndarray | None
        Shape (T, 4): [t_seconds, x, y, z], or None if no valid rows.
    """
    rows = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in "#@":
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 4:
                    continue
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    except OSError as exc:
        logger.warning(f"Cannot open file {filepath}: {exc}")
        return None

    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    arr[:, 0] *= 1e-9  # ns → s
    return arr


def merge_acc_gyro(acc: np.ndarray, gyro: np.ndarray, rate_hz: float) -> Optional[np.ndarray]:
    """
    Time-align accelerometer and gyroscope streams and resample both to rate_hz.

    Parameters
    ----------
    acc, gyro : np.ndarray
        Outputs of parse_sensor_file (acc in m/s², gyro in rad/s).

    Returns
    -------
    np.ndarray | None
        (T, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] in m/s² and °/s,
        or None if the streams do not overlap.
    """
    t0 = max(acc[0, 0], gyro[0, 0])
    t1 = min(acc[-1, 0], gyro[-1, 0])
    if t1 - t0 <= 0:
        return None
    _, a = resample_uniform(acc[:, 0], acc[:, 1:4], rate_hz, t0, t1)
    _, g = resample_uniform(gyro[:, 0], gyro[:, 1:4], rate_hz, t0, t1)
    n = min(len(a), len(g))
    return np.hstack([a[:n], np.rad2deg(g[:n])]).astype(np.float32)


@dataclass
class Recording:
    code: str
    subject: str
    trial: str
    label: int
    data: np.ndarray  # (T, 6) at the target rate


def find_mobifall_root(path: str) -> str:
    for root, dirs, _ in os.walk(path):
        if "MobiFall_Dataset_v2.0" in dirs:
            return os.path.join(root, "MobiFall_Dataset_v2.0")
    return path


def load_mobifall_recordings(dataset_root: str, config) -> list[Recording]:
    """Walk the MobiFall tree and return one synchronised Recording per trial."""
    fall_classes = set(config.data.fall_classes)
    adl_classes = set(config.data.adl_classes)
    rate = config.signal.sample_rate_hz
    recordings, skipped = [], 0

    for root, _, files in os.walk(find_mobifall_root(dataset_root)):
        for fname in sorted(files):
            m = _MOBIFALL_FILE.match(fname)
            if not m or m["sensor"] != "acc":
                continue
            code = m["code"]
            if code in fall_classes:
                label = 1
            elif code in adl_classes:
                label = 0
            else:
                continue

            gyro_path = os.path.join(root, fname.replace("_acc_", "_gyro_"))
            acc = parse_sensor_file(os.path.join(root, fname))
            gyro = parse_sensor_file(gyro_path) if os.path.exists(gyro_path) else None
            if acc is None or gyro is None or len(acc) < 10 or len(gyro) < 10:
                skipped += 1
                continue

            data = merge_acc_gyro(acc, gyro, rate)
            if data is None:
                skipped += 1
                continue
            recordings.append(Recording(code, f"sub{int(m['subject']):02d}", m["trial"], label, data))

    logger.info(f"Loaded {len(recordings)} MobiFall trials ({skipped} skipped: missing/unaligned gyro)")
    return recordings


# ---------------------------------------------------------------------------
# 3. Windowing
# ---------------------------------------------------------------------------

def windows_from_recordings(recordings: list[Recording], config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Window every recording. ADL trials: all windows → 0.
    Fall trials: impact-centred windows → 1 (pre-fall / post-fall windows dropped).
    """
    ws = config.windowing.window_size
    step = int(ws * (1 - config.windowing.overlap))
    margin = config.data.impact_margin
    all_X, all_y, all_g = [], [], []
    too_short = 0

    for rec in recordings:
        data = clip_to_sensor_range(rec.data, config.signal.acc_range, config.signal.gyro_range)
        if len(data) < ws:
            too_short += 1
            continue
        impact = find_impact_index(data) if rec.label == 1 else None
        X, y = sliding_windows(data, rec.label, ws, step, impact_idx=impact, impact_margin=margin)
        all_X.append(X)
        all_y.append(y)
        all_g.append(np.full(len(y), rec.subject))

    if not all_X:
        raise RuntimeError("No valid windows produced. Check the dataset path and folder structure.")
    if too_short:
        logger.info(f"{too_short} recordings shorter than one window were skipped")
    return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_g)


def load_car_dataset(config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Window the synthetic crash dataset. Normal sequences → 0; crash sequences:
    windows ending before the impact → 0 (it's still normal driving),
    impact-centred windows → 1, post-crash windows dropped.
    Groups are sequence IDs so overlapping windows never straddle splits.
    """
    path = resolve_path(config.data.car_raw_dir).with_suffix(".npz")
    if not path.exists():
        logger.info(f"{path} not found – generating synthetic crash data …")
        import subprocess
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_car_crash_data.py")],
                       check=True)

    npz = np.load(path, allow_pickle=False)
    data, offsets, impacts, kinds = npz["data"], npz["offsets"], npz["impact"], npz["kind"]
    ws = config.windowing.window_size
    step = int(ws * (1 - config.windowing.overlap))
    margin = config.data.impact_margin
    all_X, all_y, all_g = [], [], []

    for i, imp in enumerate(impacts):
        seq = data[offsets[i]:offsets[i + 1]]
        group = f"{kinds[i]}_{i:04d}"
        if imp < 0:
            X, y = sliding_windows(seq, 0, ws, step)
        else:
            Xc, yc = sliding_windows(seq, 1, ws, step, impact_idx=int(imp), impact_margin=margin)
            pre = seq[: max(0, int(imp) - margin)]
            Xn, yn = sliding_windows(pre, 0, ws, step) if len(pre) >= ws else (Xc[:0], yc[:0])
            X, y = np.concatenate([Xc, Xn]), np.concatenate([yc, yn])
        all_X.append(X)
        all_y.append(y)
        all_g.append(np.full(len(y), group))

    return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_g)


# ---------------------------------------------------------------------------
# 4. Save / Load
# ---------------------------------------------------------------------------

def processed_path(config, task: str) -> Path:
    return resolve_path(config.data.processed_dir) / f"{task}_windows.npz"


def save_processed(X: np.ndarray, y: np.ndarray, groups: np.ndarray, config, task: str) -> Path:
    path = processed_path(config, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, groups=groups.astype(str))
    logger.info(f"Saved → {path}  X{X.shape} | events: {int(y.sum()):,} | normal: {int((y == 0).sum()):,} | "
                f"groups: {len(np.unique(groups))}")
    return path


def load_processed(config, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = processed_path(config, task)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python src/preprocess.py --task {task}")
    npz = np.load(path, allow_pickle=False)
    logger.info(f"Loaded {path.name} — X{npz['X'].shape}")
    return npz["X"], npz["y"], npz["groups"]


def recordings_path(config, task: str) -> Path:
    return resolve_path(config.data.processed_dir) / f"{task}_recordings.npz"


def save_recordings(recordings: list[Recording], config, task: str) -> Path:
    """Store whole trials (not windows) for event-level evaluation by replay."""
    path = recordings_path(config, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(r.data) for r in recordings])])
    np.savez_compressed(
        path,
        data=np.concatenate([r.data for r in recordings]).astype(np.float32),
        offsets=offsets,
        code=np.array([r.code for r in recordings]),
        subject=np.array([r.subject for r in recordings]),
        trial=np.array([r.trial for r in recordings]),
        label=np.array([r.label for r in recordings], dtype=np.int32),
    )
    logger.info(f"Saved → {path}  ({len(recordings)} recordings)")
    return path


def load_recordings(config, task: str) -> list[Recording]:
    path = recordings_path(config, task)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python src/preprocess.py --task {task}")
    with np.load(path, allow_pickle=False) as npz:
        # read every array ONCE – npz["x"] decompresses the whole array on each access
        data, o = npz["data"], npz["offsets"]
        code, subject, trial, label = npz["code"], npz["subject"], npz["trial"], npz["label"]
    return [Recording(str(code[i]), str(subject[i]), str(trial[i]), int(label[i]), data[o[i]:o[i + 1]])
            for i in range(len(label))]


def build_dataset(config, task: str, local_data: Optional[str] = None):
    """Return (X, y, groups, recordings); recordings is None for the synthetic car task."""
    recordings = None
    if task == "human":
        root = local_data or download_dataset(config)
        recordings = load_mobifall_recordings(root, config)
        X, y, groups = windows_from_recordings(recordings, config)
    elif task == "sisfall":
        if not local_data:
            raise ValueError("SisFall must be downloaded manually; pass --local-data <SisFall folder>")
        from src.sisfall import load_sisfall_recordings
        recordings = load_sisfall_recordings(local_data, config)
        X, y, groups = windows_from_recordings(recordings, config)
    elif task == "car":
        X, y, groups = load_car_dataset(config)
    else:
        raise ValueError(f"Unknown task '{task}' (expected human | sisfall | car)")
    return X, y, groups, recordings


# ---------------------------------------------------------------------------
# 5. Dry-Run Helper (for tests / CI without the dataset)
# ---------------------------------------------------------------------------

def generate_dummy_data(config, n_windows: int = 240, n_groups: int = 8, seed: int = 0):
    """Random windows with a learnable signal (events have a large spike)."""
    rng = np.random.default_rng(seed)
    ws = config.windowing.window_size
    X = rng.normal(0, 1, (n_windows, ws, 6)).astype(np.float32)
    X[:, :, 1] += 9.81
    y = (np.arange(n_windows) % 3 == 0).astype(np.int32)
    X[y == 1, ws // 2, 0:3] += 40.0
    groups = np.array([f"sub{i % n_groups:02d}" for i in range(n_windows)])
    return X, y, groups


# ---------------------------------------------------------------------------
# 6. CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Build windowed datasets.")
    parser.add_argument("--task", choices=["human", "sisfall", "car"], default="human")
    parser.add_argument("--local-data", type=str, default=None, metavar="DIR",
                        help="Local dataset folder (MobiFall: skips Kaggle download; SisFall: required).")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    X, y, groups, recordings = build_dataset(config, args.task, args.local_data)
    save_processed(X, y, groups, config, args.task)
    if recordings is not None:
        save_recordings(recordings, config, args.task)


if __name__ == "__main__":
    main()
