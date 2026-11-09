"""
src/sisfall.py
--------------
Loader for SisFall (Sucerquia et al., Sensors 2017) – used as an EXTERNAL
test set: different people (incl. 15 elderly adults aged 60–75), a different
device (waist-worn ADXL345 + ITG3200, not a phone) and a different body
position (waist instead of trouser pocket).

File format: <CODE>_<SUBJECT>_<TRIAL>.txt, 200 Hz, 9 integer columns per line
ending in ';':
    ADXL345 acc x,y,z (13 bit, ±16 g) | ITG3200 gyro x,y,z (16 bit, ±2000 °/s) |
    MMA8451Q acc x,y,z (unused)
Conversion from the dataset Readme: value = (2 * range / 2**resolution) * bits.

Codes: F01–F15 falls, D01–D19 activities of daily living.
SA01–SA23 young adults (19–30), SE01–SE15 elderly (60–75; only SE06 performed falls).

Download (manual, free): https://www.mdpi.com/1424-8220/17/1/198 → Supplementary Materials.
"""

import os
import re

import numpy as np

from src.preprocess import Recording
from src.utils.logger import get_logger

logger = get_logger(__name__)

SISFALL_RATE_HZ = 200
_G = 9.80665
ACC_SCALE = (2 * 16 / 2 ** 13) * _G       # ADXL345 bits → m/s²
GYRO_SCALE = 2 * 2000 / 2 ** 16           # ITG3200 bits → °/s
_FILE = re.compile(r"^(?P<code>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")


def parse_sisfall_file(path: str) -> np.ndarray | None:
    """Return (T, 6) [acc m/s², gyro °/s] at 200 Hz, or None if unreadable."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().rstrip(";").split(",")
            if len(parts) < 6:
                continue
            try:
                rows.append([float(p) for p in parts[:6]])
            except ValueError:
                continue
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    arr[:, 0:3] *= ACC_SCALE
    arr[:, 3:6] *= GYRO_SCALE
    return arr


def downsample(data: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """
    Integer-factor downsampling with a box (moving-average) anti-aliasing filter:
    average every `src_hz // dst_hz` samples. Plain sample-dropping would alias
    the 200 Hz impact spikes.
    """
    factor = src_hz // dst_hz
    n = len(data) // factor * factor
    return data[:n].reshape(-1, factor, data.shape[1]).mean(axis=1)


def load_sisfall_recordings(root: str, config) -> list[Recording]:
    rate = config.signal.sample_rate_hz
    recordings, skipped = [], 0
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            m = _FILE.match(fname)
            if not m:
                continue
            raw = parse_sisfall_file(os.path.join(dirpath, fname))
            if raw is None or len(raw) < SISFALL_RATE_HZ:
                skipped += 1
                continue
            data = downsample(raw, SISFALL_RATE_HZ, rate).astype(np.float32)
            label = 1 if m["code"].startswith("F") else 0
            recordings.append(Recording(m["code"], m["subject"], m["trial"], label, data))
    logger.info(f"Loaded {len(recordings)} SisFall trials ({skipped} skipped)")
    return recordings
