"""
scripts/generate_mobifall_synthetic.py
--------------------------------------
Generates a high-quality synthetic IMU dataset that mimics the MobiFall v2.0
dataset's PHYSICAL sensor characteristics:

  - Phone in trouser pocket -> gravity on Y-axis (~9.8 m/s²)
  - Accelerometer in m/s²
  - Gyroscope in deg/s
  - Sampled at 50 Hz (128-sample windows)

Generates:
  - 1200 Normal Activity windows (label=0) — Walking, Standing, Jogging, Stairs
  - 800 Fall windows (label=1) — Forward fall, Backward fall, Sideways fall, Trip

Output:
  data/processed/X_windows.npy  — shape (2000, 128, 6)
  data/processed/y_labels.npy   — shape (2000,)
  data/processed/subjects.npy   — shape (2000,) mock subject IDs
"""

import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

RNG = np.random.default_rng(42)
WINDOW_SIZE = 128
HZ = 50
T = np.linspace(0, WINDOW_SIZE / HZ, WINDOW_SIZE)  # 2.56 seconds


# ─────────────────────────────────────────────────────────────────────────────
# ADL (Normal Activity) Generators
# Each function returns a (128, 6) array in physical units
# acc: m/s², gyro: deg/s, gravity on Y-axis
# ─────────────────────────────────────────────────────────────────────────────

def gen_standing() -> np.ndarray:
    """Still standing — tiny sway, gravity on Y."""
    w = np.zeros((WINDOW_SIZE, 6))
    w[:, 0] = RNG.normal(0,   0.05, WINDOW_SIZE)   # acc_x: tiny lateral
    w[:, 1] = RNG.normal(9.81, 0.05, WINDOW_SIZE)  # acc_y: gravity
    w[:, 2] = RNG.normal(0,   0.05, WINDOW_SIZE)   # acc_z: tiny depth
    w[:, 3] = RNG.normal(0,   0.5,  WINDOW_SIZE)   # gyro_x
    w[:, 4] = RNG.normal(0,   0.5,  WINDOW_SIZE)   # gyro_y
    w[:, 5] = RNG.normal(0,   0.5,  WINDOW_SIZE)   # gyro_z
    return w


def gen_walking() -> np.ndarray:
    """Normal walking — 1.8 Hz cadence, gravity on Y + vertical bounce."""
    w = np.zeros((WINDOW_SIZE, 6))
    freq = RNG.uniform(1.6, 2.0)  # cadence Hz
    phase = RNG.uniform(0, 2 * np.pi)
    w[:, 0] = RNG.normal(0, 0.3, WINDOW_SIZE) + np.sin(2*np.pi*freq*T + phase) * 0.8
    w[:, 1] = RNG.normal(9.81, 0.3, WINDOW_SIZE) + np.sin(2*np.pi*freq*T + phase) * 2.0
    w[:, 2] = RNG.normal(0, 0.3, WINDOW_SIZE) + np.cos(2*np.pi*freq*T + phase) * 0.5
    w[:, 3] = RNG.normal(0, 2,   WINDOW_SIZE) + np.sin(2*np.pi*freq*T) * 5
    w[:, 4] = RNG.normal(0, 2,   WINDOW_SIZE)
    w[:, 5] = RNG.normal(0, 3,   WINDOW_SIZE) + np.sin(2*np.pi*freq*T) * 8
    return w


def gen_jogging() -> np.ndarray:
    """Jogging — higher frequency, higher amplitude than walking."""
    w = np.zeros((WINDOW_SIZE, 6))
    freq = RNG.uniform(2.5, 3.5)
    phase = RNG.uniform(0, 2 * np.pi)
    w[:, 0] = RNG.normal(0, 0.8, WINDOW_SIZE) + np.sin(2*np.pi*freq*T + phase) * 3.0
    w[:, 1] = RNG.normal(9.81, 0.8, WINDOW_SIZE) + np.sin(2*np.pi*freq*T + phase) * 5.0
    w[:, 2] = RNG.normal(0, 0.8, WINDOW_SIZE) + np.cos(2*np.pi*freq*T + phase) * 1.5
    w[:, 3] = RNG.normal(0, 5,  WINDOW_SIZE) + np.sin(2*np.pi*freq*T) * 12
    w[:, 4] = RNG.normal(0, 5,  WINDOW_SIZE)
    w[:, 5] = RNG.normal(0, 8,  WINDOW_SIZE) + np.sin(2*np.pi*freq*T) * 20
    return w


def gen_stairs() -> np.ndarray:
    """Climbing stairs — step pattern, slower than jogging."""
    w = np.zeros((WINDOW_SIZE, 6))
    freq = RNG.uniform(0.8, 1.2)
    phase = RNG.uniform(0, 2 * np.pi)
    w[:, 0] = RNG.normal(0, 0.4, WINDOW_SIZE)
    w[:, 1] = RNG.normal(9.81, 0.5, WINDOW_SIZE) + np.sin(2*np.pi*freq*T + phase) * 3.5
    w[:, 2] = RNG.normal(0, 0.6, WINDOW_SIZE) + np.cos(2*np.pi*freq*T + phase) * 1.0
    w[:, 3] = RNG.normal(0, 3,  WINDOW_SIZE)
    w[:, 4] = RNG.normal(0, 8,  WINDOW_SIZE) + np.sin(2*np.pi*freq*T) * 20
    w[:, 5] = RNG.normal(0, 3,  WINDOW_SIZE)
    return w


def gen_sitting_down() -> np.ndarray:
    """Sitting down — brief tilt then stable."""
    w = np.zeros((WINDOW_SIZE, 6))
    # First 0.5s: tilt movement
    n_tilt = int(0.5 * HZ)
    w[:n_tilt, 0] = RNG.normal(0, 1.5, n_tilt)
    w[:n_tilt, 1] = RNG.normal(7.0, 1.0, n_tilt)   # gravity component shifts
    w[:n_tilt, 2] = RNG.normal(5.0, 1.0, n_tilt)
    w[:n_tilt, 4] = RNG.normal(-25, 5, n_tilt)      # forward lean gyro
    # Rest: stable sitting
    w[n_tilt:, 0] = RNG.normal(0,   0.1, WINDOW_SIZE - n_tilt)
    w[n_tilt:, 1] = RNG.normal(9.0, 0.1, WINDOW_SIZE - n_tilt)
    w[n_tilt:, 2] = RNG.normal(3.5, 0.1, WINDOW_SIZE - n_tilt)
    w[n_tilt:, 3:] = RNG.normal(0, 0.5, (WINDOW_SIZE - n_tilt, 3))
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Fall Generators
# Clear 3-phase structure: pre-fall normal → impact spike → post-fall rest
# ─────────────────────────────────────────────────────────────────────────────

def _inject_fall(w: np.ndarray, impact_start: int,
                 impact_duration: int,
                 primary_acc_axis: int,
                 primary_acc_val: float,
                 gyro_spike_mag: float) -> np.ndarray:
    """Inject a fall impact + post-fall rest into window w."""
    ie = impact_start + impact_duration
    ie = min(ie, WINDOW_SIZE)

    # Impact phase: high-G spike
    for ax in range(3):
        val = primary_acc_val if ax == primary_acc_axis else RNG.normal(0, 5, ie - impact_start)
        if ax == primary_acc_axis:
            w[impact_start:ie, ax] = RNG.normal(primary_acc_val, 8, ie - impact_start)
        else:
            w[impact_start:ie, ax] = RNG.normal(0, 10, ie - impact_start)
    # Gyro spike
    w[impact_start:ie, 3] = RNG.normal(0, gyro_spike_mag, ie - impact_start)
    w[impact_start:ie, 4] = RNG.normal(0, gyro_spike_mag, ie - impact_start)
    w[impact_start:ie, 5] = RNG.normal(0, gyro_spike_mag, ie - impact_start)

    # Post-fall rest: gravity redistributed (person lying on ground)
    if ie < WINDOW_SIZE:
        rest_len = WINDOW_SIZE - ie
        # Gravity now spreads across axes (person lying down)
        g_y = RNG.uniform(0.0, 6.0)    # partial gravity on Y
        g_z = np.sqrt(max(0, 9.81**2 - g_y**2))  # rest on Z
        w[ie:, 0] = RNG.normal(0,   0.3, rest_len)
        w[ie:, 1] = RNG.normal(g_y, 0.3, rest_len)
        w[ie:, 2] = RNG.normal(g_z, 0.3, rest_len)
        w[ie:, 3] = RNG.normal(0,   1.0, rest_len)
        w[ie:, 4] = RNG.normal(0,   1.0, rest_len)
        w[ie:, 5] = RNG.normal(0,   1.0, rest_len)
    return w


def gen_forward_fall() -> np.ndarray:
    """Forward fall (FOL) — person falls forward, high acc on Z axis."""
    w = gen_walking()  # pre-fall context
    impact_start = RNG.integers(30, 70)
    impact_dur   = RNG.integers(8, 18)
    # Forward fall: large negative acc_y (loss of gravity support) + large acc_z
    w = _inject_fall(w, impact_start, impact_dur,
                     primary_acc_axis=2,
                     primary_acc_val=RNG.uniform(-40, -25),
                     gyro_spike_mag=RNG.uniform(80, 200))
    return w


def gen_backward_fall() -> np.ndarray:
    """Backward stumble (BSC) — large backward spike on Z."""
    w = gen_walking()
    impact_start = RNG.integers(30, 70)
    impact_dur   = RNG.integers(8, 15)
    w = _inject_fall(w, impact_start, impact_dur,
                     primary_acc_axis=2,
                     primary_acc_val=RNG.uniform(25, 45),
                     gyro_spike_mag=RNG.uniform(80, 180))
    return w


def gen_sideways_fall() -> np.ndarray:
    """Sideways lateral fall (SDL) — large spike on X axis."""
    w = gen_walking()
    impact_start = RNG.integers(30, 70)
    impact_dur   = RNG.integers(6, 15)
    w = _inject_fall(w, impact_start, impact_dur,
                     primary_acc_axis=0,
                     primary_acc_val=RNG.uniform(-45, -20),
                     gyro_spike_mag=RNG.uniform(100, 250))
    return w


def gen_knee_fall() -> np.ndarray:
    """Fall on knee (FKL) — moderate impact, slower descent."""
    w = gen_walking()
    impact_start = RNG.integers(40, 80)
    impact_dur   = RNG.integers(12, 22)
    w = _inject_fall(w, impact_start, impact_dur,
                     primary_acc_axis=1,
                     primary_acc_val=RNG.uniform(-30, -15),
                     gyro_spike_mag=RNG.uniform(50, 130))
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────

NORMAL_GENERATORS = [gen_standing, gen_walking, gen_jogging, gen_stairs, gen_sitting_down]
FALL_GENERATORS   = [gen_forward_fall, gen_backward_fall, gen_sideways_fall, gen_knee_fall]

N_NORMAL = 1200
N_FALL   = 800


def generate_dataset(n_normal: int = N_NORMAL, n_fall: int = N_FALL) -> tuple:
    """
    Generate balanced synthetic IMU dataset in physical units.

    Returns
    -------
    X : np.ndarray  shape (N, 128, 6)  float32
    y : np.ndarray  shape (N,)          int32
    subjects : np.ndarray  shape (N,)   object (mock subject IDs)
    """
    logger.info(f"Generating synthetic MobiFall-like dataset — "
                f"Normal: {n_normal}, Fall: {n_fall}")

    X_list, y_list, sub_list = [], [], []
    n_subjects = 24  # mimic MobiFall's 24 subjects

    # --- Normal Activity ---
    for i in range(n_normal):
        gen_fn = NORMAL_GENERATORS[i % len(NORMAL_GENERATORS)]
        window = gen_fn().astype(np.float32)
        X_list.append(window)
        y_list.append(0)
        sub_list.append(f"sub{(i % n_subjects) + 1:02d}")

    # --- Falls ---
    for i in range(n_fall):
        gen_fn = FALL_GENERATORS[i % len(FALL_GENERATORS)]
        window = gen_fn().astype(np.float32)
        X_list.append(window)
        y_list.append(1)
        sub_list.append(f"sub{(i % n_subjects) + 1:02d}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(sub_list, dtype=object)

    # Shuffle
    idx = RNG.permutation(len(X))
    X, y, subjects = X[idx], y[idx], subjects[idx]

    logger.info(f"Dataset generated — shape: {X.shape} | "
                f"Normal: {(y==0).sum()} | Fall: {(y==1).sum()}")

    # Sanity: print value ranges
    logger.info(f"acc_y range: [{X[:,:,1].min():.2f}, {X[:,:,1].max():.2f}] "
                f"mean={X[:,:,1].mean():.2f}  (expect ~9.8 for normal)")

    return X, y, subjects


def save(X, y, subjects, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_windows.npy"), X)
    np.save(os.path.join(out_dir, "y_labels.npy"),  y)
    np.save(os.path.join(out_dir, "subjects.npy"),   subjects)
    logger.info(f"Saved to {out_dir}/  — X:{X.shape} y:{y.shape}")


if __name__ == "__main__":
    X, y, subjects = generate_dataset()
    save(X, y, subjects)
    logger.info("Done. Now run: python src/train.py")
