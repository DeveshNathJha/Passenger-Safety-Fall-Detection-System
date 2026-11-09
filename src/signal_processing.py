"""
src/signal_processing.py
------------------------
Signal utilities shared by the offline training pipeline and the online
inference server, so that both see *identical* input distributions:

  - resample_uniform()   offline: irregular timestamps → uniform 50 Hz grid
  - StreamResampler      online:  same linear interpolation, one sample at a time
  - clip_to_sensor_range() saturate like a real phone IMU (±16 g, ±2000 °/s)
  - sliding_windows()    fixed-size windows with optional impact-centred labels
  - random_rotation() / augment_rotation()  orientation-robustness augmentation
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import numpy as np

N_CHANNELS = 6


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_uniform(t_s: np.ndarray, values: np.ndarray, rate_hz: float,
                     t_start: Optional[float] = None,
                     t_end: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate an irregularly-sampled signal onto a uniform grid.

    Parameters
    ----------
    t_s : np.ndarray
        Strictly increasing timestamps in seconds, shape (T,).
    values : np.ndarray
        Signal values, shape (T, C).
    rate_hz : float
        Target sampling rate.
    t_start, t_end : float | None
        Grid bounds (default: first / last timestamp).

    Returns
    -------
    (grid, resampled) : grid shape (N,), resampled shape (N, C)
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(t_s, kind="stable")
    t_s, values = t_s[order], values[order]
    keep = np.concatenate([[True], np.diff(t_s) > 0])  # drop duplicate timestamps
    t_s, values = t_s[keep], values[keep]

    t0 = t_s[0] if t_start is None else t_start
    t1 = t_s[-1] if t_end is None else t_end
    if t1 <= t0:
        return np.empty(0), np.empty((0, values.shape[1]))
    grid = np.arange(t0, t1, 1.0 / rate_hz)
    out = np.column_stack([np.interp(grid, t_s, values[:, c]) for c in range(values.shape[1])])
    return grid, out


class StreamResampler:
    """
    Online linear-interpolation resampler.

    Feed (timestamp_ms, sample) pairs at whatever rate the device produces
    them (browsers deliver DeviceMotion at 60–200 Hz with jitter); receive
    samples on an exact ``rate_hz`` grid. If timestamps are not provided
    the stream is assumed to already be at ``rate_hz`` and is passed through.
    """

    # Gaps longer than this restart the grid instead of interpolating across them
    MAX_GAP_S = 0.5

    def __init__(self, rate_hz: float = 50.0):
        self.period = 1.0 / rate_hz
        self._prev_t: Optional[float] = None
        self._prev_v: Optional[np.ndarray] = None
        self._next_grid_t: Optional[float] = None

    def reset(self) -> None:
        self._prev_t = self._prev_v = self._next_grid_t = None

    def push(self, sample, t_ms: Optional[float] = None) -> list[np.ndarray]:
        v = np.asarray(sample, dtype=np.float32)
        if t_ms is None:
            return [v]

        t = t_ms / 1000.0
        if self._prev_t is None or t - self._prev_t > self.MAX_GAP_S or t < self._prev_t:
            # First sample, long gap, or clock reset → (re)start the grid here
            self._prev_t, self._prev_v, self._next_grid_t = t, v, t + self.period
            return [v]
        if t == self._prev_t:
            self._prev_v = v
            return []

        out = []
        while self._next_grid_t <= t:
            alpha = (self._next_grid_t - self._prev_t) / (t - self._prev_t)
            out.append(((1 - alpha) * self._prev_v + alpha * v).astype(np.float32))
            self._next_grid_t += self.period
        self._prev_t, self._prev_v = t, v
        return out


# ---------------------------------------------------------------------------
# Sensor realism
# ---------------------------------------------------------------------------

def clip_to_sensor_range(x: np.ndarray, acc_range: float, gyro_range: float) -> np.ndarray:
    """Saturate acc / gyro channels like a real IMU with the given full-scale range."""
    x = np.array(x, dtype=np.float32, copy=True)
    x[..., 0:3] = np.clip(x[..., 0:3], -acc_range, acc_range)
    x[..., 3:6] = np.clip(x[..., 3:6], -gyro_range, gyro_range)
    return x


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_starts(n: int, window_size: int, step: int) -> Iterator[int]:
    """Start indices of all full windows (the last full window is always included)."""
    start = 0
    while start + window_size <= n:
        yield start
        start += step


def sliding_windows(
    data: np.ndarray,
    label: int,
    window_size: int = 128,
    step: int = 64,
    impact_idx: Optional[int] = None,
    impact_margin: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a recording into overlapping windows.

    For event recordings (falls, crashes) pass ``impact_idx``: only windows
    whose impact lies inside [margin, window_size - margin) are kept and
    labelled ``label``. Windows *before* the impact (still standing / still
    driving) and long *after* it are ambiguous and are dropped, instead of
    being mislabelled as events.

    Returns
    -------
    (X, y) : X shape (N, window_size, C) float32, y shape (N,) int32
    """
    data = np.asarray(data, dtype=np.float32)
    windows = []
    for s in window_starts(len(data), window_size, step):
        if impact_idx is not None:
            pos = impact_idx - s
            if not (impact_margin <= pos < window_size - impact_margin):
                continue
        windows.append(data[s:s + window_size])

    # Guarantee at least one impact-centred window for short event recordings
    if impact_idx is not None and not windows and len(data) >= window_size:
        s = int(np.clip(impact_idx - window_size // 2, 0, len(data) - window_size))
        windows.append(data[s:s + window_size])

    if not windows:
        return (np.empty((0, window_size, data.shape[1] if data.ndim == 2 else N_CHANNELS),
                         dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    return np.stack(windows), np.full(len(windows), label, dtype=np.int32)


def find_impact_index(data: np.ndarray) -> int:
    """Index of the peak acceleration magnitude (the impact of a fall / crash)."""
    return int(np.argmax(np.linalg.norm(np.asarray(data)[:, 0:3], axis=1)))


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3-D rotation matrix (via QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotate_windows(X: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Apply one rotation per window to both the acc and gyro triplets."""
    acc = np.einsum("nij,ntj->nti", rotations, X[..., 0:3])
    gyr = np.einsum("nij,ntj->nti", rotations, X[..., 3:6])
    return np.concatenate([acc, gyr], axis=-1).astype(np.float32)


def augment_rotation(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a randomly rotated copy of every window (device-orientation invariance)."""
    rots = np.stack([random_rotation(rng) for _ in range(len(X))]) if len(X) else np.empty((0, 3, 3))
    return rotate_windows(X, rots)


# ---------------------------------------------------------------------------
# Orientation-invariant features
# ---------------------------------------------------------------------------

INVARIANT_FEATURES = ["acc_norm", "acc_vertical", "acc_horizontal",
                      "gyro_norm", "gyro_vertical", "gyro_horizontal"]
_EPS = 1e-6


def invariant_features(X: np.ndarray) -> np.ndarray:
    """
    Map windows (N, T, 6) of raw phone-frame IMU data to 6 channels that do not
    change when the phone is rotated:

        |a|, a·u, |a − (a·u)u|,  |ω|, ω·u, |ω − (ω·u)u|

    where u is the unit gravity direction, estimated as the mean acceleration
    of the window. Rotating the phone rotates a, ω and u together, so every dot
    product and norm stays the same: the features are rotation-invariant by
    construction, not just by augmentation.

    The Keras layer `GravityAlignedFeatures` (src/model_layers.py) computes the
    same thing inside the model; tests check that both agree.
    """
    X = np.asarray(X, dtype=np.float64)
    squeeze = X.ndim == 2
    if squeeze:
        X = X[None]
    acc, gyr = X[..., 0:3], X[..., 3:6]
    g = acc.mean(axis=1, keepdims=True)
    u = g / (np.linalg.norm(g, axis=-1, keepdims=True) + _EPS)

    def split(v):
        sq = (v * v).sum(-1, keepdims=True)
        vert = (v * u).sum(-1, keepdims=True)
        horiz = np.sqrt(np.maximum(sq - vert * vert, 0.0) + _EPS)
        return np.sqrt(sq + _EPS), vert, horiz

    out = np.concatenate([*split(acc), *split(gyr)], axis=-1).astype(np.float32)
    return out[0] if squeeze else out
