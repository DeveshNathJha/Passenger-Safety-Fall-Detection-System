"""
scripts/generate_car_crash_data.py
----------------------------------
Physics-informed synthetic IMU data for "normal driving" vs "crash".

Real, labelled smartphone crash recordings are essentially unavailable for
academic use, so this generator models what a phone inside a car measures.
It is deliberately *hard* so the model must learn more than "big spike = crash":

  Normal driving  : road vibration, smooth acceleration/braking (≤0.4 g),
                    turns (yaw 10–40 °/s) and HARD NEGATIVES –
                      * potholes / speed bumps (short 1–3 g vertical jolts)
                      * emergency braking (0.8–1.1 g for 1–2.5 s)
                      * phone picked up / moved by the passenger
                      * phone dropped onto the car floor (free fall + 3–12 g impact),
                        40 % of them in a parked car (phone then lies still)
  Crash           : half-sine crash pulse of 5–40 g (30 % low-speed) lasting 60–150 ms along a
                    random horizontal direction (frontal / rear / side), optional
                    spin-out (yaw up to 500 °/s) or rollover (roll up to 700 °/s,
                    gravity vector ends up rotated), then the vehicle at rest.

Every sequence is then rotated by a random phone orientation and clipped to
a smartphone IMU's full-scale range (±16 g, ±2000 °/s) — phones saturate,
so the model never sees the physically "true" 40 g.

Output (no CSV sprawl; deterministic with --seed):
    data/raw/synthetic_car_crash.npz
        data      (T_total, 6) float32   all sequences concatenated, 50 Hz
        offsets   (S + 1,)     int64     sequence boundaries
        impact    (S,)         int64     impact sample index, -1 for normal
        kind      (S,)         str       normal | pothole | hard_brake | handling | drop | crash_*

Usage:
    python scripts/generate_car_crash_data.py --n-normal 600 --n-crash 400
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.signal_processing import clip_to_sensor_range, random_rotation
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

G = 9.81
HZ = 50


def _smooth_noise(rng, n, scale, smooth=25):
    """Low-frequency random signal (moving-average filtered white noise)."""
    k = np.ones(smooth) / smooth
    return np.convolve(rng.normal(0, scale, n + smooth), k, mode="same")[:n] * np.sqrt(smooth)


def driving(rng, n):
    """
    Vehicle frame: x = forward, y = left, z = up. Returns (n, 6) in m/s², °/s.
    """
    d = np.zeros((n, 6))
    d[:, 0] = _smooth_noise(rng, n, 0.12 * G)              # throttle / braking
    d[:, 1] = _smooth_noise(rng, n, 0.06 * G)              # lateral
    d[:, 2] = G + rng.normal(0, 0.05 * G, n)               # gravity + road vibration
    d[:, 3:5] = rng.normal(0, 1.5, (n, 2))                 # roll / pitch jitter
    d[:, 5] = _smooth_noise(rng, n, 3.0)                   # yaw drift

    for _ in range(rng.integers(0, 3)):                    # turns
        s = rng.integers(0, max(1, n - 150))
        length = rng.integers(60, 150)
        rate = rng.uniform(10, 40) * rng.choice([-1, 1])
        bump = np.sin(np.linspace(0, np.pi, length))
        e = min(n, s + length)
        d[s:e, 5] += rate * bump[: e - s]
        d[s:e, 1] += np.deg2rad(rate) * rng.uniform(8, 20) * bump[: e - s]  # centripetal a = ω·v
    return d


def add_pothole(rng, d):
    s = rng.integers(10, len(d) - 10)
    width = rng.integers(2, 5)
    d[s:s + width, 2] += rng.uniform(1.0, 3.0) * G * rng.choice([-1, 1]) * np.hanning(width + 2)[1:-1]
    d[s:s + width + 3, 4] += rng.normal(0, 25, width + 3)
    return s


def add_hard_brake(rng, d):
    s = rng.integers(0, len(d) - 130)
    length = rng.integers(50, 125)
    d[s:s + length, 0] -= rng.uniform(0.8, 1.1) * G * np.sin(np.linspace(0, np.pi, length)) ** 0.5
    d[s:s + length, 4] += rng.uniform(2, 6)  # nose dive
    return s


def add_handling(rng, d):
    """Passenger picks the phone up, waves / rotates it, puts it back."""
    s = rng.integers(0, len(d) - 160)
    length = rng.integers(60, 150)
    d[s:s + length, 0:3] += rng.normal(0, 0.5 * G, (length, 3))
    d[s:s + length, 3:6] += _smooth_noise(rng, length * 3, 120.0)[: length * 3].reshape(length, 3)
    return s


def add_drop(rng, d):
    """
    Phone falls ~0.5–1 m onto the floor: free fall, a hard 3–12 g impact,
    bounce, then rest. In 40 % of cases the car is parked, so the phone ends
    up perfectly still afterwards – exactly like the post-crash phase.
    """
    s = rng.integers(20, len(d) - 60)
    if rng.random() < 0.4:  # parked car: no driving signal at all
        d[:, 0:3] = [0.0, 0.0, G]
        d[:, 3:6] = 0.0
        d += np.concatenate([rng.normal(0, 0.03 * G, (len(d), 3)), rng.normal(0, 0.5, (len(d), 3))], axis=1)
    ff = rng.integers(12, 22)                               # 0.25–0.45 s free fall
    d[s:s + ff, 0:3] = rng.normal(0, 0.3, (ff, 3))
    d[s:s + ff, 3:6] += rng.normal(0, 150, (ff, 3))
    imp = s + ff
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    width = rng.integers(1, 4)
    d[imp:imp + width, 0:3] += np.outer(np.hanning(width + 2)[1:-1], direction) * rng.uniform(3, 12) * G
    d[imp + width:imp + width + 6, 3:6] += rng.normal(0, 80, (6, 3))
    return imp


def add_crash(rng, d, kind):
    """Insert a crash; returns the impact index. Everything after it is 'vehicle at rest'."""
    n = len(d)
    imp = rng.integers(int(0.3 * n), int(0.7 * n))
    pulse_len = rng.integers(3, 8)                          # 60–150 ms at 50 Hz
    # 30 % low-speed collisions (5–10 g) that overlap with hard phone drops
    peak = (rng.uniform(5, 10) if rng.random() < 0.3 else rng.uniform(10, 40)) * G
    theta = rng.uniform(0, 2 * np.pi) if kind != "crash_frontal" else rng.normal(np.pi, 0.3)
    shape = np.sin(np.linspace(0, np.pi, pulse_len + 2))[1:-1]
    d[imp:imp + pulse_len, 0] += peak * np.cos(theta) * shape
    d[imp:imp + pulse_len, 1] += peak * np.sin(theta) * shape
    d[imp:imp + pulse_len, 2] += rng.normal(0, 3 * G, pulse_len)
    d[imp:imp + pulse_len, 3:6] += rng.normal(0, 300, (pulse_len, 3))

    after = imp + pulse_len
    rest = d[after:]
    rest[:, 0:3] = [0.0, 0.0, G]
    rest[:, 3:6] = 0.0

    if kind == "crash_spin":
        length = min(len(rest), rng.integers(25, 75))
        rate = rng.uniform(150, 500) * rng.choice([-1, 1])
        rest[:length, 5] = rate * np.linspace(1, 0, length)
        rest[:length, 0:2] += rng.normal(0, 0.8 * G, (length, 2))
    elif kind == "crash_rollover":
        length = min(len(rest), rng.integers(40, 100))
        rate = rng.uniform(300, 700) * rng.choice([-1, 1])
        rest[:length, 3] = rate
        angle = np.deg2rad(np.cumsum(np.full(length, rate)) / HZ)
        rest[:length, 1] = G * np.sin(angle) + rng.normal(0, G, length)
        rest[:length, 2] = G * np.cos(angle) + rng.normal(0, G, length)
        final = angle[-1]
        rest[length:, 1] = G * np.sin(final)
        rest[length:, 2] = G * np.cos(final)

    rest[:, 0:3] += rng.normal(0, 0.03 * G, (len(rest), 3))
    rest[:, 3:6] += rng.normal(0, 0.5, (len(rest), 3))
    return imp


def phone_frame(rng, d):
    """Rotate vehicle-frame signals into a random (but fixed) phone orientation."""
    r = random_rotation(rng)
    out = d.copy()
    out[:, 0:3] = d[:, 0:3] @ r.T
    out[:, 3:6] = d[:, 3:6] @ r.T
    return out


NEGATIVE_KINDS = ["normal", "pothole", "hard_brake", "handling", "drop"]
CRASH_KINDS = ["crash_frontal", "crash_side", "crash_spin", "crash_rollover"]


def generate(n_normal: int, n_crash: int, seconds: float, seed: int, acc_range: float, gyro_range: float):
    rng = np.random.default_rng(seed)
    n = int(seconds * HZ)
    seqs, impacts, kinds = [], [], []

    for i in range(n_normal):
        kind = NEGATIVE_KINDS[i % len(NEGATIVE_KINDS)]
        d = driving(rng, n)
        {"pothole": add_pothole, "hard_brake": add_hard_brake,
         "handling": add_handling, "drop": add_drop}.get(kind, lambda r, x: None)(rng, d)
        seqs.append(d)
        impacts.append(-1)
        kinds.append(kind)

    for i in range(n_crash):
        kind = CRASH_KINDS[i % len(CRASH_KINDS)]
        d = driving(rng, n)
        if rng.random() < 0.3:
            add_pothole(rng, d)  # crashes also happen on rough roads
        impacts.append(add_crash(rng, d, kind))
        seqs.append(d)
        kinds.append(kind)

    seqs = [clip_to_sensor_range(phone_frame(rng, s), acc_range, gyro_range) for s in seqs]
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in seqs])])
    return np.concatenate(seqs).astype(np.float32), offsets, np.array(impacts), np.array(kinds)


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic car-crash IMU dataset.")
    parser.add_argument("--n-normal", type=int, default=600)
    parser.add_argument("--n-crash", type=int, default=400)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    data, offsets, impact, kind = generate(args.n_normal, args.n_crash, args.seconds, args.seed,
                                           config.signal.acc_range, config.signal.gyro_range)
    out = resolve_path(config.data.car_raw_dir).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=data, offsets=offsets, impact=impact, kind=kind)
    logger.info(f"Saved {len(kind)} sequences ({len(data):,} samples) → {out}")
    for k in NEGATIVE_KINDS + CRASH_KINDS:
        logger.info(f"  {k:<15} {int((kind == k).sum())}")


if __name__ == "__main__":
    main()
