"""
Shared pytest fixtures.

Most tests use FakeModel – a deterministic stand-in for a TFLite model –
so the API / alert / streaming logic is tested fast and without TensorFlow.
tests/test_models.py exercises the real .tflite files separately.
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FDS_LOG_FILE", "")  # no log files from test runs

from src.inference import ModelRegistry  # noqa: E402
from src.utils.config_loader import AttrDict, load_config  # noqa: E402
from src.utils.notifier import MockNotifier  # noqa: E402


class FakeModel:
    """Scores 0.95 if the window contains an |acc| spike above 30 m/s², else 0.05."""

    def __init__(self, name: str, threshold: float = 0.5, window_size: int = 128):
        self.name = name
        self.threshold = threshold
        self.window_size = window_size
        self.n_channels = 6
        self.labels = {0: "Normal", 1: "EVENT"}
        self.calls = 0

    def score(self, window: np.ndarray):
        assert window.shape == (self.window_size, 6)
        self.calls += 1
        peak = np.linalg.norm(window[:, 0:3], axis=1).max()
        return (0.95 if peak > 30 else 0.05), 0.1

    def info(self) -> dict:
        return {"file": f"{self.name}.tflite", "threshold": self.threshold, "labels": self.labels,
                "window_size": self.window_size, "size_kb": 1.0}


@pytest.fixture(scope="session")
def base_config():
    return load_config()


@pytest.fixture
def config(base_config):
    cfg = AttrDict(copy.deepcopy(dict(base_config)))
    cfg.alerts.countdown_s = 0.3
    cfg.alerts.cooldown_s = 60
    return cfg


@pytest.fixture
def fake_registry():
    return ModelRegistry({"human": FakeModel("human"), "car": FakeModel("car")})


@pytest.fixture
def notifier():
    return MockNotifier("+10000000000")


def walking(n: int, seed: int = 0) -> np.ndarray:
    """Plausible walking-like IMU samples (no spike)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 50
    x = np.zeros((n, 6), dtype=np.float32)
    x[:, 1] = 9.81 + 2 * np.sin(2 * np.pi * 1.8 * t)
    x[:, 0] = rng.normal(0, 1, n)
    x[:, 2] = rng.normal(0, 1, n)
    x[:, 3:6] = rng.normal(0, 20, (n, 3))
    return x


def jogging(n: int, seed: int = 0) -> np.ndarray:
    """Vigorous movement (jogging / jumping-like): clearly not 'lying still'."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 50
    x = np.zeros((n, 6), dtype=np.float32)
    x[:, 1] = 9.81 + 8 * np.sin(2 * np.pi * 2.8 * t)
    x[:, 0] = rng.normal(0, 3, n)
    x[:, 2] = rng.normal(0, 3, n)
    x[:, 3:6] = rng.normal(0, 80, (n, 3))
    return x


def as_reading(row, **extra) -> dict:
    keys = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    return {**{k: float(v) for k, v in zip(keys, row)}, **extra}
