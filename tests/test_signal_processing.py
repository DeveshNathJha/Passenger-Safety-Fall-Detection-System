"""Unit tests for src/signal_processing.py."""

import numpy as np
import pytest

from src.signal_processing import (
    StreamResampler,
    augment_rotation,
    clip_to_sensor_range,
    find_impact_index,
    random_rotation,
    resample_uniform,
    sliding_windows,
)


class TestResampleUniform:
    def test_linear_signal_is_reproduced_exactly(self):
        t = np.sort(np.random.default_rng(0).uniform(0, 2, 300))
        v = np.column_stack([3 * t + 1, -t])
        grid, out = resample_uniform(t, v, 50)
        np.testing.assert_allclose(out[:, 0], 3 * grid + 1, atol=1e-9)
        np.testing.assert_allclose(np.diff(grid), 0.02, atol=1e-9)

    def test_unsorted_and_duplicate_timestamps(self):
        t = np.array([0.0, 0.04, 0.02, 0.02, 0.06])
        v = np.array([[0.0], [4.0], [2.0], [2.0], [6.0]])
        grid, out = resample_uniform(t, v, 50)
        np.testing.assert_allclose(out[:, 0], grid * 100, atol=1e-9)


class TestStreamResampler:
    def test_passthrough_without_timestamps(self):
        r = StreamResampler(50)
        assert len(r.push([1] * 6)) == 1
        assert len(r.push([2] * 6)) == 1

    def test_irregular_100hz_stream_becomes_50hz(self):
        r = StreamResampler(50)
        rng = np.random.default_rng(1)
        t = np.cumsum(rng.uniform(8, 12, 500))  # ~100 Hz with jitter, ms
        out = []
        for ti in t:
            out += r.push([ti / 1000.0] * 6, t_ms=ti)  # value = time in s → linear
        duration_s = (t[-1] - t[0]) / 1000
        assert abs(len(out) - duration_s * 50) <= 2
        vals = np.array([o[0] for o in out])
        np.testing.assert_allclose(np.diff(vals), 0.02, atol=1e-4)

    def test_long_gap_restarts_grid(self):
        r = StreamResampler(50)
        r.push([0] * 6, t_ms=0)
        r.push([0] * 6, t_ms=20)
        out = r.push([1] * 6, t_ms=5000)  # 5 s gap → no 250 interpolated samples
        assert len(out) == 1


class TestWindowing:
    def test_counts_with_overlap(self):
        X, y = sliding_windows(np.zeros((300, 6)), 0, 128, 64)
        assert X.shape == (3, 128, 6)  # starts 0, 64, 128
        assert (y == 0).all() and y.dtype == np.int32

    def test_too_short(self):
        X, y = sliding_windows(np.zeros((100, 6)), 1, 128, 64)
        assert X.shape == (0, 128, 6) and y.shape == (0,)

    def test_impact_centred_labelling_drops_pre_and_post_event_windows(self):
        data = np.zeros((640, 6), dtype=np.float32)
        data[300, 0] = 50.0
        X, y = sliding_windows(data, 1, 128, 64, impact_idx=300, impact_margin=16)
        assert len(X) > 0 and (y == 1).all()
        for w in X:
            pos = int(np.argmax(w[:, 0]))
            assert 16 <= pos < 112  # impact well inside every kept window

    def test_impact_at_edge_still_yields_one_window(self):
        data = np.zeros((130, 6), dtype=np.float32)
        X, _ = sliding_windows(data, 1, 128, 64, impact_idx=129, impact_margin=16)
        assert len(X) == 1

    def test_find_impact_index(self):
        data = np.zeros((50, 6))
        data[17, 2] = -40
        assert find_impact_index(data) == 17


class TestSensorRealism:
    def test_clip(self):
        x = np.array([[300, -300, 5, 3000, -3000, 10]], dtype=np.float32)
        out = clip_to_sensor_range(x, 156.9, 2000)
        np.testing.assert_allclose(out[0], [156.9, -156.9, 5, 2000, -2000, 10], rtol=1e-6)
        assert x[0, 0] == 300  # input untouched


class TestRotation:
    def test_is_proper_rotation(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            r = random_rotation(rng)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
            assert np.linalg.det(r) == pytest.approx(1.0)

    def test_augment_preserves_magnitudes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 5, (4, 128, 6)).astype(np.float32)
        Xr = augment_rotation(X, rng)
        np.testing.assert_allclose(np.linalg.norm(Xr[..., :3], axis=-1),
                                   np.linalg.norm(X[..., :3], axis=-1), rtol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(Xr[..., 3:], axis=-1),
                                   np.linalg.norm(X[..., 3:], axis=-1), rtol=1e-4)
        assert not np.allclose(Xr, X)
