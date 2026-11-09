"""
Tests for the MobiFall parsing / synchronisation pipeline and the
synthetic car-crash dataset, using small files in the real MobiFall format.
"""

import numpy as np
import pytest

from src.preprocess import (
    generate_dummy_data,
    load_mobifall_recordings,
    merge_acc_gyro,
    parse_sensor_file,
    windows_from_recordings,
)

HEADER = """#Activity: FOL - Forward-lying
#Subject ID: 3
#Sensor: Accelerometer
#Timestamp(ns), x, y, z
@DATA
"""


def write_sensor(path, t_ns, xyz):
    with open(path, "w") as f:
        f.write(HEADER)
        for t, (x, y, z) in zip(t_ns, xyz):
            f.write(f"{int(t)}, {x:.6f}, {y:.6f}, {z:.6f}\n")


def make_trial(folder, code, subject, trial, seconds=6.0, impact_s=None, seed=0):
    """Write matching acc (~87 Hz) and gyro (~200 Hz) files with different start times."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    t_acc = np.cumsum(rng.uniform(10e6, 13e6, int(seconds * 87))) + 1e12
    t_gyr = np.cumsum(rng.uniform(4.5e6, 5.5e6, int(seconds * 200))) + 1e12 + 30e6
    acc = np.column_stack([np.zeros_like(t_acc), np.full_like(t_acc, 9.81), np.zeros_like(t_acc)])
    if impact_s is not None:
        k = np.argmin(np.abs((t_acc - t_acc[0]) / 1e9 - impact_s))
        acc[k] = [20, 30, 10]
    gyr = np.column_stack([np.full_like(t_gyr, 1.0), np.zeros_like(t_gyr), np.zeros_like(t_gyr)])  # 1 rad/s
    write_sensor(folder / f"{code}_acc_{subject}_{trial}.txt", t_acc, acc)
    write_sensor(folder / f"{code}_gyro_{subject}_{trial}.txt", t_gyr, gyr)


@pytest.fixture
def mobifall_tree(tmp_path):
    root = tmp_path / "MobiFall_Dataset_v2.0"
    make_trial(root / "sub1" / "ADL" / "WAL", "WAL", 1, 1, seed=1)
    make_trial(root / "sub1" / "FALLS" / "FOL", "FOL", 1, 1, impact_s=3.0, seed=2)
    make_trial(root / "sub2" / "ADL" / "STD", "STD", 2, 1, seed=3)
    (root / "sub2" / "ADL" / "STD" / "STD_ori_2_1.txt").write_text(HEADER)  # ignored sensor
    return tmp_path


class TestParsing:
    def test_parse_sensor_file(self, tmp_path):
        p = tmp_path / "x.txt"
        write_sensor(p, [1e9, 2e9], [(1, 2, 3), (4, 5, 6)])
        arr = parse_sensor_file(str(p))
        assert arr.shape == (2, 4)
        np.testing.assert_allclose(arr[:, 0], [1.0, 2.0])  # ns → s

    def test_parse_rejects_headers_only(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text(HEADER)
        assert parse_sensor_file(str(p)) is None

    def test_merge_aligns_and_converts_units(self):
        acc = np.column_stack([np.arange(0, 3, 0.011), np.zeros(273), np.full(273, 9.81), np.zeros(273)])
        gyro = np.column_stack([np.arange(0.5, 3.5, 0.005), np.full(600, np.pi), np.zeros(600), np.zeros(600)])
        out = merge_acc_gyro(acc, gyro, 50)
        assert out.shape[1] == 6
        assert abs(len(out) - (3.0 - 0.5) * 50) <= 2     # common span only
        np.testing.assert_allclose(out[:, 3], 180.0, rtol=1e-5)  # rad/s → °/s
        np.testing.assert_allclose(out[:, 1], 9.81, rtol=1e-6)

    def test_merge_non_overlapping(self):
        a = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
        g = np.array([[5, 0, 0, 0], [6, 0, 0, 0]], dtype=float)
        assert merge_acc_gyro(a, g, 50) is None


class TestMobiFallPipeline:
    def test_loads_trials_with_subjects_and_labels(self, mobifall_tree, base_config):
        recs = load_mobifall_recordings(str(mobifall_tree), base_config)
        assert {(r.code, r.subject, r.label) for r in recs} == {
            ("WAL", "sub01", 0), ("FOL", "sub01", 1), ("STD", "sub02", 0)}
        for r in recs:
            assert r.data.shape[1] == 6
            assert abs(len(r.data) / 50 - 6.0) < 0.5

    def test_fall_windows_are_impact_centred(self, mobifall_tree, base_config):
        recs = load_mobifall_recordings(str(mobifall_tree), base_config)
        X, y, groups = windows_from_recordings(recs, base_config)
        assert X.shape[1:] == (128, 6)
        fall = X[y == 1]
        assert len(fall) >= 1
        for w in fall:
            assert np.linalg.norm(w[:, :3], axis=1).max() > 30  # every fall window contains the impact
        assert set(groups) == {"sub01", "sub02"}


class TestCarDataset:
    def test_generator_respects_phone_sensor_range(self, base_config):
        from scripts.generate_car_crash_data import generate
        data, offsets, impact, kind = generate(10, 8, 12.0, 0, base_config.signal.acc_range,
                                               base_config.signal.gyro_range)
        assert len(offsets) == len(kind) + 1 == 19
        assert np.abs(data[:, :3]).max() <= base_config.signal.acc_range + 1e-3
        assert np.abs(data[:, 3:]).max() <= base_config.signal.gyro_range + 1e-3
        assert (impact[kind == "normal"] == -1).all()
        assert (impact[np.char.startswith(kind.astype(str), "crash")] >= 0).all()


def test_dummy_data_is_learnable_and_grouped(base_config):
    X, y, g = generate_dummy_data(base_config)
    assert X.shape == (240, 128, 6) and set(np.unique(y)) == {0, 1}
    assert len(np.unique(g)) == 8
