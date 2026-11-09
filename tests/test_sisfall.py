"""Tests for the SisFall loader (src/sisfall.py) using files in the real SisFall format."""

import numpy as np
import pytest

from src.sisfall import ACC_SCALE, GYRO_SCALE, SISFALL_RATE_HZ, downsample, load_sisfall_recordings, parse_sisfall_file

G = 9.80665


def write_sisfall(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(f"{v:4d}" for v in r) + ";\n")


def test_unit_conversion_follows_readme_formula():
    # ADXL345: 13 bit, ±16 g → 32 g / 8192 bits → 256 bits = 1 g
    assert 256 * ACC_SCALE == pytest.approx(G)
    # ITG3200: 16 bit, ±2000 °/s → 4000 / 65536 °/s per bit → 16.384 bits = 1 °/s
    assert 16.384 * GYRO_SCALE == pytest.approx(1.0)


def test_parse_file(tmp_path):
    p = tmp_path / "F01_SA01_R01.txt"
    p.write_text("  -9,-257, -25,  84, 247,  27,-120,-987,  63;\n"
                 "   0, 256,   0,   0,   0,  0,   0,   0,   0;\n\n")
    arr = parse_sisfall_file(str(p))
    assert arr.shape == (2, 6)  # MMA8451Q columns dropped
    np.testing.assert_allclose(arr[1], [0, G, 0, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(arr[0, 3], 84 * GYRO_SCALE)


def test_parse_empty_file(tmp_path):
    p = tmp_path / "D01_SA01_R01.txt"
    p.write_text("\n")
    assert parse_sisfall_file(str(p)) is None


def test_downsample_averages_blocks():
    data = np.arange(10 * 2, dtype=float).reshape(10, 2)
    out = downsample(data, 200, 50)          # factor 4, trailing 2 rows dropped
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], data[:4].mean(axis=0))
    np.testing.assert_allclose(out[1], data[4:8].mean(axis=0))


def test_load_tree(tmp_path, base_config):
    n = SISFALL_RATE_HZ * 3  # 3 s → 150 samples at 50 Hz
    still = [[0, 256, 0, 0, 0, 0, 0, 0, 0]] * n
    for subj, code in [("SA01", "F01"), ("SA01", "D07"), ("SE03", "D17")]:
        (tmp_path / subj).mkdir(exist_ok=True)
        write_sisfall(tmp_path / subj / f"{code}_{subj}_R01.txt", still)
    (tmp_path / "SA01" / "desktop.ini").write_text("ignored")
    (tmp_path / "Readme.txt").write_text("ignored")

    recs = load_sisfall_recordings(str(tmp_path), base_config)
    got = {(r.code, r.subject, r.trial, r.label) for r in recs}
    assert got == {("F01", "SA01", "R01", 1), ("D07", "SA01", "R01", 0), ("D17", "SE03", "R01", 0)}
    for r in recs:
        assert r.data.shape == (150, 6) and r.data.dtype == np.float32
        np.testing.assert_allclose(r.data[:, 1], G, rtol=1e-5)
