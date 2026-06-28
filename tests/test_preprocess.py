"""
tests/test_preprocess.py
------------------------
Unit tests for the sliding-window preprocessing logic.
Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import create_windows, parse_txt_file, generate_dummy_data
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_sensor_data():
    """300 rows of random 6-axis IMU data."""
    rng = np.random.default_rng(0)
    return rng.random((300, 6)).astype(np.float32)


# ---------------------------------------------------------------------------
# create_windows tests
# ---------------------------------------------------------------------------

class TestCreateWindows:
    def test_output_shape(self, sample_sensor_data):
        """Each window should be (128, 6)."""
        X, y = create_windows(sample_sensor_data, label=0)
        assert X.ndim == 3
        assert X.shape[1] == 128
        assert X.shape[2] == 6

    def test_label_correctness(self, sample_sensor_data):
        """All labels should match the provided label."""
        for label in [0, 1]:
            _, y = create_windows(sample_sensor_data, label=label)
            assert np.all(y == label), f"Expected all labels={label}"

    def test_window_count_50pct_overlap(self):
        """
        With window=128, step=64, 300 samples → 
        floor((300 - 128) / 64) + 1 = 3 windows.
        """
        data = np.zeros((300, 6), dtype=np.float32)
        X, y = create_windows(data, label=1, window_size=128, step=64)
        expected = (300 - 128) // 64 + 1
        assert X.shape[0] == expected, f"Expected {expected} windows, got {X.shape[0]}"

    def test_no_overlap(self):
        """With step=128 (no overlap), windows = floor(300/128) = 2."""
        data = np.zeros((300, 6), dtype=np.float32)
        X, _ = create_windows(data, label=0, window_size=128, step=128)
        assert X.shape[0] == 2

    def test_input_shorter_than_window(self):
        """Data shorter than window_size should produce 0 windows."""
        data = np.zeros((64, 6), dtype=np.float32)
        X, y = create_windows(data, label=0, window_size=128)
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_exact_window_fit(self):
        """Data exactly = window_size should produce exactly 1 window."""
        data = np.random.rand(128, 6).astype(np.float32)
        X, y = create_windows(data, label=1, window_size=128)
        assert X.shape[0] == 1

    def test_dtype_float32(self, sample_sensor_data):
        """Windows should be float32 for TFLite compatibility."""
        X, _ = create_windows(sample_sensor_data, label=0)
        assert X.dtype == np.float32

    def test_window_content_integrity(self):
        """First window should match the first 128 rows of input data."""
        data = np.arange(300 * 6, dtype=np.float32).reshape(300, 6)
        X, _ = create_windows(data, label=0, window_size=128)
        np.testing.assert_array_equal(X[0], data[:128])


# ---------------------------------------------------------------------------
# parse_txt_file tests
# ---------------------------------------------------------------------------

class TestParseTxtFile:
    def test_valid_file(self, tmp_path):
        """Should parse a valid comma-separated 6-column file."""
        p = tmp_path / "test.txt"
        lines = ["Header: IMU data\n"]
        lines += [f"{i*0.1:.2f},{i*0.2:.2f},{i*0.3:.2f},"
                  f"{i*0.4:.2f},{i*0.5:.2f},{i*0.6:.2f}\n"
                  for i in range(50)]
        p.write_text("".join(lines))

        df = parse_txt_file(str(p))
        assert df is not None
        assert df.shape == (50, 6)
        assert list(df.columns) == [
            "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"
        ]

    def test_empty_file_returns_none(self, tmp_path):
        """Empty or header-only file should return None."""
        p = tmp_path / "empty.txt"
        p.write_text("# This is a comment\nHeader: no data\n")
        result = parse_txt_file(str(p))
        assert result is None

    def test_missing_file(self):
        """Missing file should return None (not raise)."""
        result = parse_txt_file("/nonexistent/path/to/file.txt")
        assert result is None


# ---------------------------------------------------------------------------
# generate_dummy_data tests
# ---------------------------------------------------------------------------

class TestGenerateDummyData:
    def test_shape(self, config):
        X, y = generate_dummy_data(config)
        assert X.shape[1] == 128
        assert X.shape[2] == 6
        assert len(X) == len(y)

    def test_binary_labels(self, config):
        _, y = generate_dummy_data(config)
        assert set(np.unique(y)).issubset({0, 1})

    def test_with_subjects(self, config):
        X, y, subjects = generate_dummy_data(config, return_subjects=True)
        assert len(X) == len(y) == len(subjects)
        assert len(np.unique(subjects)) > 0
        assert subjects[0].startswith("sub")


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_load_and_preprocess_integration(self, tmp_path, config):
        """Should walk folder, parse files, segment into windows and extract subject IDs."""
        fol_dir = tmp_path / "FOL"
        std_dir = tmp_path / "STD"
        fol_dir.mkdir()
        std_dir.mkdir()

        # Write dummy txt files with name format: <activity>_<trial>_sub<id>.txt
        fall_file = fol_dir / "BSC_01_sub01.txt"
        adl_file = std_dir / "WAL_01_sub02.txt"

        # Must have at least window_size (128) samples
        rows_fall = [f"{i*0.01:.2f},{9.8+i*0.01:.2f},{i*0.02:.2f},0.01,0.01,0.01\n" for i in range(150)]
        rows_adl = [f"{i*0.01:.2f},{9.8+i*0.01:.2f},{i*0.02:.2f},0.01,0.01,0.01\n" for i in range(150)]
        fall_file.write_text("".join(rows_fall))
        adl_file.write_text("".join(rows_adl))

        from src.preprocess import load_and_preprocess
        X, y, subjects = load_and_preprocess(config, dataset_root=str(tmp_path))

        assert X.ndim == 3
        assert X.shape[1] == 128
        assert X.shape[2] == 6
        assert len(X) == len(y) == len(subjects)
        
        # Windows: 150 - 128 // 64 + 1 = 1 window per file, so 2 windows total
        assert X.shape[0] == 2
        assert set(np.unique(subjects)) == {"sub01", "sub02"}
        assert set(np.unique(y)) == {0, 1}
