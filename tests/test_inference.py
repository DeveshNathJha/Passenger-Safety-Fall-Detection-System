"""
tests/test_inference.py
------------------------
Smoke tests for FallDetectionEngine.
Uses the existing models/mobifall_edge_model.tflite without any network I/O.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import generate_dummy_csv
from src.utils.config_loader import load_config

# ---------------------------------------------------------------------------
# Skip TFLite-dependent tests if neither runtime is available
# ---------------------------------------------------------------------------
_TFLITE_AVAILABLE = True
try:
    try:
        import tflite_runtime.interpreter  # noqa
    except ImportError:
        import tensorflow  # noqa
    from src.inference import FallDetectionEngine
except (ImportError, ModuleNotFoundError):
    _TFLITE_AVAILABLE = False

tflite_required = pytest.mark.skipif(
    not _TFLITE_AVAILABLE,
    reason="Neither tflite_runtime nor tensorflow is installed.",
)

MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "mobifall_edge_model.tflite")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def engine(config):
    tflite_paths = {
        "human": MODEL_PATH,
        "car": MODEL_PATH  # Use same model just for the smoke test
    }
    return FallDetectionEngine(
        tflite_paths=tflite_paths,
        window_size=config.windowing.window_size,
        step=int(config.windowing.window_size * (1 - config.windowing.overlap)),
        threshold=config.model.prediction_threshold,
        labels={int(k): v for k, v in config.model.labels.items()},
        static_variance_threshold=config.model.static_variance_threshold,
    )


# ---------------------------------------------------------------------------
# Engine Initialization Tests
# ---------------------------------------------------------------------------

@tflite_required
class TestEngineInit:
    def test_engine_loads(self, engine):
        """Engine should initialize without errors."""
        assert engine is not None

    def test_invalid_model_path(self):
        """Should raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError):
            FallDetectionEngine(tflite_paths={"human": "/nonexistent/model.tflite"})

    def test_initial_buffer_empty(self, engine):
        engine.reset_buffer()
        assert engine.buffer_fill == 0


# ---------------------------------------------------------------------------
# push_sample / Buffer Tests
# ---------------------------------------------------------------------------

@tflite_required
class TestPushSample:
    def test_returns_none_before_window_fill(self, engine):
        """Buffer should return None until 128 samples are pushed."""
        engine.reset_buffer()
        sample = [0.1, 9.8, 0.3, 0.01, 0.02, 0.0]
        for _ in range(127):
            result = engine.push_sample(sample)
            assert result is None

    def test_returns_prediction_at_window_boundary(self, engine, config):
        """128th sample should trigger a prediction."""
        engine.reset_buffer()
        sample = [0.1, 9.8, 0.3, 0.01, 0.02, 0.0]
        result = None
        for _ in range(config.windowing.window_size):
            result = engine.push_sample(sample)
        assert result is not None, "Expected prediction after full window."

    def test_invalid_sample_length(self, engine):
        """Samples with wrong length should raise ValueError."""
        with pytest.raises(ValueError):
            engine.push_sample([1.0, 2.0, 3.0])  # only 3 values

    def test_buffer_increments(self, engine):
        engine.reset_buffer()
        engine.push_sample([0.0] * 6)
        assert engine.buffer_fill == 1


# ---------------------------------------------------------------------------
# Prediction Output Schema Tests
# ---------------------------------------------------------------------------

@tflite_required
class TestPredictionOutput:
    @pytest.fixture(scope="class")
    def prediction(self, engine, config):
        """Force a prediction by pushing a full window."""
        engine.reset_buffer()
        rng = np.random.default_rng(42)
        result = None
        for _ in range(config.windowing.window_size):
            row = rng.random(6).tolist()
            result = engine.push_sample(row)
        return result

    def test_has_required_keys(self, prediction):
        required = {"label", "confidence", "raw_score", "latency_ms"}
        assert required.issubset(prediction.keys())

    def test_label_is_valid_string(self, prediction, config):
        valid_labels = set(config.model.labels.values())
        assert prediction["label"] in valid_labels

    def test_confidence_in_range(self, prediction):
        assert 0.0 <= prediction["confidence"] <= 1.0

    def test_latency_is_positive(self, prediction):
        assert prediction["latency_ms"] > 0


# ---------------------------------------------------------------------------
# Direct Window Prediction Tests
# ---------------------------------------------------------------------------

@tflite_required
class TestDirectPredict:
    def test_shape_validation(self, engine):
        """Window with wrong shape should raise ValueError."""
        bad_window = np.random.rand(64, 6).astype(np.float32)  # too short
        with pytest.raises(ValueError):
            engine.predict(bad_window)

    def test_predict_on_zeros(self, engine, config):
        """Engine should return a result without crashing on zero data."""
        window = np.zeros((config.windowing.window_size, 6), dtype=np.float32)
        result = engine.predict(window)
        assert "label" in result


# ---------------------------------------------------------------------------
# Stats Tests
# ---------------------------------------------------------------------------

@tflite_required
class TestEngineStats:
    def test_stats_structure(self, engine):
        stats = engine.get_stats()
        for key in ["active_mode", "available_modes", "total_inferences", "fall_count", "buffer_fill"]:
            assert key in stats


# ---------------------------------------------------------------------------
# Heuristic Guardrail Tests
# ---------------------------------------------------------------------------

@tflite_required
class TestHeuristicGuardrail:
    def test_stationary_phone_forces_normal(self, engine, config):
        """If variance is very low, it should override predictions to normal."""
        window = np.zeros((config.windowing.window_size, 6), dtype=np.float32)
        window[:, 1] = 9.8  # gravity on Y, completely constant (std = 0)
        
        result = engine.predict(window)
        assert result["label"] == config.model.labels[0]
        assert result["raw_score"] == 0.01
        assert result["confidence"] == 0.99


# ---------------------------------------------------------------------------
# Dummy CSV Generation Test
# ---------------------------------------------------------------------------

class TestDummyCsvGeneration:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "test.csv")
        generate_dummy_csv(out, n_rows=200)
        assert Path(out).exists()

    def test_row_count(self, tmp_path):
        import csv
        out = str(tmp_path / "test.csv")
        generate_dummy_csv(out, n_rows=200)
        with open(out) as f:
            rows = list(csv.reader(f))
        # 1 header + 200 data rows
        assert len(rows) == 201
