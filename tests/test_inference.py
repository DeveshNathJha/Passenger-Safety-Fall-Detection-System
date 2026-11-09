"""Tests for StreamSession (per-client sliding-window inference)."""

import numpy as np
import pytest

from src.inference import StreamSession, TFLiteModel, generate_dummy_csv, run_simulation
from tests.conftest import FakeModel, walking


@pytest.fixture
def session():
    return StreamSession(FakeModel("human"), step=64, static_variance_threshold=0.05)


class TestBuffering:
    def test_no_prediction_until_window_full(self, session):
        for row in walking(127):
            assert session.push(row) == []
        assert session.buffer_fill == 127

    def test_prediction_at_window_then_every_step(self, session):
        preds = [i for i, row in enumerate(walking(128 + 64 * 3)) if session.push(row)]
        assert preds == [127, 191, 255, 319]

    def test_invalid_sample_length(self, session):
        with pytest.raises(ValueError):
            session.push([1.0, 2.0, 3.0])

    def test_reset(self, session):
        for row in walking(50):
            session.push(row)
        session.reset()
        assert session.buffer_fill == 0

    def test_timestamped_100hz_input_is_resampled(self, session):
        rows = walking(256 * 2)
        t = np.arange(len(rows)) * 10.0  # 100 Hz in ms
        n_pred = sum(len(session.push(r, t_ms=ti)) for r, ti in zip(rows, t))
        assert session.total_samples in range(255, 258)  # 5.12 s at 50 Hz
        assert n_pred == 3


class TestPredictionSchema:
    def test_keys_and_values(self, session):
        out = [p for row in walking(128) for p in session.push(row)][0]
        assert {"label", "is_event", "confidence", "raw_score", "threshold",
                "guardrail_fired", "latency_ms", "total_inferences", "event_count"} <= set(out)
        assert out["label"] == "Normal" and out["is_event"] is False
        assert 0.0 <= out["confidence"] <= 1.0

    def test_spike_is_event(self, session):
        rows = walking(128)
        rows[100, 0:3] = [30, 30, 10]
        out = [p for row in rows for p in session.push(row)][0]
        assert out["is_event"] and out["label"] == "EVENT"
        assert session.event_count == 1

    def test_stationary_guardrail_forces_normal(self):
        model = FakeModel("human")
        model.score = lambda w: (0.99, 0.1)  # model would say "event"
        s = StreamSession(model, step=64, static_variance_threshold=0.05)
        still = np.tile([0.0, 0.0, 9.81, 0, 0, 0], (128, 1))
        out = [p for row in still for p in s.push(row)][0]
        assert out["guardrail_fired"] and not out["is_event"] and out["raw_score"] == 0.99


class TestConfirmation:
    def test_k_of_n_requires_consecutive_hits(self):
        model = FakeModel("human")
        model.score = lambda w: (0.9, 0.1)
        s = StreamSession(model, step=64, confirm_k=2, confirm_n=3)
        preds = [p for row in walking(128 + 64 * 2) for p in s.push(row)]
        assert [p["is_event"] for p in preds] == [False, True, False]


def test_sessions_are_isolated():
    """Regression: v1 shared ONE global buffer between every connected phone."""
    model = FakeModel("human")
    a = StreamSession(model, step=64)
    b = StreamSession(model, step=64)
    for row in walking(100):
        a.push(row)
    assert a.buffer_fill == 100 and b.buffer_fill == 0


def test_simulation_on_dummy_csv(tmp_path):
    csv_path = tmp_path / "dummy.csv"
    generate_dummy_csv(str(csv_path), n_rows=512)
    stats = run_simulation(StreamSession(FakeModel("human"), step=64), str(csv_path), simulate_realtime=False)
    assert stats["total_inferences"] == 7
    assert stats["event_count"] >= 1  # the synthetic impact is detected


def test_missing_model_file():
    with pytest.raises(FileNotFoundError):
        TFLiteModel("human", "/nonexistent/model.tflite")
