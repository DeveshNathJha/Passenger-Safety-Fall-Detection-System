"""Tests for the event-level replay evaluation (src/evaluate_events.py)."""

import numpy as np

from src.evaluate_events import CachedModel, evaluate, filter_recordings, grid_search
from src.inference import PostFallCheck
from src.preprocess import Recording
from tests.conftest import FakeModel, jogging, walking


def lying(n: int) -> np.ndarray:
    rows = np.tile(np.array([0.0, 0.0, 9.81, 0, 0, 0], dtype=np.float32), (n, 1))
    rows[:, 0:3] += np.random.default_rng(1).normal(0, 0.05, (n, 3))
    return rows


def fall_recording(subject="SA01", trial="R01"):
    data = np.concatenate([walking(200), lying(250)])
    data[200, 0:3] = [35, 30, 10]  # impact, then lying still
    return Recording("F01", subject, trial, 1, data)


def jump_recording(subject="SA01", trial="R01"):
    data = jogging(450, seed=4)
    data[200, 0:3] = [35, 30, 10]  # spike, but the person keeps moving
    return Recording("D19", subject, trial, 0, data)


def walk_recording(subject="SE01"):
    return Recording("D01", subject, "R01", 0, walking(600, seed=5))


RECS = [fall_recording(), jump_recording(), walk_recording()]


def test_check_off_counts_jump_as_false_alarm(base_config):
    m = evaluate(FakeModel("human"), RECS, base_config, None)
    assert m["fall_recall"] == 1.0
    assert m["false_alarms"] == 1 and m["false_alarms_by_code"]["D19"]["false_alarms"] == 1
    assert m["false_alarms_by_code"]["D01"]["false_alarms"] == 0
    assert m["adl_hours"] > 0 and m["false_alarms_per_hour"] > 0


def test_check_on_keeps_fall_and_removes_jump(base_config):
    m = evaluate(FakeModel("human"), RECS, base_config, PostFallCheck())
    assert m["fall_recall"] == 1.0
    assert m["false_alarms"] == 0
    assert m["missed_falls_by_code"] == {}


def test_filter_by_prefix_and_subjects():
    assert [r.subject for r in filter_recordings(RECS, prefix="SE")] == ["SE01"]
    assert len(filter_recordings(RECS, subjects=["SA01"])) == 2


def test_cached_model_scores_once():
    fake = FakeModel("human")
    cm = CachedModel(fake)
    w = walking(128)
    assert cm.score(w) == cm.score(w)
    assert fake.calls == 1


def test_grid_search_respects_recall_floor(base_config):
    grid = {"settle_s": [1.0], "still_s": [1.5], "still_acc_std": [0.01, 1.5], "still_gyro_dps": [30.0]}
    gs = grid_search(FakeModel("human"), RECS, base_config, grid)
    assert len(gs["rows"]) == 2
    # acc_std 0.01 is too strict (rejects the real fall) → infeasible; 1.5 wins
    assert gs["best"]["still_acc_std"] == 1.5 and gs["best"]["false_alarms"] == 0
