"""Tests for the post-fall stillness check in StreamSession."""

import numpy as np
import pytest

from src.inference import PostFallCheck, StreamSession, post_fall_check_from_config
from tests.conftest import FakeModel, jogging, walking

CHECK = PostFallCheck(settle_s=1.0, still_s=1.5, still_acc_std=1.5, still_gyro_dps=30.0)


def spike(rows: np.ndarray, at: int) -> np.ndarray:
    rows = rows.copy()
    rows[at, 0:3] = [35, 30, 10]
    return rows


def still(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.tile(np.array([0.0, 0.0, 9.81, 0, 0, 0], dtype=np.float32), (n, 1))
    rows[:, 0:3] += rng.normal(0, 0.05, (n, 3))
    rows[:, 3:6] += rng.normal(0, 0.5, (n, 3))
    return rows


def run(session: StreamSession, rows: np.ndarray) -> list[dict]:
    return [r for row in rows for r in session.push(row)]


def make(check=CHECK) -> StreamSession:
    return StreamSession(FakeModel("human"), step=64, post_fall_check=check)


def test_spike_then_stillness_is_confirmed():
    s = make()
    out = run(s, np.concatenate([spike(walking(128), 100), still(200)]))
    windows = [r for r in out if r["kind"] == "window"]
    confs = [r for r in out if r["kind"] == "confirmation"]
    assert windows[0]["candidate"] and not windows[0]["is_event"]
    assert len(confs) == 1 and confs[0]["is_event"] and confs[0]["label"] == "EVENT"
    assert confs[0]["impact_sample"] == 100
    # decided exactly when impact + settle + still samples are available
    assert confs[0]["delay_s"] == pytest.approx((50 + 75 - 1) / 50, abs=0.02)
    assert s.event_count == 1 and s.candidate_count == 1


def test_spike_then_movement_is_rejected():
    s = make()
    out = run(s, np.concatenate([spike(walking(128), 100), jogging(200, seed=7)]))
    confs = [r for r in out if r["kind"] == "confirmation"]
    assert len(confs) == 1 and confs[0]["is_event"] is False
    assert confs[0]["stillness"]["gyro_mean"] > 30 or confs[0]["stillness"]["acc_std"] > 1.5
    assert s.event_count == 0


def test_disabled_check_keeps_old_behaviour():
    s = make(check=None)
    out = run(s, spike(walking(128), 100))
    assert len(out) == 1 and out[0]["is_event"] and not out[0]["candidate"]
    assert s.event_count == 1


def test_only_one_candidate_per_impact():
    """The spike stays in 2 consecutive windows; it must produce one candidate, not two."""
    s = make()
    out = run(s, np.concatenate([spike(walking(128), 100), still(300)]))
    assert sum(r.get("candidate", False) for r in out) == 1
    assert sum(r["kind"] == "confirmation" for r in out) == 1


def test_new_fall_after_the_check_period_is_a_new_candidate():
    s = make()
    rows = np.concatenate([spike(walking(128), 100), still(300), spike(walking(128, seed=2), 20), still(200)])
    confs = [r for r in run(s, rows) if r["kind"] == "confirmation"]
    assert len(confs) == 2 and all(c["is_event"] for c in confs)
    assert s.event_count == 2


def test_pending_candidate_is_not_an_event_yet():
    s = make()
    run(s, spike(walking(128), 100))  # recording ends before the still-period
    assert s.event_count == 0 and s.candidate_count == 1


def test_reset_clears_pending_candidate():
    s = make()
    run(s, spike(walking(128), 100))
    s.reset()
    assert not [r for r in run(s, still(300)) if r["kind"] == "confirmation"]


def test_config_wiring(base_config):
    cfg = base_config.inference.post_fall_check
    check = post_fall_check_from_config(base_config, "human")
    assert (check is not None) == bool(cfg.enabled and "human" in cfg.modes)
    assert post_fall_check_from_config(base_config, "plane") is None
