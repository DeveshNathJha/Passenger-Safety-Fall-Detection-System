"""Tests for the splitting / metric helpers in src/train.py (needs TensorFlow)."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None,
                                reason="tensorflow not installed")


@pytest.fixture(scope="module")
def train_mod():
    from src import train
    return train


def mobifall_like_groups():
    """9 subjects with normal + fall windows, 15 subjects with falls only (like MobiFall v2.0)."""
    groups, y = [], []
    for s in range(24):
        if s < 9:
            groups += [f"sub{s:02d}"] * 12
            y += [0] * 10 + [1] * 2
        else:
            groups += [f"sub{s:02d}"] * 2
            y += [1, 1]
    return np.array(groups), np.array(y)


def test_split_is_disjoint_and_stratified(train_mod):
    groups, y = mobifall_like_groups()
    for seed in range(20):
        tr, va, te = train_mod.group_split(groups, y, 0.2, 0.15, seed)
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
        assert (tr | va | te).all()
        # validation and test must both contain normal windows (needed for threshold tuning)
        assert (y[va] == 0).any() and (y[te] == 0).any()
        assert (y[va] == 1).any() and (y[te] == 1).any()


def test_tune_threshold_maximises_f1(train_mod):
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    thr = train_mod.tune_threshold(y, scores)
    assert 0.7 < thr <= 0.8


def test_metrics_and_pooled_summary(train_mod):
    y = np.array([0, 0, 1, 1])
    m = train_mod.compute_metrics(y, np.array([0.1, 0.9, 0.8, 0.2]), 0.5, step_s=1.28)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == m["recall"] == 0.5

    only_falls = train_mod.compute_metrics(np.array([1, 1]), np.array([0.9, 0.9]), 0.5, 1.28)
    summary = train_mod.summarize_folds([m, only_falls], 1.28)
    assert summary["pooled"]["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 3}
    assert summary["per_fold"]["n_folds_with_both_classes"] == 1


def test_combined_training_data_keeps_subjects_distinct(train_mod, monkeypatch):
    data = {
        "human": (np.zeros((4, 128, 6)), np.array([0, 1, 0, 1]), np.array(["sub01", "sub01", "sub02", "sub02"])),
        "sisfall": (np.ones((2, 128, 6)), np.array([0, 1]), np.array(["SA01", "SE01"])),
    }
    monkeypatch.setattr(train_mod, "load_processed", lambda config, task: data[task])
    X, y, g = train_mod.load_training_data(None, "human", "mobifall")
    assert len(X) == 4
    X, y, g = train_mod.load_training_data(None, "human", "mobifall+sisfall")
    assert len(X) == 6 and set(g) == {"sub01", "sub02", "SA01", "SE01"}
    assert y.tolist() == [0, 1, 0, 1, 0, 1]
