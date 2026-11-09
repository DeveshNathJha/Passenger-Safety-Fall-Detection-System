"""
Tests against the real exported .tflite models and their metadata.
Skipped automatically if no TFLite runtime is installed.
"""

import importlib.util
import json

import numpy as np
import pytest

from src.inference import TFLiteModel
from src.utils.config_loader import resolve_path

HAVE_RUNTIME = any(importlib.util.find_spec(m) for m in ("ai_edge_litert", "tflite_runtime", "tensorflow"))

pytestmark = pytest.mark.skipif(not HAVE_RUNTIME, reason="no TFLite runtime installed")


@pytest.fixture(scope="module", params=["human", "car"])
def model(request, base_config):
    spec = base_config.models[request.param]
    return TFLiteModel(request.param, str(resolve_path(spec.tflite)), str(resolve_path(spec.metadata)),
                       labels=spec.labels)


def test_metadata_is_complete(model):
    md = model.metadata
    for key in ("architecture", "threshold", "test_metrics", "split", "tflite_parity", "sample_rate_hz"):
        assert key in md, key
    assert md["window_size"] == model.window_size == 128
    assert md["tflite_parity"]["decision_agreement"] >= 0.99
    # train / val / test never share a subject or sequence
    s = md["split"]
    assert not set(s["train_groups"]) & set(s["test_groups"])
    assert not set(s["val_groups"]) & set(s["test_groups"])


def test_edge_model_is_small(model):
    assert model.info()["size_kb"] < 64


def test_scores_are_probabilities(model):
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = rng.normal(0, 3, (128, 6)).astype(np.float32)
        w[:, 2] += 9.81
        score, latency = model.score(w)
        assert 0.0 <= score <= 1.0 and latency >= 0


def test_metrics_file_matches_metadata(model, base_config):
    report = resolve_path(base_config.paths.reports_dir) / f"{model.name}_metrics.json"
    with open(report, encoding="utf-8") as f:
        assert json.load(f)["threshold"] == model.metadata["threshold"]
