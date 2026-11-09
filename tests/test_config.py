"""Tests for the config loader."""

import pytest

from src.utils.config_loader import AttrDict, load_config, resolve_path


def test_nested_writes_persist():
    """Regression: v1 returned a fresh copy on every access, so CLI overrides were silently lost."""
    cfg = AttrDict({"model": {"epochs": 10, "layers": [{"units": 8}]}})
    cfg.model.epochs = 3
    cfg.model["batch_size"] = 16
    assert cfg.model.epochs == 3 and cfg["model"]["batch_size"] == 16
    assert cfg.model.layers[0].units == 8


def test_missing_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = AttrDict({}).nope


def test_default_config_loads_and_is_consistent():
    cfg = load_config()
    assert cfg.windowing.window_size == cfg.model.input_shape[0] == 128
    assert set(cfg.models) == {"human", "car"}
    assert 0 < cfg.inference.static_variance_threshold < 0.2


def test_env_override(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("api: {port: 9999}\n")
    monkeypatch.setenv("FDS_CONFIG", str(p))
    assert load_config().api.port == 9999


def test_resolve_path_is_project_relative():
    assert resolve_path("models/x.tflite").is_absolute()
