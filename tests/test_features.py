"""
Tests for the orientation-invariant input features: the NumPy reference
(src/signal_processing.invariant_features) and the Keras layer exported into
the TFLite model (src/model_layers.GravityAlignedFeatures).
"""

import importlib.util

import numpy as np
import pytest

from src.signal_processing import augment_rotation, invariant_features

HAVE_TF = importlib.util.find_spec("tensorflow") is not None
tf_required = pytest.mark.skipif(not HAVE_TF, reason="tensorflow not installed")


def imu_windows(n=8, seed=0):
    """Gravity on a random axis + motion + one impact spike, like real windows."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 2, (n, 128, 6)).astype(np.float32)
    X[:, :, 1] += 9.81
    X[:, 60, 0:3] += 30.0
    X[:, :, 3:6] *= 20
    return X


def test_shape_and_single_window():
    X = imu_windows()
    F = invariant_features(X)
    assert F.shape == X.shape and F.dtype == np.float32
    np.testing.assert_allclose(invariant_features(X[0]), F[0], rtol=1e-6)


def test_numpy_features_are_rotation_invariant():
    X = imu_windows()
    Xr = augment_rotation(X, np.random.default_rng(1))
    np.testing.assert_allclose(invariant_features(Xr), invariant_features(X), atol=1e-3)


def test_vertical_component_of_still_phone_is_gravity():
    still = np.tile([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], (128, 1)).astype(np.float32)
    f = invariant_features(still)
    np.testing.assert_allclose(f[:, 0], 9.81, atol=1e-3)   # |a|
    np.testing.assert_allclose(f[:, 1], 9.81, atol=1e-3)   # vertical
    np.testing.assert_allclose(f[:, 2], 0.0, atol=1e-2)    # horizontal


@tf_required
def test_keras_layer_matches_numpy():
    from src.model_layers import GravityAlignedFeatures
    X = imu_windows()
    out = GravityAlignedFeatures()(X).numpy()
    np.testing.assert_allclose(out, invariant_features(X), atol=1e-4, rtol=1e-4)


@tf_required
def test_keras_layer_is_rotation_invariant():
    from src.model_layers import GravityAlignedFeatures
    X = imu_windows()
    Xr = augment_rotation(X, np.random.default_rng(2))
    layer = GravityAlignedFeatures()
    np.testing.assert_allclose(layer(Xr).numpy(), layer(X).numpy(), atol=1e-3)


@pytest.fixture(scope="module")
def invariant_model(base_config):
    import copy

    from src.train import build_model
    from src.utils.config_loader import AttrDict
    cfg = AttrDict(copy.deepcopy(dict(base_config)))
    cfg.model.input_features = "invariant"
    return build_model("1D-CNN", imu_windows(64), cfg)


@tf_required
def test_invariant_model_tflite_builtin_ops_and_rotation_invariance(invariant_model, tmp_path):
    from src.inference import TFLiteModel
    from src.tflite_converter import convert_to_tflite

    path = str(tmp_path / "inv.tflite")
    convert_to_tflite(invariant_model, path, quantize=True)
    tfl = TFLiteModel("inv", path)  # builtin interpreter: fails if the Flex delegate were needed

    X = imu_windows(4, seed=3)
    keras_scores = invariant_model.predict(X, verbose=0).ravel()
    tfl_scores = np.array([tfl.score(w)[0] for w in X])
    np.testing.assert_allclose(tfl_scores, keras_scores, atol=1e-3)

    Xr = augment_rotation(X, np.random.default_rng(4))
    rot_scores = np.array([tfl.score(w)[0] for w in Xr])
    np.testing.assert_allclose(rot_scores, tfl_scores, atol=1e-3)


@tf_required
def test_keras_save_load_roundtrip(invariant_model, tmp_path):
    import tensorflow as tf

    import src.model_layers  # noqa: F401  (registers the custom layer)
    path = tmp_path / "inv.keras"
    invariant_model.save(path)
    loaded = tf.keras.models.load_model(path)
    X = imu_windows(3, seed=5)
    np.testing.assert_allclose(loaded.predict(X, verbose=0), invariant_model.predict(X, verbose=0), atol=1e-6)
