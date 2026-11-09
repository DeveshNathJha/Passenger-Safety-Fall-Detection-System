"""
src/tflite_converter.py
-----------------------
Convert a trained Keras model to TFLite (dynamic-range int8 weight
quantization) for edge deployment, and verify the converted model still
agrees with the original.

Standalone usage:
    python src/tflite_converter.py --task human
    python src/tflite_converter.py --input models/human_fall.keras --output models/human_fall.tflite
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.model_layers  # noqa: E402,F401  (registers custom layers for load_model)
from src.utils.config_loader import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conversion Logic
# ---------------------------------------------------------------------------

def _convert(model: tf.keras.Model, quantize: bool) -> bytes:
    # Fixed batch size 1 (the server scores one window at a time). With a
    # dynamic batch, Keras LSTMs lower to TensorList ops that need the Flex
    # delegate; with batch 1 they map to the builtin LSTM kernel.
    inp = tf.keras.Input(shape=model.input_shape[1:], batch_size=1)
    fixed = tf.keras.Model(inp, model(inp, training=False))

    def make():
        c = tf.lite.TFLiteConverter.from_keras_model(fixed)
        if quantize:
            c.optimizations = [tf.lite.Optimize.DEFAULT]
        return c

    # Built-in ops only: the result runs on ai-edge-litert / tflite-runtime
    # without the (large) Flex delegate.
    try:
        return make().convert()
    except Exception as exc:
        logger.warning(f"Builtin-only conversion failed ({exc}); retrying with SELECT_TF_OPS "
                       "(model will need the Flex delegate at runtime).")
    converter = make()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    try:
        return converter.convert()
    except Exception as exc:
        raise RuntimeError(f"TFLite conversion failed: {exc}") from exc


def convert_to_tflite(model, tflite_path: str, quantize: bool = True) -> dict:
    """
    Convert a Keras model (object or path to .keras/.h5) to TFLite.

    Returns
    -------
    dict
        tflite_size_kb, float32_size_kb (unquantized TFLite, the fair
        baseline for the quantization saving), reduction_pct.
    """
    if isinstance(model, (str, os.PathLike)):
        if not os.path.exists(model):
            raise FileNotFoundError(f"Source model not found: {model}")
        model = tf.keras.models.load_model(model)

    float_bytes = _convert(model, quantize=False)
    out_bytes = _convert(model, quantize=True) if quantize else float_bytes

    os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(out_bytes)

    metrics = {
        "tflite_size_kb": round(len(out_bytes) / 1024, 2),
        "float32_size_kb": round(len(float_bytes) / 1024, 2),
        "reduction_pct": round((1 - len(out_bytes) / len(float_bytes)) * 100, 1),
    }
    logger.info(f"TFLite → {tflite_path}: {metrics['tflite_size_kb']} KB "
                f"(float32 {metrics['float32_size_kb']} KB, −{metrics['reduction_pct']}%)")
    return metrics


def tflite_parity(model: tf.keras.Model, tflite_path: str, X: np.ndarray, threshold: float) -> dict:
    """
    Compare Keras and TFLite outputs on the same windows.

    Returns max |score difference| and the fraction of identical decisions.
    """
    from src.inference import TFLiteModel

    if len(X) == 0:
        return {}
    keras_scores = model.predict(X, verbose=0).ravel()
    tfl = TFLiteModel("parity", tflite_path)
    tfl_scores = np.array([tfl.score(w)[0] for w in X])
    agree = float(np.mean((keras_scores >= threshold) == (tfl_scores >= threshold)))
    result = {
        "n_windows": int(len(X)),
        "max_abs_score_diff": round(float(np.max(np.abs(keras_scores - tfl_scores))), 5),
        "decision_agreement": round(agree, 4),
    }
    logger.info(f"TFLite parity: {result}")
    return result


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert a Keras model to quantized TFLite.")
    parser.add_argument("--task", choices=["human", "car"], default="human")
    parser.add_argument("--input", type=str, default=None, metavar="KERAS_PATH")
    parser.add_argument("--output", type=str, default=None, metavar="TFLITE_PATH")
    parser.add_argument("--no-quantize", action="store_true", help="Keep float32 weights.")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    spec = config.models[args.task]
    convert_to_tflite(
        args.input or str(resolve_path(spec.keras)),
        args.output or str(resolve_path(spec.tflite)),
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
