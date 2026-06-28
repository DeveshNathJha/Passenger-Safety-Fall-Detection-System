"""
src/tflite_converter.py
-----------------------
Convert a trained Keras (.h5) model to a quantized TFLite (.tflite) format
for edge deployment (Raspberry Pi, Android MPU6050).

Standalone usage:
    python src/tflite_converter.py
    python src/tflite_converter.py --input models/mobifall_model.h5
    python src/tflite_converter.py --input models/mobifall_model.h5 \\
                                   --output models/mobifall_edge_model.tflite
"""

import argparse
import os
import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conversion Logic
# ---------------------------------------------------------------------------

def convert_to_tflite(
    h5_path: str,
    tflite_path: str,
    quantize: bool = True,
) -> dict:
    """
    Load a Keras model and convert it to TFLite with optional quantization.

    Parameters
    ----------
    h5_path : str
        Path to the source Keras .h5 model.
    tflite_path : str
        Destination path for the .tflite output file.
    quantize : bool
        If True (default), apply dynamic range quantization.

    Returns
    -------
    dict
        Conversion metrics: original_size_kb, tflite_size_kb, reduction_pct.

    Raises
    ------
    FileNotFoundError
        If the source .h5 file does not exist.
    RuntimeError
        If the conversion process fails.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Source model not found: {h5_path}")

    logger.info(f"Loading Keras model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path)
    logger.info(
        f"Model loaded — Architecture: {model.name} | "
        f"Parameters: {model.count_params():,}"
    )

    # --- TFLite Conversion ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("Dynamic range quantization: ENABLED")
    else:
        logger.info("Quantization: DISABLED (float32 model)")

    logger.info("Starting TFLite conversion …")
    try:
        tflite_model = converter.convert()
    except Exception as exc:
        raise RuntimeError(f"TFLite conversion failed: {exc}") from exc

    # --- Persist to disk ---
    os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # --- Size comparison ---
    size_h5_kb = os.path.getsize(h5_path) / 1024
    size_tflite_kb = os.path.getsize(tflite_path) / 1024
    reduction = (1 - size_tflite_kb / size_h5_kb) * 100

    metrics = {
        "original_size_kb": round(size_h5_kb, 2),
        "tflite_size_kb": round(size_tflite_kb, 2),
        "reduction_pct": round(reduction, 1),
    }

    logger.info(
        f"Conversion successful!\n"
        f"  Source     : {h5_path} ({size_h5_kb:.1f} KB)\n"
        f"  Destination: {tflite_path} ({size_tflite_kb:.1f} KB)\n"
        f"  Size reduction: {reduction:.1f}%"
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Keras .h5 model to quantized TFLite."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="H5_PATH",
        help="Path to source .h5 Keras model (default from config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="TFLITE_PATH",
        help="Destination path for the .tflite file (default from config).",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization (keep float32 weights).",
    )
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)

    h5_path = args.input or config.paths.model_h5
    tflite_path = args.output or config.paths.model_tflite_human

    metrics = convert_to_tflite(
        h5_path=h5_path,
        tflite_path=tflite_path,
        quantize=not args.no_quantize,
    )
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
