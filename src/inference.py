"""
src/inference.py
----------------
Stateful fall detection inference engine for edge deployment.

Key design:
  - FallDetectionEngine wraps a TFLite interpreter behind a thread-safe push_sample() API
  - Internally maintains a NumPy ring buffer of `window_size` samples
  - Every time the buffer fills up (128 samples), it runs inference and resets by `step` samples
  - Designed for real-time 50 Hz sensor streams (smartphone, MPU6050, Raspberry Pi)

Standalone usage:
    # Simulate real-time inference on a CSV file:
    python src/inference.py --model models/mobifall_edge_model.tflite --data data/test.csv

    # Generate a dummy test CSV then run:
    python src/inference.py --generate-dummy
"""

import argparse
import csv
import os
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TFLite Interpreter Helper
# ---------------------------------------------------------------------------

def _load_interpreter(tflite_path: str):
    """
    Load a TFLite interpreter. Tries in order:
      1. tflite_runtime  – lightweight, for edge devices (Raspberry Pi)
      2. tensorflow.lite – full TF, for development machines

    Raises
    ------
    RuntimeError
        With a clear install command if neither package is found.
    """
    # --- Try tflite_runtime (edge devices) ---
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
        interp = tflite.Interpreter(model_path=tflite_path)
        logger.info("Using tflite_runtime interpreter (edge mode).")
        return interp
    except ImportError:
        pass

    # --- Try tensorflow.lite (dev machines) ---
    try:
        import tensorflow as tf  # type: ignore
        interp = tf.lite.Interpreter(model_path=tflite_path)
        logger.info("Using tensorflow.lite interpreter (development mode).")
        return interp
    except ImportError:
        pass

    # --- Neither available ---
    raise RuntimeError(
        "\n\n"
        "  No TFLite runtime found. Install one of the following:\n\n"
        "  Development machine (recommended):\n"
        "      pip install tensorflow-cpu\n\n"
        "  Edge device (Raspberry Pi / ARM):\n"
        "      pip install tflite-runtime\n\n"
        "Then restart the server."
    )



# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class FallDetectionEngine:
    """
    Stateful, thread-safe inference engine for fall detection.

    Push raw sensor samples one at a time via push_sample().
    When the internal buffer accumulates `window_size` samples, the engine
    automatically runs TFLite inference and returns a prediction dict.

    Parameters
    ----------
    tflite_path : str
        Path to the .tflite model file.
    window_size : int
        Number of timesteps per inference window (default: 128).
    step : int
        Number of new samples needed to trigger the next inference.
        step = window_size * (1 - overlap).  Default: 64 (50% overlap).
    threshold : float
        Sigmoid output threshold above which label = "FALL" (default: 0.5).
    labels : dict
        Maps int label → string label name.
    """

    def __init__(
        self,
        tflite_paths: dict[str, str],
        window_size: int = 128,
        step: int = 64,
        threshold: float = 0.5,
        labels: Optional[dict] = None,
        static_variance_threshold: float = 0.05,  # 0.05 = only fire for truly stationary phone
    ):
        self.tflite_paths = tflite_paths
        self.active_mode = "human"  # default mode
        self.window_size = window_size
        self.step = step
        self.threshold = threshold
        self.labels = labels or {0: "Normal Activity", 1: "FALL DETECTED"}
        self.static_variance_threshold = static_variance_threshold

        # --- TFLite setup for all modes ---
        self._interpreters = {}
        self._input_details = {}
        self._output_details = {}
        
        for mode, path in tflite_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"TFLite model not found for mode '{mode}': {path}")
            
            interp = _load_interpreter(path)
            interp.allocate_tensors()
            self._interpreters[mode] = interp
            self._input_details[mode] = interp.get_input_details()[0]
            self._output_details[mode] = interp.get_output_details()[0]

        # --- Ring buffer (stores the last `window_size` samples) ---
        self._buffer: deque = deque(maxlen=window_size)
        self._samples_since_last_predict = 0
        self._lock = threading.Lock()

        # --- Stats ---
        self.total_inferences = 0
        self.fall_count = 0

        logger.info(
            f"FallDetectionEngine ready — modes: {list(tflite_paths.keys())} | "
            f"window: {window_size} | step: {step} | threshold: {threshold}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        """Hot-swap the active inference model."""
        if mode not in self._interpreters:
            raise ValueError(f"Unknown mode '{mode}'. Available: {list(self._interpreters.keys())}")
        with self._lock:
            self.active_mode = mode
            self._buffer.clear()
            self._samples_since_last_predict = 0
            logger.info(f"Engine mode switched to: {mode}")

    def push_sample(self, sample: list[float]) -> Optional[dict]:
        """
        Push one 6-axis IMU sample into the ring buffer.

        Parameters
        ----------
        sample : list[float]
            Six values: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].

        Returns
        -------
        dict | None
            Prediction dict when a full window is ready, else None.
            Structure: {"label": str, "confidence": float, "latency_ms": float,
                        "total_inferences": int, "fall_count": int}
        """
        if len(sample) != 6:
            raise ValueError(f"Expected 6 IMU values, got {len(sample)}.")

        with self._lock:
            self._buffer.append(sample)
            self._samples_since_last_predict += 1

            # Only run inference once window is full AND enough new samples have arrived
            if (
                len(self._buffer) == self.window_size
                and self._samples_since_last_predict >= self.step
            ):
                self._samples_since_last_predict = 0
                return self.predict(np.array(self._buffer, dtype=np.float32))

        return None

    def predict(self, window: np.ndarray) -> dict:
        """
        Run TFLite inference on a pre-built window.

        Parameters
        ----------
        window : np.ndarray
            Shape (window_size, 6), dtype float32.

        Returns
        -------
        dict
            {"label", "confidence", "raw_score", "acc_mag_std",
             "guardrail_fired", "latency_ms", "total_inferences", "fall_count"}
        """
        if window.shape != (self.window_size, 6):
            raise ValueError(
                f"Expected window shape ({self.window_size}, 6), got {window.shape}."
            )

        mode = self.active_mode
        interp = self._interpreters[mode]
        in_det = self._input_details[mode]
        out_det = self._output_details[mode]

        input_tensor = window.reshape(1, self.window_size, 6).astype(in_det["dtype"])

        t_start = time.perf_counter()
        interp.set_tensor(in_det["index"], input_tensor)
        interp.invoke()
        raw_score = float(interp.get_tensor(out_det["index"])[0][0])
        latency_ms = (time.perf_counter() - t_start) * 1000

        label_idx = int(raw_score >= self.threshold)

        # ── Stationary Phone Guardrail ─────────────────────────────────────
        # If the phone is completely still (e.g. lying flat on a table),
        # its constant gravity vector looks anomalous to the model.
        # Only override if variance is VERY low (threshold=0.05).
        # NOTE: Real walking has acc_mag std ~0.2, so this only fires
        # for genuinely motionless phones (~0.01-0.03 std).
        acc_mag = np.linalg.norm(window[:, 0:3], axis=1)
        acc_mag_std = float(np.std(acc_mag))
        guardrail_fired = False

        if acc_mag_std < self.static_variance_threshold:
            guardrail_fired = True
            label_idx = 0
            raw_score = 0.01  # Override to high-confidence Normal
            logger.debug(
                f"[Guardrail FIRED] acc_mag_std={acc_mag_std:.4f} < "
                f"threshold={self.static_variance_threshold} → forced Normal"
            )
        else:
            logger.debug(
                f"[Model] raw_score={raw_score:.4f} | acc_mag_std={acc_mag_std:.4f} | "
                f"threshold={self.threshold} → {'FALL' if label_idx == 1 else 'Normal'}"
            )

        label = self.labels[label_idx]
        confidence = raw_score if label_idx == 1 else 1.0 - raw_score

        self.total_inferences += 1
        if label_idx == 1:
            self.fall_count += 1

        result = {
            "label": label,
            "confidence": round(confidence, 4),
            "raw_score": round(raw_score, 4),
            "acc_mag_std": round(acc_mag_std, 4),
            "guardrail_fired": guardrail_fired,
            "latency_ms": round(latency_ms, 2),
            "total_inferences": self.total_inferences,
            "fall_count": self.fall_count,
        }
        return result

    def reset_buffer(self):
        """Clear the ring buffer (e.g., on passenger change or journey start)."""
        with self._lock:
            self._buffer.clear()
            self._samples_since_last_predict = 0
        logger.info("Inference buffer reset.")

    @property
    def buffer_fill(self) -> int:
        """Current number of samples in the ring buffer."""
        return len(self._buffer)

    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            "active_mode": self.active_mode,
            "available_modes": list(self._interpreters.keys()),
            "total_inferences": self.total_inferences,
            "fall_count": self.fall_count,
            "buffer_fill": self.buffer_fill,
            "window_size": self.window_size,
            "step": self.step,
        }


# ---------------------------------------------------------------------------
# Real-Time Simulation Runner
# ---------------------------------------------------------------------------

def run_simulation(
    engine: FallDetectionEngine,
    csv_path: str,
    sensor_hz: int = 50,
    simulate_realtime: bool = True,
) -> None:
    """
    Simulate real-time sensor streaming on a CSV test file.

    Each row in the CSV should contain 6 columns (acc_x to gyro_z).
    Rows are pushed one at a time at the specified rate.

    Parameters
    ----------
    engine : FallDetectionEngine
    csv_path : str
        Path to test CSV file.
    sensor_hz : int
        Simulated sensor frequency in Hz. Set to 0 to run as fast as possible.
    simulate_realtime : bool
        If True, sleeps between samples to match sensor frequency.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    sleep_s = 1.0 / sensor_hz if simulate_realtime and sensor_hz > 0 else 0

    logger.info(f"Starting simulation — source: {csv_path} | {sensor_hz} Hz")
    processed = 0
    predictions_made = 0

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            logger.debug(f"CSV header: {header}")

        for i, row in enumerate(reader):
            try:
                sample = [float(v) for v in row[:6]]
            except (ValueError, IndexError) as exc:
                logger.warning(f"Row {i}: skipping invalid data — {exc}")
                continue

            result = engine.push_sample(sample)
            processed += 1

            if result:
                predictions_made += 1
                flag = " ALERT" if result["label"] != "Normal Activity" else " OK   "
                logger.info(
                    f"{flag} | Window #{predictions_made:>4} | "
                    f"{result['label']:<18} | Confidence: {result['confidence']:.1%} | "
                    f"Latency: {result['latency_ms']:.1f} ms"
                )

            if sleep_s:
                time.sleep(sleep_s)

    stats = engine.get_stats()
    logger.info(
        f"\n{'='*60}\n"
        f"Simulation complete\n"
        f"  Samples processed : {processed:,}\n"
        f"  Inferences made   : {stats['total_inferences']}\n"
        f"  Falls detected    : {stats['fall_count']}\n"
        f"{'='*60}"
    )


# ---------------------------------------------------------------------------
# Dummy CSV Generator
# ---------------------------------------------------------------------------

def generate_dummy_csv(output_path: str, n_rows: int = 512, seed: int = 42) -> None:
    """
    Generate a test CSV with synthetic IMU data for pipeline validation.
    First 256 rows are ~normal activity; last 256 rows simulate a fall (high variance).
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    half = n_rows // 2

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])

        # Normal (low variance)
        for _ in range(half):
            row = [
                rng.normal(0.1, 0.5),   # acc_x
                rng.normal(9.8, 0.2),   # acc_y (gravity)
                rng.normal(0.1, 0.5),   # acc_z
                rng.normal(0.0, 0.5),   # gyro_x
                rng.normal(0.0, 0.5),   # gyro_y
                rng.normal(0.0, 0.5),   # gyro_z
            ]
            writer.writerow([f"{v:.4f}" for v in row])

        # Fall (high variance, sudden spikes)
        for _ in range(half):
            row = [
                rng.normal(0.0, 8.0),
                rng.normal(0.0, 10.0),
                rng.normal(0.0, 8.0),
                rng.normal(0.0, 150.0),
                rng.normal(0.0, 150.0),
                rng.normal(0.0, 150.0),
            ]
            writer.writerow([f"{v:.4f}" for v in row])

    logger.info(f"Dummy CSV generated: {output_path} ({n_rows} rows)")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-time fall detection inference simulation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="TFLITE_PATH",
        help="Path to .tflite model (default from config).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Path to test CSV file.",
    )
    parser.add_argument(
        "--generate-dummy",
        action="store_true",
        help="Generate a dummy test CSV, run simulation, then exit.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Disable sleep delays (run as fast as possible).",
    )
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)

    tflite_paths = {
        "human": args.model or config.paths.model_tflite_human,
        "car": config.paths.model_tflite_car
    }
    dummy_csv = "data/test_dummy.csv"

    if args.generate_dummy or args.data is None:
        generate_dummy_csv(dummy_csv)
        csv_path = dummy_csv
    else:
        csv_path = args.data

    engine = FallDetectionEngine(
        tflite_paths=tflite_paths,
        window_size=config.windowing.window_size,
        step=int(config.windowing.window_size * (1 - config.windowing.overlap)),
        threshold=config.model.prediction_threshold,
        labels={int(k): v for k, v in config.model.labels.items()},
        static_variance_threshold=config.model.static_variance_threshold,
    )

    run_simulation(
        engine,
        csv_path=csv_path,
        sensor_hz=config.api.sensor_frequency_hz,
        simulate_realtime=not args.no_realtime,
    )


if __name__ == "__main__":
    main()
