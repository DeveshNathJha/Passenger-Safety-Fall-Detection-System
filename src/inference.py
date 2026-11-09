"""
src/inference.py
----------------
Real-time inference for edge deployment.

Design (v2):
  - TFLiteModel     one loaded interpreter + its metadata (threshold, labels,
                    sample rate). Shared by all clients; guarded by a lock
                    because a TFLite interpreter is not thread-safe.
  - ModelRegistry   {"human": TFLiteModel, "car": TFLiteModel}, loaded once.
  - StreamSession   per-connection state: resampler, ring buffer, k-of-n
                    confirmation, optional post-fall stillness check
                    (PostFallCheck). Two phones never share a buffer or a mode.

Standalone usage:
    # Simulate real-time inference on a CSV file (acc_x..gyro_z[, t_ms]):
    python src/inference.py --mode human --data data/test.csv

    # Generate a synthetic test CSV then run:
    python src/inference.py --generate-dummy --no-realtime
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.signal_processing import StreamResampler
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TFLite Interpreter Helper
# ---------------------------------------------------------------------------

def _load_interpreter(tflite_path: str):
    """
    Load a TFLite interpreter. Tries in order:
      1. ai_edge_litert  – LiteRT, the maintained lightweight runtime (Docker / edge)
      2. tflite_runtime  – legacy lightweight runtime (older Raspberry Pi images)
      3. tensorflow.lite – full TF, for development machines
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        logger.debug("Using ai_edge_litert interpreter.")
        return Interpreter(model_path=tflite_path)
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
        logger.debug("Using tflite_runtime interpreter.")
        return tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        pass
    try:
        import tensorflow as tf  # type: ignore
        logger.debug("Using tensorflow.lite interpreter.")
        return tf.lite.Interpreter(model_path=tflite_path)
    except ImportError:
        pass
    raise RuntimeError(
        "No TFLite runtime found. Install one of:\n"
        "  pip install ai-edge-litert      # lightweight (server / edge)\n"
        "  pip install tensorflow-cpu      # full TF (training)"
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TFLiteModel:
    """
    A loaded .tflite model plus the metadata written by train.py
    (decision threshold, labels, sample rate, evaluation metrics).
    """

    def __init__(self, name: str, tflite_path: str, metadata_path: Optional[str] = None,
                 labels: Optional[dict] = None):
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"TFLite model not found for '{name}': {tflite_path}")

        self.name = name
        self.path = str(tflite_path)
        self.metadata: dict = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.threshold = float(self.metadata.get("threshold", 0.5))
        self.labels = {int(k): v for k, v in (labels or self.metadata.get("labels")
                                              or {0: "Normal", 1: "EVENT"}).items()}

        self._interp = _load_interpreter(self.path)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self.window_size = int(self._in["shape"][1])
        self.n_channels = int(self._in["shape"][2])
        self._lock = threading.Lock()

    def score(self, window: np.ndarray) -> tuple[float, float]:
        """Return (sigmoid score, latency_ms) for one (window_size, 6) window."""
        if window.shape != (self.window_size, self.n_channels):
            raise ValueError(
                f"Expected window shape ({self.window_size}, {self.n_channels}), got {window.shape}."
            )
        x = window.reshape(1, self.window_size, self.n_channels).astype(self._in["dtype"])
        with self._lock:
            t0 = time.perf_counter()
            self._interp.set_tensor(self._in["index"], x)
            self._interp.invoke()
            score = float(self._interp.get_tensor(self._out["index"]).reshape(-1)[0])
            latency_ms = (time.perf_counter() - t0) * 1000
        return score, latency_ms

    def info(self) -> dict:
        return {
            "file": Path(self.path).name,
            "threshold": self.threshold,
            "labels": self.labels,
            "window_size": self.window_size,
            "size_kb": round(os.path.getsize(self.path) / 1024, 1),
            "architecture": self.metadata.get("architecture"),
            "trained_on": self.metadata.get("dataset"),
            "test_metrics": self.metadata.get("test_metrics"),
        }


class ModelRegistry:
    """All models the server can serve, keyed by mode name."""

    def __init__(self, models: dict[str, TFLiteModel]):
        self.models = models

    @classmethod
    def from_config(cls, config) -> "ModelRegistry":
        models = {}
        for mode, spec in config.models.items():
            models[mode] = TFLiteModel(
                mode,
                str(resolve_path(spec.tflite)),
                str(resolve_path(spec.metadata)),
                labels=spec.get("labels"),
            )
            logger.info(f"Loaded model '{mode}': {models[mode].path} (threshold={models[mode].threshold:.3f})")
        return cls(models)

    def get(self, mode: str) -> TFLiteModel:
        if mode not in self.models:
            raise ValueError(f"Unknown mode '{mode}'. Available: {list(self.models)}")
        return self.models[mode]

    def __contains__(self, mode: str) -> bool:
        return mode in self.models


# ---------------------------------------------------------------------------
# Post-fall stillness check
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostFallCheck:
    """
    Confirmation stage after a positive window.

    A real fall (or crash) ends with the person (or vehicle) lying still,
    while jumping, jogging or getting into a car keeps moving. A window above
    the threshold is therefore only a CANDIDATE; it becomes an event if, in
    the period [impact + settle_s, impact + settle_s + still_s], the signal
    is quiet: std(|acc|) <= still_acc_std and mean(|gyro|) <= still_gyro_dps.

    Parameters
    ----------
    settle_s : float
        Time after the impact that is skipped (bounces, rolling over).
    still_s : float
        Length of the period that must be still.
    still_acc_std : float
        Max std of the acceleration magnitude in that period (m/s²).
    still_gyro_dps : float
        Max mean rotation-rate magnitude in that period (°/s).
    """
    settle_s: float = 1.0
    still_s: float = 1.5
    still_acc_std: float = 1.5
    still_gyro_dps: float = 30.0

    def is_still(self, period: np.ndarray) -> tuple[bool, dict]:
        acc_std = float(np.std(np.linalg.norm(period[:, 0:3], axis=1)))
        gyro_mean = float(np.mean(np.linalg.norm(period[:, 3:6], axis=1)))
        still = acc_std <= self.still_acc_std and gyro_mean <= self.still_gyro_dps
        return still, {"acc_std": round(acc_std, 4), "gyro_mean": round(gyro_mean, 3)}


# ---------------------------------------------------------------------------
# Per-client streaming session
# ---------------------------------------------------------------------------

class StreamSession:
    """
    Stateful sliding-window inference for ONE client stream.

    Push raw samples (optionally timestamped); they are resampled to the
    model's rate, buffered, and every `step` new samples a window is scored.

    Parameters
    ----------
    model : TFLiteModel
    step : int
        New samples between consecutive predictions (window_size * (1 - overlap)).
    sample_rate_hz : float
        Rate the model was trained at; timestamped input is resampled to it.
    confirm_k, confirm_n : int
        Report an event only if k of the last n windows exceed the threshold.
    static_variance_threshold : float
        If std(|acc|) over the window is below this the phone is motionless
        and the window is forced to "normal" (guards against a resting phone
        whose constant gravity vector looks unusual to the model).
    post_fall_check : PostFallCheck | None
        If set, positive windows become candidates that are confirmed or
        rejected later by a stillness check (see PostFallCheck). If None, a
        positive window is an event immediately.

    Results returned by push():
        window prediction  {"kind": "window", "is_event", "candidate", "checking", ...}
        check resolution   {"kind": "confirmation", "is_event", "impact_sample",
                            "stillness", "delay_s", ...}
    """

    def __init__(self, model: TFLiteModel, step: int = 64, sample_rate_hz: float = 50.0,
                 confirm_k: int = 1, confirm_n: int = 1,
                 static_variance_threshold: float = 0.05,
                 post_fall_check: Optional[PostFallCheck] = None):
        self.model = model
        self.window_size = model.window_size
        self.step = step
        self.sample_rate_hz = sample_rate_hz
        self.confirm_k = confirm_k
        self.static_variance_threshold = static_variance_threshold
        self.post_fall_check = post_fall_check

        self._resampler = StreamResampler(sample_rate_hz)
        self._buffer: deque = deque(maxlen=self.window_size)
        self._since_last = 0
        self._recent_hits: deque = deque(maxlen=confirm_n)

        # post-fall check state (absolute sample indices)
        if post_fall_check is not None:
            self._settle_n = int(round(post_fall_check.settle_s * sample_rate_hz))
            self._still_n = max(1, int(round(post_fall_check.still_s * sample_rate_hz)))
            history_len = self.window_size + self._settle_n + self._still_n + step
        else:
            self._settle_n = self._still_n = 0
            history_len = 1
        self._history: deque = deque(maxlen=history_len)
        self._pending: Optional[dict] = None
        self._blocked_until = -1  # candidates with an impact before this index are ignored

        self.total_samples = 0
        self.total_inferences = 0
        self.event_count = 0
        self.candidate_count = 0

    @property
    def buffer_fill(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._history.clear()
        self._resampler.reset()
        self._recent_hits.clear()
        self._since_last = 0
        self._pending = None
        self._blocked_until = -1

    def push(self, sample, t_ms: Optional[float] = None) -> list[dict]:
        """
        Push one 6-axis sample [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].

        Returns a (usually empty) list of result dicts – resampling can emit
        0, 1 or several model-rate samples per input sample, and a pending
        post-fall check can resolve on any sample.
        """
        if len(sample) != 6:
            raise ValueError(f"Expected 6 IMU values, got {len(sample)}.")

        results = []
        for s in self._resampler.push(sample, t_ms):
            self._buffer.append(s)
            self._history.append(s)
            self.total_samples += 1
            self._since_last += 1
            if self._pending is not None:
                resolved = self._try_resolve()
                if resolved is not None:
                    results.append(resolved)
            if len(self._buffer) == self.window_size and self._since_last >= self.step:
                self._since_last = 0
                results.append(self.predict(np.asarray(self._buffer, dtype=np.float32)))
        return results

    def predict(self, window: np.ndarray) -> dict:
        """Score one full window and apply the guardrail, k-of-n and candidate logic."""
        raw_score, latency_ms = self.model.score(window)

        acc_mag = np.linalg.norm(window[:, 0:3], axis=1)
        acc_mag_std = float(np.std(acc_mag))
        guardrail_fired = acc_mag_std < self.static_variance_threshold
        score = 0.0 if guardrail_fired else raw_score

        hit = score >= self.model.threshold
        self._recent_hits.append(hit)
        positive = hit and sum(self._recent_hits) >= self.confirm_k

        self.total_inferences += 1
        is_event = candidate = False
        if positive and self.post_fall_check is None:
            is_event = True
            self.event_count += 1
            self._recent_hits.clear()  # one incident → one event
        elif positive and self._pending is None:
            # the window ends at absolute sample index total_samples - 1
            impact = self.total_samples - len(window) + int(np.argmax(acc_mag))
            if impact >= self._blocked_until:
                candidate = True
                self.candidate_count += 1
                self._recent_hits.clear()
                period_end = impact + self._settle_n + self._still_n
                self._pending = {"impact": impact, "score": raw_score, "period_end": period_end}
                self._blocked_until = period_end

        label_idx = int(is_event)
        return {
            "kind": "window",
            "label": self.model.labels[label_idx],
            "is_event": is_event,
            "candidate": candidate,
            "checking": self._pending is not None,
            "confidence": round(score if is_event else 1.0 - score, 4),
            "raw_score": round(raw_score, 4),
            "threshold": round(self.model.threshold, 4),
            "acc_mag_std": round(acc_mag_std, 4),
            "guardrail_fired": guardrail_fired,
            "latency_ms": round(latency_ms, 3),
            "total_inferences": self.total_inferences,
            "event_count": self.event_count,
        }

    def _try_resolve(self) -> Optional[dict]:
        """Confirm or reject the pending candidate once its still-period has arrived."""
        p = self._pending
        if self.total_samples < p["period_end"]:
            return None
        oldest = self.total_samples - len(self._history)  # absolute index of history[0]
        start = p["impact"] + self._settle_n
        hist = np.asarray(self._history, dtype=np.float32)
        period = hist[max(0, start - oldest):p["period_end"] - oldest]
        confirmed, stillness = self.post_fall_check.is_still(period)
        self._pending = None
        if confirmed:
            self.event_count += 1
        return {
            "kind": "confirmation",
            "label": self.model.labels[int(confirmed)],
            "is_event": confirmed,
            "confidence": round(p["score"], 4),
            "raw_score": round(p["score"], 4),
            "threshold": round(self.model.threshold, 4),
            "impact_sample": p["impact"],
            "stillness": stillness,
            "delay_s": round((self.total_samples - 1 - p["impact"]) / self.sample_rate_hz, 2),
            "total_inferences": self.total_inferences,
            "event_count": self.event_count,
        }

    def stats(self) -> dict:
        return {
            "mode": self.model.name,
            "buffer_fill": self.buffer_fill,
            "window_size": self.window_size,
            "step": self.step,
            "total_samples": self.total_samples,
            "total_inferences": self.total_inferences,
            "candidate_count": self.candidate_count,
            "event_count": self.event_count,
        }


def post_fall_check_from_config(config, mode: str) -> Optional[PostFallCheck]:
    """The configured PostFallCheck for this mode, or None if disabled."""
    cfg = config.inference.get("post_fall_check")
    if not cfg or not cfg.get("enabled", False) or mode not in cfg.get("modes", []):
        return None
    return PostFallCheck(settle_s=cfg.settle_s, still_s=cfg.still_s,
                         still_acc_std=cfg.still_acc_std, still_gyro_dps=cfg.still_gyro_dps)


def session_from_config(model: TFLiteModel, config) -> StreamSession:
    return StreamSession(
        model,
        step=int(config.windowing.window_size * (1 - config.windowing.overlap)),
        sample_rate_hz=config.signal.sample_rate_hz,
        confirm_k=config.inference.confirm_k,
        confirm_n=config.inference.confirm_n,
        static_variance_threshold=config.inference.static_variance_threshold,
        post_fall_check=post_fall_check_from_config(config, model.name),
    )


# ---------------------------------------------------------------------------
# Real-Time Simulation Runner
# ---------------------------------------------------------------------------

def run_simulation(session: StreamSession, csv_path: str, sensor_hz: int = 50,
                   simulate_realtime: bool = True) -> dict:
    """
    Replay a CSV (acc_x..gyro_z, optional 7th column t_ms) through a session.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    sleep_s = 1.0 / sensor_hz if simulate_realtime and sensor_hz > 0 else 0
    logger.info(f"Starting simulation — source: {csv_path} | model: {session.model.name}")
    processed = 0

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for i, row in enumerate(reader):
            try:
                sample = [float(v) for v in row[:6]]
                t_ms = float(row[6]) if len(row) > 6 and row[6] != "" else None
            except (ValueError, IndexError) as exc:
                logger.warning(f"Row {i}: skipping invalid data — {exc}")
                continue

            for result in session.push(sample, t_ms):
                if result["kind"] == "confirmation":
                    verdict = "CONFIRMED" if result["is_event"] else "REJECTED (still moving)"
                    logger.info(f"CHECK | candidate at sample {result['impact_sample']} {verdict} | "
                                f"stillness {result['stillness']}")
                    continue
                flag = "ALERT" if result["is_event"] else ("CAND " if result["candidate"] else "OK   ")
                logger.info(
                    f"{flag} | Window #{result['total_inferences']:>4} | {result['label']:<16} | "
                    f"score {result['raw_score']:.3f} | {result['latency_ms']:.2f} ms"
                )
            processed += 1
            if sleep_s:
                time.sleep(sleep_s)

    stats = session.stats()
    logger.info(
        f"Simulation complete — samples: {processed:,} | inferences: {stats['total_inferences']} | "
        f"events: {stats['event_count']}"
    )
    return stats


# ---------------------------------------------------------------------------
# Dummy CSV Generator
# ---------------------------------------------------------------------------

def generate_dummy_csv(output_path: str, n_rows: int = 512, seed: int = 42) -> None:
    """
    Write a synthetic 50 Hz test CSV: walking-like motion with one fall-like
    impact (a ~4 g spike followed by lying still) in the second half.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    t = np.arange(n_rows) / 50.0
    data = np.zeros((n_rows, 6))
    data[:, 0] = 1.5 * np.sin(2 * np.pi * 1.8 * t) + rng.normal(0, 0.3, n_rows)
    data[:, 1] = 9.81 + 2.0 * np.sin(2 * np.pi * 1.8 * t) + rng.normal(0, 0.3, n_rows)
    data[:, 2] = rng.normal(0, 0.5, n_rows)
    data[:, 3:6] = rng.normal(0, 20, (n_rows, 3))

    impact = int(n_rows * 0.7)
    data[impact - 10:impact, 3:6] += rng.normal(0, 200, (10, 3))  # tumbling
    data[impact:impact + 4, 0:3] = rng.normal(0, 25, (4, 3)) + [0, 35, 0]  # impact spike
    data[impact + 4:, 0:3] = [9.81, 0.0, 0.5] + rng.normal(0, 0.05, (n_rows - impact - 4, 3))  # lying
    data[impact + 4:, 3:6] = rng.normal(0, 0.5, (n_rows - impact - 4, 3))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        writer.writerows([[f"{v:.4f}" for v in row] for row in data])
    logger.info(f"Dummy CSV generated: {output_path} ({n_rows} rows)")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Replay a sensor CSV through the inference engine.")
    parser.add_argument("--mode", default="human", help="Model to use: human | car.")
    parser.add_argument("--data", type=str, default=None, metavar="CSV_PATH")
    parser.add_argument("--generate-dummy", action="store_true",
                        help="Generate a synthetic test CSV and replay it.")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Disable sleep delays (run as fast as possible).")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)

    registry = ModelRegistry.from_config(config)
    session = session_from_config(registry.get(args.mode), config)

    csv_path = args.data
    if args.generate_dummy or csv_path is None:
        csv_path = "data/test_dummy.csv"
        generate_dummy_csv(csv_path)

    run_simulation(session, csv_path, sensor_hz=config.signal.sample_rate_hz,
                   simulate_realtime=not args.no_realtime)


if __name__ == "__main__":
    main()
