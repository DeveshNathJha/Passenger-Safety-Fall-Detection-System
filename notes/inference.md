# Notes: `src/inference.py`

## Purpose
The **heart of the edge deployment pipeline.** `FallDetectionEngine` is a stateful, thread-safe inference wrapper that accepts sensor samples one at a time (as a stream) and returns predictions only when a full window is accumulated.

---

## Key Concept: The Ring Buffer State Machine

```
push_sample() called 64 times → no prediction
push_sample() called 64 more  → PREDICTION RETURNED (window = 128 samples)
push_sample() called 63 times → no prediction (sliding window, need 1 more)
push_sample() called 1 more   → PREDICTION RETURNED (50% overlap → step=64)
```

The `deque(maxlen=128)` acts as a **circular ring buffer**:
- Old samples are automatically dropped when new ones arrive.
- `_samples_since_last_predict` tracks how many new samples have been pushed since the last inference.
- Inference happens when `buffer_fill == 128 AND samples_since_last_predict >= 64`.

This is critical: it means inference runs at `window_size / step = 2 Hz` (every 0.5 seconds at 50 Hz input).

---

## Class: `FallDetectionEngine`

### Constructor Parameters
| Param | Default | Meaning |
|---|---|---|
| `tflite_path` | required | Path to `.tflite` model file |
| `window_size` | `128` | Samples per inference window |
| `step` | `64` | New samples needed between inferences |
| `threshold` | `0.5` | Sigmoid score above which → FALL |
| `labels` | `{0: "Normal", 1: "FALL"}` | Class name map |

### `push_sample(sample: list[float]) → dict | None`
**Main API. Call this from WebSocket handler or sensor loop.**
- Input: `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`
- Returns `dict` with prediction when window full, else `None`
- Thread-safe via `threading.Lock()`

### `predict(window: np.ndarray) → dict`
Runs one TFLite inference. Called internally by `push_sample`.
- Input: `np.ndarray` shape `(128, 6)`, dtype `float32`
- Returns:
```python
{
    "label": "Normal Activity",  # or "FALL DETECTED"
    "confidence": 0.9312,        # 0.0–1.0 (how sure the model is)
    "raw_score": 0.0781,         # raw sigmoid output
    "latency_ms": 2.1,           # TFLite inference time
    "total_inferences": 42,
    "fall_count": 3
}
```

### `reset_buffer()`
Clears the ring buffer. Call at journey start or on passenger change.

### `get_stats() → dict`
Returns session statistics. Used by `/health` endpoint.

---

## TFLite Runtime auto-detection
```python
try:
    import tflite_runtime.interpreter  # → edge (Pi, Android)
except ImportError:
    import tensorflow.lite             # → development (laptop)
```
Falls back gracefully with a clear install instruction if neither is found.

---

## `run_simulation(engine, csv_path, sensor_hz=50)`
Reads a CSV row-by-row, sends to `engine.push_sample()`, simulates real-time streaming. Useful for validating the model on test data without a live sensor.

```bash
python src/inference.py --generate-dummy --no-realtime
```

---

## Thread Safety Note
`push_sample()` uses a `threading.Lock()` — safe for concurrent WebSocket connections. However, each WebSocket connection should have its **own engine instance** for independent per-session state. (Currently the server shares one global engine for simplicity.)
