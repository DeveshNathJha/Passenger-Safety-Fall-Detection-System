# Notes: `tests/test_inference.py`

## Purpose
Smoke tests for `FallDetectionEngine`. Uses the real `.tflite` model file. Tests auto-skip if neither `tensorflow` nor `tflite_runtime` is installed.

## Skip Guard
```python
_TFLITE_AVAILABLE = True
try:
    import tflite_runtime.interpreter
except ImportError:
    import tensorflow
    from src.inference import FallDetectionEngine
except:
    _TFLITE_AVAILABLE = False

@tflite_required  # ← these tests skip if TFLite unavailable
class TestEngineInit: ...
```

## Test Classes

### `TestEngineInit` (3 tests) — requires TFLite
| Test | Checks |
|---|---|
| `test_engine_loads` | Engine initializes without errors |
| `test_invalid_model_path` | `FileNotFoundError` raised for bad path |
| `test_initial_buffer_empty` | `buffer_fill == 0` after reset |

### `TestPushSample` (4 tests) — requires TFLite
| Test | Checks |
|---|---|
| `test_returns_none_before_window_fill` | No prediction for first 127 samples |
| `test_returns_prediction_at_window_boundary` | 128th sample triggers prediction |
| `test_invalid_sample_length` | `ValueError` for wrong-length input |
| `test_buffer_increments` | `buffer_fill` increases by 1 per push |

### `TestPredictionOutput` (4 tests) — requires TFLite
| Test | Checks |
|---|---|
| `test_has_required_keys` | `{label, confidence, raw_score, latency_ms}` all present |
| `test_label_is_valid_string` | Label is one of `config.model.labels.values()` |
| `test_confidence_in_range` | `0.0 ≤ confidence ≤ 1.0` |
| `test_latency_is_positive` | `latency_ms > 0` |

### `TestDirectPredict` (2 tests) — requires TFLite
| Test | Checks |
|---|---|
| `test_shape_validation` | Wrong window shape → `ValueError` |
| `test_predict_on_zeros` | All-zeros input doesn't crash |

### `TestDummyCsvGeneration` (2 tests) — **always runs**
| Test | Checks |
|---|---|
| `test_creates_file` | File is created at specified path |
| `test_row_count` | File has header + N data rows |

## Run
```bash
python3 -m pytest tests/test_inference.py -v
```
With tensorflow installed: 16 passed, 0 skipped.
Without tensorflow: 13 skipped, 2 passed (CSV tests always run).
