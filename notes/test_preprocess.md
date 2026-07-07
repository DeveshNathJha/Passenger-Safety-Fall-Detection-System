# Notes: `tests/test_preprocess.py`

## Purpose
Unit tests for all preprocessing logic. Tests run on synthetic data — **no internet or dataset required.**

## Test Classes

### `TestCreateWindows` (8 tests)
Tests the core `create_windows()` sliding-window function.

| Test | What it checks |
|---|---|
| `test_output_shape` | X.shape is (N, 128, 6) |
| `test_label_correctness` | All y == provided label |
| `test_window_count_50pct_overlap` | `(300-128)//64 + 1 = 3` windows |
| `test_no_overlap` | step=128 → 2 windows |
| `test_input_shorter_than_window` | 64 rows → 0 windows (no crash) |
| `test_exact_window_fit` | 128 rows → exactly 1 window |
| `test_dtype_float32` | TFLite requires float32 |
| `test_window_content_integrity` | Window[0] == data[0:128] |

### `TestParseTxtFile` (3 tests)
| Test | What it checks |
|---|---|
| `test_valid_file` | Correctly parses 50 data rows from mixed file |
| `test_empty_file_returns_none` | Returns None instead of crashing |
| `test_missing_file` | Returns None for nonexistent path |

### `TestGenerateDummyData` (2 tests)
| Test | What it checks |
|---|---|
| `test_shape` | Returns (200, 128, 6) and (200,) |
| `test_binary_labels` | Only 0 and 1 in y |

## Run
```bash
python3 -m pytest tests/test_preprocess.py -v
```
All 13 tests pass in < 1 second.
