# Notes: `src/utils/config_loader.py`

## Purpose
Loads `config/config.yaml` and returns an **`AttrDict`** — a dict subclass that supports dot-notation access. This means you write `config.model.epochs` instead of `config['model']['epochs']`.

## Class: `AttrDict`
```python
class AttrDict(dict):
    def __getattr__(self, key): ...
    def __setattr__(self, key, value): ...
```
- Recursively wraps nested dicts: `config.windowing.window_size` works at any depth.
- Inherits all standard `dict` methods — can still do `config['model']['epochs']` if preferred.

## Function: `load_config(config_path=None) → AttrDict`

### Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str \| None` | `None` | Custom YAML path. If None, resolves to `config/config.yaml` relative to project root. |

### What it does
1. Resolves path relative to `this_file.parents[2]` (two levels up from `src/utils/`) = project root.
2. Opens and `yaml.safe_load()`s the file.
3. Wraps the raw dict in `AttrDict` and returns it.

### Raises
- `FileNotFoundError` if the config file doesn't exist.

## Usage Pattern
```python
# In any module:
from src.utils.config_loader import load_config

config = load_config()
print(config.windowing.window_size)   # → 128
print(config.model.epochs)            # → 10
print(config.paths.model_tflite)      # → "models/mobifall_edge_model.tflite"

# For testing with a custom config:
config = load_config("tests/test_config.yaml")
```

## Key Design Decision
Using `AttrDict` (not `SimpleNamespace` or dataclasses) means you can still do dict operations like `config.model.items()` or `for k, v in config.data.items()` — useful when iterating over label mappings.

## Common Mistake
The `labels` field in config is a dict with **integer keys** stored as YAML strings. When reading:
```python
labels = {int(k): v for k, v in config.model.labels.items()}
# → {0: "Normal Activity", 1: "FALL DETECTED"}
```
Always cast keys with `int()` before using as a lookup dict.
