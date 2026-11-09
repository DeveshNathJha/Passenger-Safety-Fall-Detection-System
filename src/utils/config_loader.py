"""
src/utils/config_loader.py
--------------------------
Loads config/config.yaml and returns a dot-accessible AttrDict so that
all modules can reference config values as config.model.epochs instead
of config['model']['epochs'].

Nested dicts are converted once, at load time, so that writes such as
``config.model.epochs = 5`` persist (the previous implementation built a
fresh copy on every attribute access, silently discarding CLI overrides).

Usage:
    from src.utils.config_loader import load_config
    config = load_config()
    print(config.windowing.window_size)  # 128
"""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default config path relative to project root (override with FDS_CONFIG)
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class AttrDict(dict):
    """
    A dict subclass that allows attribute-style access.
    Nested dicts (including dicts inside lists) are converted recursively.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in self.items():
            super().__setitem__(key, _convert(val))

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'") from None

    def __setattr__(self, key: str, value):
        self[key] = _convert(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, _convert(value))


def _convert(val):
    if isinstance(val, dict) and not isinstance(val, AttrDict):
        return AttrDict(val)
    if isinstance(val, list):
        return [_convert(v) for v in val]
    return val


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(config_path: str | None = None) -> AttrDict:
    """
    Load the YAML config file and return an AttrDict.

    Parameters
    ----------
    config_path : str | None
        Explicit path to a YAML config file. If None, uses $FDS_CONFIG,
        then falls back to `config/config.yaml` relative to the project root.

    Returns
    -------
    AttrDict
        Nested dot-accessible configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    """
    path = Path(config_path or os.environ.get("FDS_CONFIG") or _DEFAULT_CONFIG_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Ensure config/config.yaml exists in the project root."
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AttrDict(raw)
