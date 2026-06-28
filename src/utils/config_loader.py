"""
src/utils/config_loader.py
--------------------------
Loads config/config.yaml and returns a dot-accessible AttrDict so that
all modules can reference config values as config.model.epochs instead
of config['model']['epochs'].

Usage:
    from src.utils.config_loader import load_config
    config = load_config()
    print(config.windowing.window_size)  # 128
"""

import os
from pathlib import Path

import yaml


# Default config path relative to project root
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class AttrDict(dict):
    """
    A dict subclass that allows attribute-style access.
    Nested dicts are converted automatically.
    """

    def __getattr__(self, key: str):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        return AttrDict(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value):
        self[key] = value


def load_config(config_path: str | None = None) -> AttrDict:
    """
    Load the YAML config file and return an AttrDict.

    Parameters
    ----------
    config_path : str | None
        Explicit path to a YAML config file. If None, falls back to
        the default `config/config.yaml` relative to the project root.

    Returns
    -------
    AttrDict
        Nested dot-accessible configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Ensure config/config.yaml exists in the project root."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AttrDict(raw)
