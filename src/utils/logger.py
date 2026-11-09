"""
src/utils/logger.py
-------------------
Centralized logging for the Fall Detection System.

All module loggers are children of a single ``fds`` logger which owns
exactly one console handler and one RotatingFileHandler. (Attaching a
separate RotatingFileHandler per module to the same file breaks log
rotation on Windows with PermissionError.)

Environment overrides:
    FDS_LOG_LEVEL  – DEBUG | INFO | WARNING | ERROR   (default INFO)
    FDS_LOG_FILE   – path to the log file, or "" to disable file logging

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded successfully.")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME = "fds"


def _configure_root() -> logging.Logger:
    root = logging.getLogger(_ROOT_NAME)
    if root.handlers:
        return root

    root.setLevel(os.environ.get("FDS_LOG_LEVEL", "INFO").upper())
    root.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console Handler (never crash on non-ASCII text in a cp1252 Windows console) ---
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # --- Rotating File Handler (one per process) ---
    log_file = os.environ.get("FDS_LOG_FILE", "logs/app.log")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that writes through the shared ``fds`` handlers.

    Parameters
    ----------
    name : str
        Logger name (use __name__ for module-level loggers).
    """
    _configure_root()
    short = name.removeprefix("src.") if name != "__main__" else "main"
    return logging.getLogger(f"{_ROOT_NAME}.{short}")
