"""
src/utils/logger.py
-------------------
Centralized logging factory for the Fall Detection System.
Returns a named logger with:
  - StreamHandler  → console (stdout)
  - RotatingFileHandler → logs/app.log (5 MB × 3 backups)

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded successfully.")
"""

import logging
import os
from logging.handlers import RotatingFileHandler


_LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Create and return a named logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Logger name (use __name__ for module-level loggers).
    log_file : str
        Path to the rotating log file. Parent directory is created automatically.
    level : int
        Logging level (e.g. logging.DEBUG, logging.INFO).
    max_bytes : int
        Maximum size of a single log file before rotation.
    backup_count : int
        Number of backup log files to retain.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console Handler ---
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- Rotating File Handler ---
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
