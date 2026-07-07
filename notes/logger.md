# Notes: `src/utils/logger.py`

## Purpose
Centralized logging factory for the entire project. Every module calls `get_logger(__name__)` to get a named logger with both console and file output. **Never use `print()` in production code** — use the logger.

## Function: `get_logger(name, ...) → logging.Logger`

### Parameters
| Param | Default | Meaning |
|---|---|---|
| `name` | required | Logger name (use `__name__` for the module name) |
| `log_file` | `"logs/app.log"` | Output file path (directory auto-created) |
| `level` | `logging.INFO` | Severity threshold |
| `max_bytes` | `5 MB` | Max size before log file rotates |
| `backup_count` | `3` | Number of rotated files to keep |

### What it does
1. Checks if a logger named `name` already has handlers (prevents duplicate handlers on multiple `get_logger()` calls).
2. Creates a **`StreamHandler`** → prints to terminal.
3. Creates a **`RotatingFileHandler`** → writes to `logs/app.log`, rotates when file exceeds 5 MB, keeps 3 backups (`app.log.1`, `app.log.2`, `app.log.3`).
4. Both handlers share the same `Formatter`:  
   `[2026-03-18 21:00:00] [INFO    ] [src.preprocess] Message here`

## Log Format Breakdown
```
[2026-03-18 20:57:40] [INFO    ] [src.app] Loading TFLite model
│                      │          │         └── The message
│                      │          └── Logger name (= module __name__)
│                      └── Log level (left-padded to 8 chars)
└── Timestamp (YYYY-MM-DD HH:MM:SS)
```

## Usage in Every Module
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)   # ← always at top of file

logger.debug("Detailed internal value: %s", value)
logger.info("Window created: shape=%s", X.shape)
logger.warning("File skipped: %s", filepath)
logger.error("Model file not found: %s", path)
logger.exception("Unexpected crash:")  # auto-adds traceback
```

## Log Severity Guide
| Level | When to Use |
|---|---|
| `DEBUG` | Internal variable values during development |
| `INFO` | Normal progress milestones (model loaded, epoch complete) |
| `WARNING` | Recoverable issues (file skipped, unexpected input) |
| `ERROR` | Operation failed but server continues |
| `EXCEPTION` | Unhandled exceptions (auto-captures traceback) |

## Log Files
```
logs/
├── app.log        ← current log
├── app.log.1      ← previous (after rotation)
├── app.log.2
└── app.log.3
```
Files are gitignored. Check them when debugging issues.
