"""
utils/logger.py — Structured logging with rotating file handler.

Provides a single ``get_logger`` factory so every module gets a
consistently-formatted logger that writes to both the console and
a rotating log file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured per ``config.py``.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── Rotating file handler ───────────────────
    log_dir = os.path.dirname(config.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ── standalone test ──────────────────────────

if __name__ == "__main__":
    print("utils.logger standalone test")
    log = get_logger("test_logger")
    log.info("Test info message")
    log.warning("Test warning message")
    assert log.name == "test_logger"
    print("Test passed ✓")

