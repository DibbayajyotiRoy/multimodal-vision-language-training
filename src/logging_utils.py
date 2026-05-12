"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure root logger for console and optional file output."""
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with the configured handlers."""
    return logging.getLogger(name)
