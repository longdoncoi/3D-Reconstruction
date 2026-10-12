"""Logging configuration for the ``cvtrain`` package.

One module, one job: put the package's records on stdout in the human-readable
format the operator reads. The machine-readable ``[PROGRESS]`` protocol the Qt
dock parses lives next door in :mod:`cvtrain.progress`, because it is a *wire
contract* with another program rather than a logging concern.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["DATE_FORMAT", "LOG_FORMAT", "setup_logging"]

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure the package logger once and return it."""

    logger = logging.getLogger("cvtrain")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
