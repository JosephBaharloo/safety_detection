from __future__ import annotations

import logging

_FORMATTER: logging.Formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler: logging.StreamHandler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
