"""
Logging helpers for DecisionFlow.

We use the standard Python logging module but define a convenience
function to obtain a logger with a consistent format.  By default the
log level is set to INFO.  In production you might configure
different handlers for file output, console output, or cloud logging.
"""

import logging


def get_logger(name: str = "decisionflow") -> logging.Logger:
    """Return a configured logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
