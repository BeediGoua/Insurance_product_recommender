"""
Monitoring utilities for the DecisionFlow system.

This package contains simple wrappers around Python's ``logging`` and
metrics collection.  It is not wired into any external monitoring
stack, but the functions defined here can be easily extended to emit
logs and metrics to your favourite observability platform.
"""

from .logs import get_logger  # noqa: F401
from .metrics import record_metric  # noqa: F401
