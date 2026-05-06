"""
Metrics recording for DecisionFlow.

This module defines a trivial in‑memory metrics collector.  In
production you would likely replace this with integration to a
telemetry system such as Prometheus, StatsD or OpenTelemetry.
"""

from __future__ import annotations

from typing import Dict, Any

_METRICS: Dict[str, float] = {}


def record_metric(name: str, value: float) -> None:
    """Record a metric value in the internal dictionary.

    If the metric is recorded multiple times the last value wins.  In a
    real system you might accumulate counts or compute averages.
    """
    _METRICS[name] = value


def get_metric(name: str) -> Any:
    """Retrieve a metric value by name."""
    return _METRICS.get(name)
