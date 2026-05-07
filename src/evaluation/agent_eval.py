"""Evaluation of agent behavior with practical operational metrics."""

from __future__ import annotations

from typing import Dict, Any, Iterable


def _to_set(value: Any) -> set[str]:
    """Convert a list-like value to a set of strings."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    try:
        return {str(x) for x in value}
    except TypeError:
        return {str(value)}


def evaluate_agent(conversations: list) -> Dict[str, Any]:
    """Evaluate agent runs from conversation/trace logs.

    Parameters
    ----------
    conversations: list
        List of conversation transcripts or interaction logs.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary with:
        - correct_tool_call_rate
        - missing_tool_rate
        - unnecessary_tool_rate
        - avg_latency_ms
        - total_cost_usd
        - error_rate
        - coverage_rate
    """
    if not conversations:
        return {
            "n_runs": 0,
            "coverage_rate": 0.0,
            "correct_tool_call_rate": 0.0,
            "missing_tool_rate": 0.0,
            "unnecessary_tool_rate": 0.0,
            "avg_latency_ms": None,
            "total_cost_usd": 0.0,
            "error_rate": 0.0,
        }

    n_runs = 0
    covered_runs = 0
    correct_sum = 0.0
    missing_sum = 0.0
    unnecessary_sum = 0.0
    error_count = 0
    latency_values: list[float] = []
    total_cost = 0.0

    for conv in conversations:
        n_runs += 1

        actual_tools = _to_set(conv.get("tool_calls") or conv.get("tools_called"))
        expected_tools = _to_set(conv.get("expected_tools") or conv.get("required_tools"))

        if expected_tools:
            covered_runs += 1
            inter = actual_tools & expected_tools
            correct_sum += len(inter) / len(expected_tools)
            missing_sum += len(expected_tools - actual_tools) / len(expected_tools)
            unnecessary_sum += len(actual_tools - expected_tools) / len(expected_tools)

        latency = conv.get("latency_ms")
        if latency is not None:
            try:
                latency_values.append(float(latency))
            except (TypeError, ValueError):
                pass

        cost = conv.get("cost_usd", 0.0)
        try:
            total_cost += float(cost)
        except (TypeError, ValueError):
            pass

        has_error = conv.get("error") or conv.get("has_error")
        if bool(has_error):
            error_count += 1

    denom_cov = covered_runs if covered_runs > 0 else 1
    avg_latency = (sum(latency_values) / len(latency_values)) if latency_values else None

    return {
        "n_runs": n_runs,
        "coverage_rate": covered_runs / n_runs if n_runs > 0 else 0.0,
        "correct_tool_call_rate": correct_sum / denom_cov if covered_runs > 0 else 0.0,
        "missing_tool_rate": missing_sum / denom_cov if covered_runs > 0 else 0.0,
        "unnecessary_tool_rate": unnecessary_sum / denom_cov if covered_runs > 0 else 0.0,
        "avg_latency_ms": avg_latency,
        "total_cost_usd": total_cost,
        "error_rate": error_count / n_runs if n_runs > 0 else 0.0,
    }
