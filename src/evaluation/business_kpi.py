"""Business KPI aggregation for evaluation outputs."""

from __future__ import annotations

from typing import Dict, Any, Optional


def evaluate_business_kpis(
    recommender_metrics: Dict[str, Any],
    policy_metrics: Dict[str, Any],
    explanation_metrics: Dict[str, Any],
    avg_latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute simple business-facing KPIs from evaluation metrics.

    The KPIs intentionally stay simple and directly map to business goals:
    - sell more (cross-sell proxy)
    - reduce invalid recommendations
    - keep explanations useful
    - monitor agent/user efficiency (latency proxy)
    """
    hit3 = float(recommender_metrics.get("Hit@3", 0.0))
    forbidden_v = float(policy_metrics.get("forbidden_violation_rate", 0.0))
    owned_v = float(policy_metrics.get("owned_violation_rate", 0.0))
    invalid_rate = forbidden_v + owned_v
    explanation_quality = float(explanation_metrics.get("explanation_quality_score", 0.0))

    readiness_score = max(
        0.0,
        min(
            1.0,
            0.5 * hit3 + 0.3 * (1.0 - invalid_rate) + 0.2 * explanation_quality,
        ),
    )

    return {
        "kpi_cross_sell_proxy": hit3,
        "kpi_invalid_recommendation_rate": invalid_rate,
        "kpi_explanation_quality": explanation_quality,
        "kpi_efficiency_latency_ms": avg_latency_ms,
        "kpi_readiness_score": readiness_score,
    }
