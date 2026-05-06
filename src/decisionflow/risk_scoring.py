"""
Risk scoring and consistency checks for recommendations.

This module implements simple heuristics to assess how confident the
system is in its recommendations.  It is not intended to be used for
regulatory risk (actuarial or underwriting) but rather to give users a
sense of how much trust to place in the recommendation.
"""

from __future__ import annotations

from typing import Dict, List, Any

from .schemas import RecommendationResult


def compute_recommendation_risk(rec: RecommendationResult) -> Dict[str, Any]:
    """Compute a set of risk and confidence indicators for a recommendation.

    The following simple heuristics are used:

    * **confidence**: the difference between the highest and second highest
      score divided by the highest score.  Values near 1 indicate
      high confidence, whereas values near 0 indicate ambiguity.
    * **risk_level**: a qualitative label derived from the confidence.
      ``"low"`` for confidence >= 0.5, ``"medium"`` for confidence >= 0.2
      and ``"high"`` otherwise.
    * **manual_review_required**: ``True`` when no scores are available or
      the top score is not significantly better than the rest.

    Parameters
    ----------
    rec: RecommendationResult
        The recommendation result produced by the ML engine and filtered
        by business rules.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the risk metrics.
    """
    scores = rec.filtered_scores
    # Remove -inf and sort descending
    valid_scores = [s for s in scores.values() if s != float("-inf")]
    if not valid_scores:
        return {
            "confidence": 0.0,
            "risk_level": "high",
            "manual_review_required": True,
            "risk_reasons": ["no valid recommendations"],
        }
    sorted_scores = sorted(valid_scores, reverse=True)
    top = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    if top == 0.0:
        confidence = 0.0
    else:
        confidence = (top - second) / top
    # Determine qualitative risk level
    if confidence >= 0.5:
        risk_level = "low"
    elif confidence >= 0.2:
        risk_level = "medium"
    else:
        risk_level = "high"
    manual_review_required = len(valid_scores) == 0 or confidence < 0.1
    risk_reasons: List[str] = []
    if manual_review_required:
        risk_reasons.append("low confidence in top recommendation")
    return {
        "confidence": round(confidence, 3),
        "risk_level": risk_level,
        "manual_review_required": manual_review_required,
        "risk_reasons": risk_reasons,
    }
