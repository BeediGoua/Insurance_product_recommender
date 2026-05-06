"""Generate deterministic explanations for recommended insurance products."""

from __future__ import annotations

from typing import Dict, List, Any

from .schemas import ClientProfile, RecommendationResult, PolicyDecision, ExplanationResult


def build_recommendation_explanation(
    profile: ClientProfile,
    rec: RecommendationResult,
    policy: PolicyDecision,
    risk: dict,
) -> list:
    """Generate a list of explanation objects for each recommended product."""
    try:
        from src.retrieval.product_catalog import get_product_info as _get_info
    except ImportError:
        _get_info = None  # type: ignore

    explanations = []
    for product in policy.allowed:
        score = rec.filtered_scores.get(product)
        if score is None:
            continue
        reasons = []
        # 1. Model confidence
        reasons.append(f"high predicted relevance (score={score:.2f})")
        # 2. Not already owned
        if product not in profile.current_products:
            reasons.append("client does not already own this product")
        # 3. Product catalog context
        if _get_info is not None:
            product_info = _get_info(product)
            if product_info:
                description = product_info.get("description")
                target_needs = product_info.get("target_needs")
                if description:
                    reasons.append(f"product context: {description}")
                if target_needs:
                    reasons.append(f"target need: {target_needs}")
        # 4. Segment information
        if profile.segment:
            reasons.append(f"client belongs to segment '{profile.segment}'")
        # 5. Data quality
        if profile.data_quality and profile.data_quality != "complete":
            reasons.append(f"data quality is {profile.data_quality}")
        # 6. Risk level
        if risk.get("risk_level"):
            reasons.append(f"overall recommendation risk is {risk['risk_level']}")
        # 7. Policy notes
        reason_block = policy.reasons.get(product)
        if reason_block:
            reasons.append(f"note: rule triggered – {reason_block}")
        explanations.append(ExplanationResult(
            product=product,
            reasons=reasons,
            limitations=["explanations are generated from available structured data only"],
        ))
    return explanations
