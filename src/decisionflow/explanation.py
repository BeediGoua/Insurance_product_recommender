"""
Explain recommendations in human language.

The goal of this module is to provide meaningful, transparent
explanations for each recommended product.  It does not rely on any
black box reasoning from language models; instead it uses only the
information available from the deterministic recommendation and
policy modules.  This keeps the system auditable and avoids
hallucinating reasons that are unsupported by the data.
"""

from __future__ import annotations

from typing import Dict, List, Any

from .schemas import ClientProfile, RecommendationResult, PolicyDecision, ExplanationResult


def build_recommendation_explanation(
    profile: ClientProfile,
    rec: RecommendationResult,
    policy: PolicyDecision,
    risk: Dict[str, Any],
) -> List[ExplanationResult]:
    """Generate a list of explanation objects for each recommended product.

    The explanations are constructed using only the inputs and outputs
    of the recommendation process.  For each product in
    ``policy.allowed``, the corresponding score and profile data are used
    to generate a set of reasons.  Products that were blocked are
    ignored.

    Parameters
    ----------
    profile: ClientProfile
        The client profile used during recommendation.
    rec: RecommendationResult
        The raw and filtered scores.
    policy: PolicyDecision
        Result of the policy rules application.
    risk: Dict[str, Any]
        Risk metrics computed for the recommendation.

    Returns
    -------
    List[ExplanationResult]
        A list of explanation dataclasses, one per allowed product.
    """
    explanations: List[ExplanationResult] = []
    for product in policy.allowed:
        score = rec.filtered_scores.get(product)
        if score is None:
            continue
        reasons: List[str] = []
        # 1. Model confidence
        reasons.append(f"high predicted relevance (score={score:.2f})")
        # 2. Not already owned
        if product not in profile.current_products:
            reasons.append("client does not already own this product")
        # 3. Segment information
        if profile.segment:
            reasons.append(f"client belongs to segment '{profile.segment}'")
        # 4. Data quality
        if profile.data_quality and profile.data_quality != "complete":
            reasons.append(f"data quality is {profile.data_quality}")
        # 5. Risk level
        if risk.get("risk_level"):
            reasons.append(f"overall recommendation risk is {risk['risk_level']}")
        # 6. Policy notes
        reason_block = policy.reasons.get(product)
        if reason_block:
            reasons.append(f"note: rule triggered – {reason_block}")
        explanations.append(ExplanationResult(
            product=product,
            reasons=reasons,
            limitations=["explanations are generated from available structured data only"],
        ))
    return explanations
