"""
Evaluation of policy rule compliance.

This module measures how often the policy layer blocks products that
should not be recommended and whether the allowed set violates any
global constraints (e.g. recommending owned products).  It expects a
list of :class:`~decisionflow.schemas.PolicyDecision` objects.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set
from collections import Counter
from src.decisionflow.schemas import PolicyDecision


def evaluate_policy(
    decisions: List[PolicyDecision],
    final_recommendations: Optional[List[List[str]]] = None,
    owned_products_by_case: Optional[List[Set[str]]] = None,
    forbidden_by_case: Optional[List[Set[str]]] = None,
    acceptable_by_case: Optional[List[Set[str]]] = None,
) -> Dict[str, Any]:
    """Compute basic statistics about policy decisions.

    Metrics include the proportion of blocked products and a breakdown of
    reasons.  A high blocked rate may indicate overly strict rules or
    mismatch between model predictions and business constraints.
    """
    total_allowed = 0
    total_blocked = 0
    reason_counts: Counter[str] = Counter()
    blocked_per_case: List[int] = []
    for d in decisions:
        total_allowed += len(d.allowed)
        total_blocked += len(d.blocked)
        blocked_per_case.append(len(d.blocked))
        reason_counts.update(d.reasons.values())
    total = total_allowed + total_blocked
    result: Dict[str, Any] = {}
    result["allowed_rate"] = total_allowed / total if total > 0 else 0.0
    result["blocked_rate"] = total_blocked / total if total > 0 else 0.0
    result["avg_blocked_per_case"] = sum(blocked_per_case) / len(blocked_per_case) if blocked_per_case else 0.0
    # Normalise reason counts
    reason_freq = {reason: count / total_blocked if total_blocked > 0 else 0.0 for reason, count in reason_counts.items()}
    result["blocked_reasons"] = reason_freq

    # Ground-truth-aware checks (when context is provided)
    if final_recommendations is not None:
        n = len(final_recommendations)
        forbidden_hits = 0
        owned_hits = 0
        acceptable_hits = 0
        acceptable_cases = 0

        for i in range(n):
            recs = set(final_recommendations[i])

            if forbidden_by_case is not None and i < len(forbidden_by_case):
                forbidden_set = forbidden_by_case[i]
                if recs & forbidden_set:
                    forbidden_hits += 1

            if owned_products_by_case is not None and i < len(owned_products_by_case):
                owned_set = owned_products_by_case[i]
                if recs & owned_set:
                    owned_hits += 1

            if acceptable_by_case is not None and i < len(acceptable_by_case):
                acceptable_set = acceptable_by_case[i]
                if acceptable_set:
                    acceptable_cases += 1
                    if recs & acceptable_set:
                        acceptable_hits += 1

        result["forbidden_violation_rate"] = forbidden_hits / n if n > 0 else 0.0
        result["owned_violation_rate"] = owned_hits / n if n > 0 else 0.0
        result["acceptable_hit_rate"] = acceptable_hits / acceptable_cases if acceptable_cases > 0 else 0.0

    return result
