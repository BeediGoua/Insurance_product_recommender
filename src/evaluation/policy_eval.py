"""
Evaluation of policy rule compliance.

This module measures how often the policy layer blocks products that
should not be recommended and whether the allowed set violates any
global constraints (e.g. recommending owned products).  It expects a
list of :class:`~decisionflow.schemas.PolicyDecision` objects.
"""

from __future__ import annotations

from typing import List, Dict, Any
from collections import Counter
from src.decisionflow.schemas import PolicyDecision


def evaluate_policy(decisions: List[PolicyDecision]) -> Dict[str, Any]:
    """Compute basic statistics about policy decisions.

    Metrics include the proportion of blocked products and a breakdown of
    reasons.  A high blocked rate may indicate overly strict rules or
    mismatch between model predictions and business constraints.
    """
    total_allowed = 0
    total_blocked = 0
    reason_counts: Counter[str] = Counter()
    for d in decisions:
        total_allowed += len(d.allowed)
        total_blocked += len(d.blocked)
        reason_counts.update(d.reasons.values())
    total = total_allowed + total_blocked
    result: Dict[str, Any] = {}
    result["allowed_rate"] = total_allowed / total if total > 0 else 0.0
    result["blocked_rate"] = total_blocked / total if total > 0 else 0.0
    # Normalise reason counts
    reason_freq = {reason: count / total_blocked if total_blocked > 0 else 0.0 for reason, count in reason_counts.items()}
    result["blocked_reasons"] = reason_freq
    return result
