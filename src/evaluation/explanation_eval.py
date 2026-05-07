"""
Evaluation of explanation quality.

Explanations are difficult to evaluate automatically.  This module
provides a minimal set of heuristics, such as the average number of
reasons per recommended product and the proportion of explanations
containing caveats.
"""

from __future__ import annotations

from typing import List, Dict, Any
from src.decisionflow.schemas import ExplanationResult


def evaluate_explanations(explanations: List[List[ExplanationResult]]) -> Dict[str, Any]:
    """Aggregate metrics over a list of explanation lists.

    Parameters
    ----------
    explanations: List[List[ExplanationResult]]
        A list for each client, containing the explanation objects for
        the recommended products.

    Returns
    -------
    Dict[str, Any]
        Dictionary with average number of reasons and caveats per
        explanation.
    """
    total_expl = 0
    total_reasons = 0
    total_limitations = 0
    factual_hits = 0
    simple_hits = 0
    useful_hits = 0

    factual_tokens = [
        "score",
        "client",
        "segment",
        "target need",
        "product context",
        "rule",
        "risk",
        "already own",
    ]
    useful_tokens = [
        "target need",
        "does not already own",
        "segment",
        "rule",
        "risk",
    ]

    for expl_list in explanations:
        for expl in expl_list:
            total_expl += 1
            total_reasons += len(expl.reasons)
            total_limitations += len(expl.limitations)

            merged = " ".join(expl.reasons).lower()
            avg_len = (sum(len(r) for r in expl.reasons) / len(expl.reasons)) if expl.reasons else 0.0

            is_factual = int(bool(expl.reasons) and any(t in merged for t in factual_tokens))
            is_simple = int(bool(expl.reasons) and avg_len <= 140)
            is_useful = int(len(expl.reasons) >= 2 and any(t in merged for t in useful_tokens))

            factual_hits += is_factual
            simple_hits += is_simple
            useful_hits += is_useful

    if total_expl == 0:
        return {
            "avg_reasons": 0.0,
            "avg_limitations": 0.0,
            "factual_rate": 0.0,
            "simple_rate": 0.0,
            "useful_rate": 0.0,
            "explanation_quality_score": 0.0,
        }

    factual_rate = factual_hits / total_expl
    simple_rate = simple_hits / total_expl
    useful_rate = useful_hits / total_expl

    return {
        "avg_reasons": total_reasons / total_expl,
        "avg_limitations": total_limitations / total_expl,
        "factual_rate": factual_rate,
        "simple_rate": simple_rate,
        "useful_rate": useful_rate,
        "explanation_quality_score": (factual_rate + simple_rate + useful_rate) / 3.0,
    }
