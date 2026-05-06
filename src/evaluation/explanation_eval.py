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
    for expl_list in explanations:
        for expl in expl_list:
            total_expl += 1
            total_reasons += len(expl.reasons)
            total_limitations += len(expl.limitations)
    if total_expl == 0:
        return {"avg_reasons": 0.0, "avg_limitations": 0.0}
    return {
        "avg_reasons": total_reasons / total_expl,
        "avg_limitations": total_limitations / total_expl,
    }
