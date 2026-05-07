"""
Evaluation runner for the DecisionFlow system.

This script ties together the various evaluation routines and can be
used from within notebooks or scripts to produce a unified report.  It
expects a dataset of clients with ground truth missing products and a
DecisionFlow implementation.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
import time

from src.decisionflow.decision_engine import run_decisionflow_from_profile
from src.decisionflow.profile_builder import build_client_profile
from .recommender_eval import evaluate_recommender
from .policy_eval import evaluate_policy
from .explanation_eval import evaluate_explanations
from .business_kpi import evaluate_business_kpis
from .agent_eval import evaluate_agent
from src.decisionflow.schemas import PolicyDecision, ExplanationResult


@dataclass
class EvalCase:
    """One evaluation case with optional policy ground truth."""

    client_id: str
    true_product: str
    raw_row: dict
    acceptable_products: Optional[Set[str]] = None
    forbidden_products: Optional[Set[str]] = None
    reason: Optional[str] = None


def run_all_evaluations(
    dataset: List[EvalCase | Tuple[str, str, dict]],
    agent_conversations: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Run recommender, policy and explanation evaluations on a dataset.

    Parameters
    ----------
    dataset: List[Tuple[str, str, dict]]
        A list of tuples ``(client_id, true_product, raw_row)``.  ``raw_row``
        should contain at least the product indicators and user features.

    Returns
    -------
    Dict[str, Any]
        Aggregated metrics for each component.
    """
    predictions: List[List[str]] = []
    truths: List[str] = []
    decisions: List[PolicyDecision] = []
    expls: List[List[ExplanationResult]] = []
    final_allowed: List[List[str]] = []
    owned_by_case: List[Set[str]] = []
    forbidden_by_case: List[Set[str]] = []
    acceptable_by_case: List[Set[str]] = []
    elapsed_ms: List[float] = []

    for item in dataset:
        if isinstance(item, EvalCase):
            case = item
        else:
            client_id, true_product, row = item
            case = EvalCase(client_id=client_id, true_product=true_product, raw_row=row)

        profile = build_client_profile(case.client_id, case.raw_row)

        t0 = time.perf_counter()
        result = run_decisionflow_from_profile(profile)
        elapsed_ms.append((time.perf_counter() - t0) * 1000.0)

        predictions.append(result["recommendations"])
        truths.append(case.true_product)
        decisions.append(result["policy"])
        expls.append(result["explanations"])

        # Policy-aware context
        final_allowed.append(list(result["policy"].allowed))
        owned_by_case.append(set(profile.current_products))
        forbidden_by_case.append(case.forbidden_products or set())
        acceptable_by_case.append(case.acceptable_products or set())

    metrics = {}
    metrics["recommender"] = evaluate_recommender(predictions, truths)
    metrics["policy"] = evaluate_policy(
        decisions,
        final_recommendations=final_allowed,
        owned_products_by_case=owned_by_case,
        forbidden_by_case=forbidden_by_case,
        acceptable_by_case=acceptable_by_case,
    )
    metrics["explanation"] = evaluate_explanations(expls)
    avg_latency = sum(elapsed_ms) / len(elapsed_ms) if elapsed_ms else None
    metrics["business_kpi"] = evaluate_business_kpis(
        recommender_metrics=metrics["recommender"],
        policy_metrics=metrics["policy"],
        explanation_metrics=metrics["explanation"],
        avg_latency_ms=avg_latency,
    )
    metrics["agent"] = evaluate_agent(agent_conversations or [])
    return metrics
